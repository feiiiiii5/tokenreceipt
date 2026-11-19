"""Check that MLX-LM's serializer retains the sampled ID and raw log probability."""

import __future__
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace


def load_serializer(source):
    tree = ast.parse(source.read_bytes(), filename=str(source))
    handler = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "APIHandler"
    )
    method = next(
        node for node in handler.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_response"
    )
    code = compile(
        ast.Module(body=[method], type_ignores=[]), str(source), "exec",
        flags=__future__.annotations.compiler_flag, dont_inherit=True,
    )
    namespace = {}
    exec(code, namespace)
    return namespace["generate_response"]


def serialize(serializer, fixture, with_candidates):
    state = SimpleNamespace(
        request_id="fixture", system_fingerprint="fixture",
        object_type="chat.completion", requested_model="fixture", created=0, stream=False,
    )
    sampled = fixture["sampled"]
    options = {"top_tokens": [deepcopy(fixture["candidates"])]} if with_candidates else {}
    response = serializer(
        state, sampled["text"], "length", prompt_token_count=0,
        completion_token_count=1, tokens=[sampled["id"]],
        token_logprobs=[sampled["logprob"]], **options,
    )
    choice, = response["choices"]
    entry, = choice["logprobs"]["content"]
    return entry


def selected_pair(entry):
    return entry["id"], entry["logprob"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Trusted MLX-LM server.py")
    parser.add_argument(
        "--mode", choices=("assert", "demonstrate"), default="assert",
        help="Fail on a contract violation, or verify the known released behavior",
    )
    args = parser.parse_args()
    serializer = load_serializer(args.source)
    fixtures = json.loads(Path(__file__).with_name("fixture.json").read_bytes())

    contract_passes = True
    history_matches = True
    for fixture in fixtures:
        sampled = selected_pair(fixture["sampled"])
        omitted = serialize(serializer, fixture, False)
        width_five = serialize(serializer, fixture, True)
        omitted_pair = selected_pair(omitted)
        width_five_pair = selected_pair(width_five)
        alternatives_preserved = (
            "top_logprobs" not in omitted and width_five.get("top_logprobs") == fixture["candidates"]
        )
        case_passes = omitted_pair == width_five_pair == sampled and alternatives_preserved
        contract_passes &= case_passes
        history_matches &= (
            omitted_pair == sampled
            and width_five_pair == selected_pair(fixture["observed_k5"])
            and alternatives_preserved
        )
        status = "PASS" if case_passes else "FAIL"
        print(
            f"{fixture['name']}: {status} | sampled {sampled} | "
            f"k omitted {omitted_pair} | k=5 {width_five_pair} | "
            f"candidates preserved: {alternatives_preserved}"
        )
    if args.mode == "demonstrate":
        if history_matches and not contract_passes:
            print("Reproduced the recorded substitution and passing control.")
            return 0
        print("The executed behavior differs from the recorded release.")
        return 1
    return 0 if contract_passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
