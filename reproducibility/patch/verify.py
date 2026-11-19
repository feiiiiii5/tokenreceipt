"""Apply the patch and run the eight isolated serializer checks."""
import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import typing

HERE = Path(__file__).resolve().parent
PATCH = HERE / "mlx-lm-0.32.2.patch"
RELEASE = json.loads((HERE / "release.json").read_bytes())


def apply_patch(stock, root):
    in_tree = root / "mlx_lm" / "server.py"
    data = stock.read_bytes()
    assert sha256(data).hexdigest() == RELEASE["server_sha256"]
    assert sha256(PATCH.read_bytes()).hexdigest() == RELEASE["patch_sha256"]
    in_tree.parent.mkdir(parents=True, exist_ok=True)
    in_tree.write_bytes(data)
    command = ["git", "apply", "--whitespace=error"]
    subprocess.run([*command, "--check", str(PATCH)], cwd=root, check=True)
    subprocess.run([*command, str(PATCH)], cwd=root, check=True)
    assert sha256(in_tree.read_bytes()).hexdigest() == RELEASE["patched_server_sha256"]
    return in_tree


def serializer(source):
    tree = ast.parse(source.read_bytes(), filename=source.name)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "APIHandler")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and
                  node.name == "generate_response")
    scope = vars(typing).copy()
    exec(compile(ast.Module(body=[method], type_ignores=[]), source.name, "exec"), scope)
    return scope["generate_response"]


def row(token_id, token, score):
    return {"id": token_id, "token": token, "logprob": score}


def handler(object_type="chat.completion"):
    return SimpleNamespace(
        request_id="cmpl-check", system_fingerprint="fp-local", object_type=object_type,
        requested_model="model", created=10, stream=False,
    )


def report(body):
    return body["choices"][0].get("logprobs", {}).get("content", [])


def non_sampled(body):
    projected = deepcopy(body)
    for entry in report(projected):
        for field in ("id", "token", "logprob"):
            entry.pop(field, None)
    return projected


def synthetic(old_method, new_method):
    candidate = row(1, "x", -0.5)
    inside = row(2, "Ġchosen", -1.5)
    outside = row(3, "y", -3.0)
    tie = row(4, "z", candidate["logprob"])
    eos = row(5, "<eos>", -2.0)
    candidates = [candidate, inside]
    checked = {}
    for name, actual, options, text, why in (
        ("sample-in-list", inside, candidates, " chosen", "length"),
        ("sample-outside-list", outside, candidates, "y", "length"),
        ("same-score-different-id", tie, candidates, "z", "length"),
        ("stop-only", eos, candidates, "", "stop"),
        ("text-completion", outside, candidates, "y", "length"),
    ):
        instance = handler("text_completion" if name == "text-completion" else "chat.completion")
        common = dict(
            text=text, finish_reason=why, prompt_token_count=2, completion_token_count=1,
            tokens=[actual["id"]], token_logprobs=[actual["logprob"]], top_tokens=[options],
        )
        before = old_method(instance, **common)
        after = new_method(instance, **common, token_strings=[actual["token"]])
        assert report(before) == [{**candidate, "top_logprobs": options}]
        assert report(after) == [{**actual, "top_logprobs": options}]
        assert non_sampled(before) == non_sampled(after)
        if why == "stop":
            assert after["choices"][0]["message"]["content"] is None
        checked[name] = "stock uses first alternative; patch uses sampled record"

    for name, scores, tokens in (("no-candidates", [-1.5], [2]), ("no-metadata", [], [2]),
                                 ("no-generated-positions", [], [])):
        common = dict(text=" chosen" if tokens else "", finish_reason="length" if tokens else "stop",
                      prompt_token_count=2, completion_token_count=len(tokens), tokens=tokens,
                      token_logprobs=scores, top_tokens=[])
        old = old_method(handler(), **common)
        new = new_method(handler(), **common)
        assert old == new
        assert report(new) == ([{"id": 2, "logprob": -1.5}] if scores else [])
        if not tokens:
            assert new["choices"][0]["message"]["content"] is None
        checked[name] = "identical entire response"
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock_source", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mlx-lm-patch-") as tmp:
        patched = apply_patch(args.stock_source, Path(tmp))
        orig_method, patched_method = serializer(args.stock_source), serializer(patched)
        result = {"offline": synthetic(orig_method, patched_method)}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
