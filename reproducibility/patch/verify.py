"""Apply the hardened patch and run all fifteen isolated serializer and streaming checks."""
import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import logging
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import typing
import uuid

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


class _StubTextStateMachine:
    @staticmethod
    def make_state(initial):
        return initial

    @staticmethod
    def split(state, text):
        return state, [(state, text)]

    @staticmethod
    def flush(state):
        return state, "", "normal"

    @staticmethod
    def discard(state):
        return state, "normal"


def load_handler_parts(source):
    tree = ast.parse(source.read_bytes(), filename=source.name)
    defs = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    cls = defs["APIHandler"]
    methods = {
        node.name: node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }
    scope = {
        **vars(typing),
        "dataclass": dataclass,
        "json": json,
        "logging": logging,
        "uuid": uuid,
        "TextStateMachine": _StubTextStateMachine,
    }
    needed_top = [
        defs[name]
        for name in (
            "ToolCallFormatter",
            "ModelDescription",
            "SamplingArguments",
            "LogitsProcessorArguments",
            "GenerationArguments",
            "CompletionRequest",
            "GenerationContext",
            "Response",
        )
    ]
    needed_cls = ast.ClassDef(
        name="APIHandlerHarness",
        bases=[],
        keywords=[],
        body=[methods["generate_response"], methods["handle_completion"]],
        decorator_list=[],
        type_params=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[*needed_top, needed_cls], type_ignores=[])
    )
    exec(compile(module, source.name, "exec"), scope)
    return scope


def serializer(source):
    return load_handler_parts(source)["APIHandlerHarness"].generate_response


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


def _run_stream_harness(scope, responses, token_map, *, top_logprobs=5, logprobs=True):
    harness_cls = scope["APIHandlerHarness"]
    wfile = io.BytesIO()

    class DummyStreamHandler(harness_cls):
        def __init__(self):
            self.request_id = "chatcmpl-stream"
            self.system_fingerprint = "fp-local"
            self.object_type = "chat.completion.chunk"
            self.requested_model = "model"
            self.created = 10
            self.stream = True
            self.stream_options = None
            self.requested_draft_model = None
            self.adapter = None
            self.temperature = 0.8
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0
            self.xtc_probability = 0.0
            self.xtc_threshold = 0.0
            self.logit_bias = None
            self.repetition_penalty = 1.0
            self.repetition_context_size = 20
            self.presence_penalty = 0.0
            self.presence_context_size = 20
            self.frequency_penalty = 0.0
            self.frequency_context_size = 20
            self.max_tokens = 16
            self.num_draft_tokens = 0
            self.logprobs = logprobs
            self.top_logprobs = top_logprobs
            self.seed = 42
            self.chat_template_kwargs = None
            self.wfile = wfile

        def _set_stream_headers(self, code):
            pass

        def end_headers(self):
            pass

    stopped = []
    ctx = SimpleNamespace(
        tool_parser=lambda txt, tools: [],
        text_sm=_StubTextStateMachine,
        initial_state="normal",
        prompt=[1, 2],
        prompt_cache_count=0,
        token_strings=lambda ids: [token_map[i] for i in ids],
        stop=lambda: stopped.append(True),
    )

    h = DummyStreamHandler()
    h.response_generator = SimpleNamespace(generate=lambda req, args, progress_callback=None: (ctx, iter(responses)))
    req = SimpleNamespace(tools=None)
    h.handle_completion(req, [])
    assert stopped == [True]
    raw_lines = [ln for ln in wfile.getvalue().decode("utf-8").splitlines() if ln.startswith("data: ")]
    assert raw_lines[-1] == "data: [DONE]"
    return [json.loads(ln[6:]) for ln in raw_lines[:-1]]


def synthetic(old_method, new_method, patched_scope):
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

    try:
        new_method(
            handler(), text="y", finish_reason="length", prompt_token_count=2, completion_token_count=1,
            tokens=[outside["id"]], token_logprobs=[outside["logprob"]], top_tokens=[candidates],
        )
        raise AssertionError("Expected ValueError when token_strings is omitted")
    except ValueError as exc:
        assert "token_strings" in str(exc)
        checked["missing-token-strings-explicit-error"] = "explicit ValueError raised"

    empty_cands = new_method(
        handler(), text="xy", finish_reason="length", prompt_token_count=2, completion_token_count=2,
        tokens=[inside["id"], outside["id"]], token_logprobs=[inside["logprob"], outside["logprob"]],
        top_tokens=[candidates, ()], token_strings=[inside["token"], outside["token"]],
    )
    assert report(empty_cands) == [{**inside, "top_logprobs": candidates}, {**outside, "top_logprobs": ()}]
    checked["empty-candidate-tuple-preserved"] = "sampled pair retained on empty candidate row"

    try:
        new_method(
            handler(), text="y", finish_reason="length", prompt_token_count=2, completion_token_count=1,
            tokens=[outside["id"]], token_logprobs=[outside["logprob"]], top_tokens=[candidates],
            token_strings=[outside["token"], "extra"],
        )
        raise AssertionError("Expected ValueError on length mismatch")
    except ValueError:
        checked["length-mismatch-strict"] = "strict ValueError raised"

    # Streaming tests (12-15): verify streaming emits sampled (id, token, logprob, top_logprobs) in lockstep.
    token_map = {1: "x", 2: " chosen", 3: "y", 5: "<eos>", 6: "à_byte0", 7: "à_byte1"}
    alt_a = (candidate, inside)
    alt_b = (candidate, row(6, "à_byte0", -0.9))

    chunks_single = _run_stream_harness(
        patched_scope,
        [
            SimpleNamespace(text="y", token=3, logprob=-3.0, finish_reason=None, top_tokens=alt_a),
            SimpleNamespace(text=" chosen", token=2, logprob=-1.5, finish_reason="length", top_tokens=alt_a),
        ],
        token_map,
        top_logprobs=2,
        logprobs=False,
    )
    rows_single = [r for ch in chunks_single for r in (ch["choices"][0].get("logprobs") or {}).get("content", [])]
    assert [r["id"] for r in rows_single] == [3, 2]
    assert [r["logprob"] for r in rows_single] == [-3.0, -1.5]
    assert [r["token"] for r in rows_single] == ["y", " chosen"]
    checked["stream-single-token-candidates"] = "streaming emits exact sampled rows even with logprobs=False"

    chunks_coalesced = _run_stream_harness(
        patched_scope,
        [
            SimpleNamespace(text="", token=6, logprob=-2.1, finish_reason=None, top_tokens=alt_b),
            SimpleNamespace(text="à", token=7, logprob=-0.4, finish_reason="length", top_tokens=alt_b),
        ],
        token_map,
        top_logprobs=2,
        logprobs=True,
    )
    assert len(chunks_coalesced) == 2
    assert chunks_coalesced[0]["choices"][0]["logprobs"]["content"] == [
        {"id": 6, "token": "à_byte0", "logprob": -2.1, "top_logprobs": list(alt_b)},
        {"id": 7, "token": "à_byte1", "logprob": -0.4, "top_logprobs": list(alt_b)},
    ]
    assert "logprobs" not in chunks_coalesced[1]["choices"][0]
    checked["stream-multibyte-coalesced-chunk"] = "empty-delta prefix token buffered and flushed in order"

    chunks_stop = _run_stream_harness(
        patched_scope,
        [
            SimpleNamespace(text="y", token=3, logprob=-3.0, finish_reason=None, top_tokens=alt_a),
            SimpleNamespace(text="", token=5, logprob=-2.0, finish_reason="stop", top_tokens=alt_a),
        ],
        token_map,
        top_logprobs=2,
        logprobs=True,
    )
    assert len(chunks_stop) == 2
    assert chunks_stop[0]["choices"][0]["logprobs"]["content"][0]["id"] == 3
    assert chunks_stop[1]["choices"][0]["finish_reason"] == "stop"
    assert chunks_stop[1]["choices"][0]["logprobs"]["content"][0]["id"] == 5
    checked["stream-stop-eos-flush"] = "stop EOS token logprob flushed on finish chunk"

    assert len(chunks_single) == 3
    assert chunks_single[2]["choices"][0]["finish_reason"] == "length"
    assert "logprobs" not in chunks_single[2]["choices"][0]
    checked["stream-length-no-duplicate-finish"] = "length finish chunk omits duplicate logprobs"

    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock_source", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mlx-lm-patch-") as tmp:
        patched = apply_patch(args.stock_source, Path(tmp))
        orig_method = serializer(args.stock_source)
        patched_scope = load_handler_parts(patched)
        patched_method = patched_scope["APIHandlerHarness"].generate_response
        result = {"offline": synthetic(orig_method, patched_method, patched_scope)}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
