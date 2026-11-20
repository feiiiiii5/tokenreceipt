"""Project the retained MLX captures onto response flags and position-local ID classes."""
import abc
import argparse
import ast
import functools
from hashlib import sha256
import json
from pathlib import Path


TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
RENDERER_SHA256 = "a6c3b727ae30989c5d65d4c1e81b8616cd3e960f02cf6e9cd5474c14926fe49f"
EOS = "<|im_end|>"


def release_renderer(source_path, tokenizer):
    """Use the release's BPE renderer without importing MLX or running inference."""
    source = source_path.read_bytes()
    assert sha256(source).hexdigest() == RENDERER_SHA256
    program = ast.parse(source)
    names = {"StreamingDetokenizer", "BPEStreamingDetokenizer", "_byte_decoder"}
    definitions = ast.Module([node for node in program.body if getattr(node, "name", None) in names], [])
    scope = {"abc": abc, "functools": functools, "List": list}
    exec(compile(definitions, str(source_path), "exec"), scope)

    class Vocabulary:
        def __len__(self):
            return tokenizer.get_vocab_size()

        def convert_ids_to_tokens(self, ids):
            return [tokenizer.id_to_token(token_id) for token_id in ids]

    return scope["BPEStreamingDetokenizer"](Vocabulary())


def render(renderer, ids, finish, eos_id):
    """Follow plain MLX generation: omit the terminal EOS text and flush the prefix."""
    if finish == "stop" and ids[-1] != eos_id:
        # The reported sequence cannot reproduce the recorded EOS stop event.
        return None
    visible_ids = ids[:-1] if finish == "stop" else ids
    if eos_id in visible_ids:
        return None
    renderer.reset()
    for token_id in visible_ids:
        renderer.add_token(token_id)
    renderer.finalize()
    return renderer.text


def pair(row):
    return [row["id"], row["logprob"]]


def project(experiments, ifeval, renderer_source):
    import tokenizers

    assert tokenizers.__version__ == "0.22.2"
    tokenizer_path = experiments / "data/tokenizer.json"
    assert sha256(tokenizer_path.read_bytes()).hexdigest() == TOKENIZER_SHA256
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id(EOS)
    assert eos_id == 2
    renderer = release_renderer(renderer_source, tokenizer)
    captures = experiments / "data/ifeval"

    for evidence in ifeval:
        key = evidence["key"]

        def load(lane):
            return json.loads((captures / f"{key}.{lane}.json").read_bytes())

        native, original, corrected, counterfactual = map(load, ("native", "before", "http", "counterfactual"))
        assert original["id"] == corrected["id"] == counterfactual["id"]
        truth = list(map(pair, native))
        assert [{"id": i, "logprob": score} for i, score in truth] == counterfactual["plain"] == counterfactual["source"]
        assert original["usage"] == corrected["usage"]
        choices = [body["choices"][0] for body in (original, corrected)]
        fields = [choice["logprobs"]["content"] for choice in choices]
        texts = [choice["message"]["content"] for choice in choices]
        finish = evidence["finish"]
        assert all(choice["finish_reason"] == finish for choice in choices)
        assert all(isinstance(text, str) for text in texts)
        assert all(not {"tool_calls", "reasoning", "reasoning_content"}.intersection(choice["message"]) for choice in choices)
        native_ids = [row["id"] for row in native]
        assert native[-1]["finish_reason"] == finish
        assert texts[0] == texts[1] == "".join(row["text"] for row in native)
        assert render(renderer, native_ids, finish, eos_id) == texts[0]

        id_classes = []
        for p, source, before, after in zip(evidence["positions"], native, *fields, strict=True):
            candidates = before["top_logprobs"]
            assert len(candidates) == 5 and len({r["id"] for r in candidates}) == 5
            assert candidates == after["top_logprobs"]
            actual = [source, before, after, *candidates]
            reference = [*p[:3], *p[3]]
            classes = {}
            for row, (token_class, score) in zip(actual, reference, strict=True):
                assert classes.setdefault(row["id"], token_class) == token_class
                assert row["logprob"] == score
            assert len(classes) == len(set(classes.values()))
            id_classes.append(classes)

        observations = {
            "key": key,
            "same_display": texts[0] == texts[1],
            "row_count_alarm": [len(rows) != body["usage"]["completion_tokens"]
                                for rows, body in zip(fields, (original, corrected), strict=True)],
            "literal_decode_alarm": [],
            "skip_special_decode_alarm": [],
            "release_render_alarm": [],
            "canonical": {},
        }
        for rows, text in zip(fields, texts, strict=True):
            reported_ids = [row["id"] for row in rows]
            observations["literal_decode_alarm"].append(tokenizer.decode(reported_ids, skip_special_tokens=False) != text)
            observations["skip_special_decode_alarm"].append(tokenizer.decode(reported_ids, skip_special_tokens=True) != text)
            observations["release_render_alarm"].append(render(renderer, reported_ids, finish, eos_id) != text)

        literal = tokenizer.encode(texts[0], add_special_tokens=False).ids
        stop_aware = [*literal, eos_id] if finish == "stop" else literal
        for name, ids in (("literal", literal), ("stop_aware", stop_aware)):
            observations["canonical"][name] = {
                "length": len(ids),
                "classes": [classes.get(token_id, -1) for classes, token_id in zip(id_classes, ids, strict=True)]
                if len(ids) == len(native_ids) else None,
            }
        yield observations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", type=Path, help="local frozen experiments directory")
    parser.add_argument("ifeval", type=Path, help="frozen evidence/ifeval.jsonl")
    parser.add_argument("renderer", type=Path, help="released MLX-LM 0.32.2 tokenizer_utils.py")
    args = parser.parse_args()
    evidence = [json.loads(line) for line in args.ifeval.read_text().splitlines()]
    for row in project(args.experiments, evidence, args.renderer):
        print(json.dumps(row, separators=(",", ":")))
