"""Regenerate the public numerical projections from a locally held research deposit."""
import argparse
from fractions import Fraction
import gzip
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_bytes())


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def save_rows(path, data):
    with path.open("w", encoding="utf-8") as stream:
        for row in data:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def chosen_pair(row):
    return [row["id"], row["logprob"]]


def position_projection(source, stock, patched):
    # These equivalence classes are reset at every position so they cannot be detokenized.
    ids = {}

    def projected(row):
        token_id = row["id"]
        return [ids.setdefault(token_id, len(ids)), row["logprob"]]

    return [projected(source), projected(stock), projected(patched),
            [projected(candidate) for candidate in stock["top_logprobs"]]]


def ifeval(root):
    for item in read(root / "data/ifeval-origin.json")["selection"]:
        key = item["key"]
        base = root / "data/ifeval"
        native = read(base / f"{key}.native.json")
        before = read(base / f"{key}.before.json")["choices"][0]["logprobs"]["content"]
        after_choice = read(base / f"{key}.http.json")["choices"][0]
        after = after_choice["logprobs"]["content"]
        plain = read(base / f"{key}.counterfactual.json")["plain"]
        positions = []
        for source, stock, patched, control in zip(native, before, after, plain, strict=True):
            assert stock["top_logprobs"] == patched["top_logprobs"]
            assert chosen_pair(source) == chosen_pair(control)
            positions.append(position_projection(source, stock, patched))
        yield {"key": key, "finish": after_choice["finish_reason"], "positions": positions}


def patch_http(root):
    for item in records(root / "extra/upstream-patch/http/http.jsonl"):
        before = item["stock_same_call"]["choices"][0]
        after = item["patched_wire"]["choices"][0]
        width = item["body"].get("top_logprobs", 0)
        stock = before.get("logprobs", {}).get("content", [])
        patched = after.get("logprobs", {}).get("content", [])
        native = item["native"]
        if width:
            assert len(stock) == len(patched) == len(native)
            positions = []
            for a, b, c in zip(native, stock, patched, strict=True):
                assert b["top_logprobs"] == c["top_logprobs"]
                assert a["token"] == c["token"]
                positions.append(position_projection(a, b, c))
        else:
            assert item["stock_same_call"] == item["patched_wire"]
            assert list(map(chosen_pair, patched)) == (
                list(map(chosen_pair, native)) if item["body"]["logprobs"] else []
            )
            positions = []
        yield {"request": item["name"], "route": item["route"], "finish": after["finish_reason"],
               "empty_display": after["message"]["content"] is None, "emitted": len(native),
               "width": width, "positions": positions}


def mean_score(rows):
    return str(sum(Fraction(r["logprob"]) for r in rows) / len(rows))


def arc(root):
    study = root / "generality/2026-11-18-arc-confirmation"
    panel = read(study / "panel.json")
    paths = sorted(study.glob("batch-*/captures.jsonl.gz"))
    draws = (row for path in paths for row in records(path))
    for panel_row, (key, group) in zip(panel, groupby(draws, key=lambda row: row["key"]), strict=True):
        assert panel_row["id"] == key
        observations = list(group)
        assert [row["draw"] for row in observations] == list(range(4))
        choices = [row["wire"]["choices"][0] for row in observations]
        outputs = [choice["message"]["content"] or "" for choice in choices]
        text_classes = [outputs.index(v) for v in outputs]
        parsed = [re.match(r"^\s*([A-D])\b", output, re.I) for output in outputs]
        native_scores = [mean_score(row["native"]) for row in observations]
        reported_scores = [mean_score(c["logprobs"]["content"]) for c in choices]
        correct = [bool(match and match[1].upper() == panel_row["answerKey"]) for match in parsed]
        yield {"key": key, "same_text_class": text_classes, "correct": correct,
               "parsed": [bool(match) for match in parsed],
               "native": native_scores, "reported": reported_scores}


def learning(root, output):
    import numpy as np

    original = root / "generality/online-learning/confirmation-sst2-t1.npz"
    with np.load(original, allow_pickle=False) as data:
        labels = data["labels"]
        scores = data["logp"].astype(np.float64)
        assert np.array_equal(scores[:, 0], scores[:, 2])
        assert np.array_equal(scores[:, 0, 0], scores[:, 1, 0])
        outcomes = scores[:, :, :, np.arange(len(labels)), labels]
        np.savez_compressed(output, labels=labels, seeds=data["seeds"],
                            initial=outcomes[:, 0, 0], selected=outcomes[:, 0, -1],
                            outer=outcomes[:, 1, -1])


def inventory(root, output):
    digest_list = (root / "SHA256SUMS").read_bytes()
    receipt = {
        "manifest_sha256": sha256(digest_list).hexdigest(),
        "manifest_entries": len(digest_list.splitlines()),
        "selected_source_sha256": {
            name: sha256((root / name).read_bytes()).hexdigest() for name in (
                "data/ifeval-expected.json",
                "extra/upstream-patch/http/http.jsonl",
                "extra/2026-11-18-wire-recovery/ambiguity.json",
                "extra/2026-11-18-review/arc-recalculation.json",
                "generality/2026-11-18-arc-confirmation/panel.json",
                "generality/multilingual-bytes/summary.json",
                "generality/online-learning/confirmation-sst2-t1.npz",
            )
        },
        "arc_capture_sha256": {
            str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "generality/2026-11-18-arc-confirmation").glob("batch-*/captures.jsonl.gz"))
        },
    }
    save(output, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deposit", type=Path)
    parser.add_argument("output", type=Path, help="catalogue directory to create for comparison")
    args = parser.parse_args()
    args.output.mkdir()
    save_rows(args.output / "ifeval.jsonl", ifeval(args.deposit))
    save_rows(args.output / "patch-http.jsonl", patch_http(args.deposit))
    save_rows(args.output / "arc.jsonl", arc(args.deposit))
    learning(args.deposit, args.output / "sst2-endpoints.npz")
    inventory(args.deposit, args.output / "source-receipt.json")


if __name__ == "__main__":
    main()
