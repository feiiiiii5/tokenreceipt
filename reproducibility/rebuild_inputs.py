"""Verify pinned upstream inputs and reconstruct public experiment selections locally."""
import argparse
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path
from urllib.request import urlopen

from verify import EVIDENCE, HERE, load, records


PROVENANCE = HERE / "provenance"
SST = "https://huggingface.co/datasets/nyu-mll/glue/resolve/a3f3ab759b9d7018046ce361f115ef38276d2e1e/sst2"
SST_HASH = {
    "train": "66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229",
    "validation": "a1371f3b3a7b0bcefa8388799a9359dc3ce76c349cc0079507a7991364fd2a9b",
}


def acquire(folder, filename, url, digest):
    path = folder / filename
    if path.exists():
        data = path.read_bytes()
    else:
        with urlopen(url, timeout=90) as response:
            data = response.read()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f"Upstream SHA-256 differs for {filename}")
    if not path.exists():
        path.write_bytes(data)
    return path


def output(folder, name, rows):
    with (folder / name).open("x", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def ifeval(folder):
    manifest = load(PROVENANCE / "ifeval-origin.json")
    ranks = load(PROVENANCE / "ifeval-rank-digests.json")
    source = acquire(folder, "ifeval-population.jsonl", manifest["source"], manifest["source_sha256"])
    population = list(records(source))
    assert len(population) == manifest["population"] == len({row["key"] for row in population})
    assert {str(row["key"]) for row in population} == set(ranks)
    selected = sorted(population, key=lambda row: ranks[str(row["key"])])[:64]
    assert [{"key": r["key"], "prompt_sha256": sha256(r["prompt"].encode()).hexdigest()} for r in selected] == manifest["selection"]
    output(folder, "ifeval-panel.json", [{k: r[k] for k in ("key", "prompt")} for r in selected])


def arc(folder):
    from pyarrow.parquet import read_table

    manifest = load(PROVENANCE / "arc-origin.json")
    source = acquire(folder, "arc-easy.parquet", manifest["url"], manifest["sha256"])
    rows = read_table(source).to_pylist()
    assert len(rows) == manifest["total"]
    eligible = [row for row in rows if row["choices"]["label"] == list("ABCD") and row["answerKey"] in "ABCD"]
    assert len(eligible) == manifest["eligible"]
    panel = sorted(eligible, key=lambda row: sha256(("arc-confidence-v1/" + row["id"]).encode()).digest())[160:]
    assert [row["id"] for row in panel] == [row["key"] for row in records(EVIDENCE / "arc.jsonl")]
    output(folder, "arc-confirmation-panel.json", panel)


def flores(folder):
    manifest = load(PROVENANCE / "flores-origin.json")
    digests = load(PROVENANCE / "flores-line-rank-digests.json")
    expected = load(EVIDENCE / "multilingual-summary.json")["pairs"]
    n = manifest["all_rows_per_language"]
    assert len(digests) == n
    selected = sorted(range(1, n + 1), key=lambda i: (digests[i - 1], i))[:12]
    base = f'{manifest["mirror"]}/resolve/{manifest["revision"]}/devtest'
    panel = []
    for language, digest in manifest["source_file_sha256"].items():
        name = f"{language}.devtest"
        path = acquire(folder, name, f"{base}/{name}", digest)
        rows = path.read_text(encoding="utf-8").removesuffix("\n").split("\n")
        assert len(rows) == n and all(rows)
        panel.extend({"language": language, "index": i, "rank": rank, "source_text": rows[i - 1],
                      "independent_witness": rank <= 2}
                     for rank, i in enumerate(selected, 1))
    assert [(r["language"], r["index"], r["rank"]) for r in panel] == [
        (r["language"], r["index"], r["rank"]) for r in expected
    ]
    output(folder, "flores-panel.json", panel)


def sst2(folder):
    from pyarrow.parquet import read_table
    from unicodedata import normalize as unicode_normalize

    def normalize(text):
        return " ".join(unicode_normalize("NFKC", text).casefold().split())

    ranks = load(PROVENANCE / "sst2-row-order.json")
    for split, digest in SST_HASH.items():
        filename = f"{split}-00000-of-00001.parquet"
        path = acquire(folder, f"sst2-{split}.parquet", f"{SST}/{filename}", digest)
        unique = {}
        for row in sorted(read_table(path).to_pylist(), key=lambda item: item["idx"]):
            unique.setdefault(normalize(row["sentence"]), row)
        by_id = {row["idx"]: row for row in unique.values()}
        assert set(by_id) == set(ranks[split]) and len(by_id) == len(ranks[split])
        output(folder, f"sst2-ranked-{split}.json", [by_id[i] for i in ranks[split]])


def auroc(rows, score):
    positive = sum(row["gold_label"] for row in rows)
    negatives_seen = 0
    favorable_pairs = 0.0
    for _, group in groupby(sorted(rows, key=score), key=score):
        tied = list(group)
        yes = sum(row["gold_label"] for row in tied)
        no = len(tied) - yes
        favorable_pairs += yes * (negatives_seen + no / 2)
        negatives_seen += no
    return favorable_pairs / (positive * (len(rows) - positive))


def indra(folder):
    from math import isclose

    origin = load(PROVENANCE / "indra-origin.json")
    path = acquire(folder, "indra-rows.jsonl", origin["url"], origin["sha256"])
    rows = list(records(path))
    expected = load(EVIDENCE / "external-use-summary.json")["public_artifacts"]["indra"]
    assert len({row["row_id"] for row in rows}) == len(rows) == expected["rows"]
    assert len({row["paper_statement_hash"] for row in rows}) == expected["unique_statement_clusters"]
    assert all(row["both_observed"] and row["contract_ok"] and row["n_generated"] == 1 for row in rows)
    def margin(row):
        return row["log_p_correct"] - row["log_p_incorrect"]
    assert all(isclose(margin(row), row["delta_logit"], rel_tol=1e-15, abs_tol=1e-15) for row in rows)
    got = {"rows": len(rows), "margin_auc": auroc(rows, lambda row: row["delta_logit"]),
           "verdict_only_auc": auroc(rows, lambda row: row["verdict"] == "correct")}
    assert all(isclose(v, expected[k]) for k, v in got.items())
    print(json.dumps(got, indent=2))


def main():
    commands = {"ifeval": ifeval, "arc": arc, "flores": flores, "sst2": sst2, "indra": indra}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("material", choices=commands)
    parser.add_argument("output", type=Path, help="a local directory for downloaded and reconstructed files")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    commands[args.material](args.output)
    print(f"Verified {args.material} against the pinned inputs and recorded selection or result.")


if __name__ == "__main__":
    main()
