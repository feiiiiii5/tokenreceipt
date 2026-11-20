"""Extract content-free comparison inputs from the local frozen records."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
LLAMA = Path("experiments/independent-engine")
MISTRAL = Path("experiments/generality/multilingual-bytes")
TOKENIZER = Path("experiments/data/tokenizer.json")
LLAMA_EOS = 2
LLAMA_INPUTS = (
    LLAMA / "inference-001/requests.jsonl",
    LLAMA / "inference-001/witness.tsv",
    TOKENIZER,
    LLAMA / "audit.py",
    LLAMA / "utf8_check.py",
)
MISTRAL_INPUTS = (
    MISTRAL / "run-001/requests.jsonl",
    MISTRAL / "run-001/witness.jsonl",
    MISTRAL / "audit.py",
    MISTRAL / "support/wire.py",
)


def classes(*values):
    code = {}
    return [None if value is None else code.setdefault(value, len(code)) for value in values]


def hashes(root, paths):
    return {str(path): sha256((root / path).read_bytes()).hexdigest() for path in paths}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def project_llama(root):
    sys.path.insert(0, str(root / LLAMA))
    from audit import align, read_witness
    from utf8_check import check as check_utf8, gpt2_byte_map

    requests_path, witness_path, tokenizer_path, *_ = (root / path for path in LLAMA_INPUTS)
    tokenizer = json.loads(tokenizer_path.read_text())
    byte_for_char = gpt2_byte_map()
    extra = {row["id"]: row["content"].encode() for row in tokenizer["added_tokens"]}
    vocab = {v: k for k, v in tokenizer["model"]["vocab"].items()}

    def token_bytes(token_id):
        if token_id in extra:
            return extra[token_id]
        return bytes(byte_for_char[char] for char in vocab[token_id])

    witnesses = read_witness(witness_path)
    raw = read_rows(requests_path)
    native_zero = {r["key"]: r["response"]["tokens"] for r in raw if not r["request"]["n_probs"]}
    confirmed = defaultdict(list)
    for row in check_utf8(requests_path, tokenizer_path):
        if row["fragment_utf8_error"] == "unexpected end of data" and row["next_row_combines_both_fragments"]:
            confirmed[row["prompt_key"], row["n_probs"]].append(row["missing_position"])

    responses = []
    for request in raw:
        key, width = request["key"], request["request"]["n_probs"]
        response = request["response"]
        wire = response.get("completion_probabilities", [])
        tokens = response["tokens"]
        if tokens != native_zero[key] or response["tokens_predicted"] != len(tokens):
            raise ValueError("The original response and replay have unequal token sequences or counts")
        aligned, ambiguous = align(tokens, wire)
        if ambiguous:
            raise ValueError("Ambiguous position in the llama.cpp projection")
        positions = []
        for pos, native_id in enumerate(tokens):
            reference_id, reference_logp = witnesses[key, 0, pos]
            if native_id != reference_id:
                raise ValueError("Reference is not conditional on the recorded native IDs")
            terminal_eos = response["stop_type"] == "eos" and pos == len(tokens) - 1 and native_id == LLAMA_EOS
            row = aligned.get(pos)
            returned, reference = None, None
            candidates = [] if row is None else row["top_logprobs"]
            ids = classes(native_id, None if row is None else row["id"], *(c["id"] for c in candidates))
            if row is not None:
                returned = bytes(row["bytes"])
                reference = token_bytes(row["id"])
            positions.append({
                "native_id": ids[0],
                "returned_id": ids[1],
                "returned_logp": None if row is None else row["logprob"],
                "reference_logp": reference_logp,
                "candidates": [{
                    "id": label, "logp": c["logprob"],
                    "bytes_match_vocabulary_literal": bytes(c["bytes"]) == token_bytes(c["id"]),
                    "bytes_match_vocabulary_with_eos_empty": (
                        bytes(c["bytes"]) == (b"" if c["id"] == LLAMA_EOS else token_bytes(c["id"]))
                    ),
                } for label, c in zip(ids[2:], candidates)],
                "wire_bytes_match_own_vocabulary_id": None if row is None else returned == reference,
                "wire_bytes_match_own_id_with_terminal_eos_empty": (
                    None if row is None else returned == (b"" if terminal_eos else reference)
                ),
                "terminal_eos": terminal_eos,
            })
        visible = response["content"].encode()
        responses.append({
            "key": key, "width": width, "recorded_token_count": response["tokens_predicted"],
            "reported_rows": len(wire),
            "wire_bytes_render_visible": None if not width else b"".join(bytes(r["bytes"]) for r in wire) == visible,
            "native_ids_literal_render_visible": b"".join(map(token_bytes, tokens)) == visible,
            "native_ids_stop_aware_render_visible": b"".join(
                token_bytes(token) for pos, token in enumerate(tokens) if not positions[pos]["terminal_eos"]
            ) == visible,
            "confirmed_buffer_offsets": confirmed[key, width],
            "positions": positions,
        })
    return {"source_sha256": hashes(root, LLAMA_INPUTS), "responses": responses}


def project_mistral(root):
    audit = runpy.run_path(str(root / MISTRAL / "audit.py"))
    parse = audit["parse_response"]
    requests_path, witness_path, *_ = (root / path for path in MISTRAL_INPUTS)

    def key(row):
        return row["language"], row["index"], row["rank"]

    witnesses = {key(row): row for row in read_rows(witness_path)}
    grouped = defaultdict(dict)
    for row in read_rows(requests_path):
        group = grouped[key(row)]
        mode = row["request"]["stream"]
        if mode in group:
            raise ValueError("Duplicate Mistral transport")
        group[mode] = row
    if len(grouped) != 84 or set(witnesses) != {k for k in grouped if k[2] <= 2}:
        raise ValueError("Inputs do not cover the frozen panel and its witness subset")

    pairs = []
    for k, requests in grouped.items():
        nonstream, stream = requests[False], requests[True]
        if {**nonstream["request"], "stream": True} != stream["request"]:
            raise ValueError("Paired request bodies differ in fields other than stream")
        outcome = audit["measure_wire"](nonstream, stream)
        if "exclusion" in outcome:
            raise ValueError(f"The frozen projection expects an aligned pair: {k}, {outcome['exclusion']}")
        nt, end, nu, nr = parse(nonstream)
        st, _, su, sr = parse(stream)
        omitted_eos = bool(outcome.get("omitted_streaming_eos"))
        both_eos = bool(outcome.get("both_eos"))
        reference = witnesses.get(k)
        reference_positions = None if reference is None else reference.get("positions")
        if reference_positions is not None and len(reference_positions) != nu["completion_tokens"]:
            raise ValueError("Independent path length does not agree with the frozen usage")
        ns_content = nr[:-1] if omitted_eos or both_eos else nr
        s_content = sr[:-1] if both_eos else sr
        if len(ns_content) != len(s_content):
            raise ValueError("Unaligned content rows")

        positions = []
        for pos, (nrow, srow) in enumerate(zip(ns_content, s_content)):
            seen = None if reference_positions is None else reference_positions[pos]
            canonical_byte_value = None if seen is None else bytes.fromhex(seen["raw_bytes_hex"])
            ns_bytes, s_bytes = bytes(nrow["bytes"]), bytes(srow["bytes"])
            n_byte_code, s_byte_code, ref_byte_code = classes(ns_bytes, s_bytes, canonical_byte_value)
            n_label_code, s_label_code, ref_label_code = classes(
                nrow["token"], srow["token"], None if seen is None else seen["singleton"]
            )
            positions.append({
                "nonstream_bytes": n_byte_code, "stream_bytes": s_byte_code, "canonical_bytes": ref_byte_code,
                "nonstream_replacement_bytes": ns_bytes == audit["REPLACEMENT"],
                "nonstream_contains_replacement_bytes": audit["REPLACEMENT"] in ns_bytes,
                "stream_replacement_bytes": s_bytes == audit["REPLACEMENT"],
                "nonstream_label": n_label_code, "stream_label": s_label_code, "canonical_label": ref_label_code,
                "nonstream_logp": nrow["logprob"], "stream_logp": srow["logprob"],
                "reference_logp": None if seen is None else seen["reporting_score"],
            })

        pairs.append({
            "stratum": k[0], "source_index": k[1], "rank": k[2],
            "recorded_tokens": [nu["completion_tokens"], su["completion_tokens"]],
            "reported_rows": [len(nr), len(sr)], "finish": end,
            "stream_omitted_terminal_eos": omitted_eos,
            "visible_replacement": ["\ufffd" in nt, "\ufffd" in st],
            "byte_render": [outcome["nonstream_reproduction"], outcome["stream_reproduction"]],
            "truncated_tail": [outcome["nonstream_truncated_tail"], outcome["stream_truncated_tail"]],
            "split_scalar_offsets": [
                piece["offsets"] for piece in audit["valid_scalars"](audit["byte_claims"](s_content))
            ],
            "reference_status": (
                "outside_subset" if reference is None else reference.get("abstention", "unique")
            ),
            "positions": positions,
        })
    return {"source_sha256": hashes(root, MISTRAL_INPUTS), "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--engine", choices=("llama", "mistral", "both"), default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.workspace.resolve()
    for name, project in (("llama", project_llama), ("mistral", project_mistral)):
        if args.engine in (name, "both"):
            (args.output / f"{name}.json").write_text(json.dumps(project(root), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
