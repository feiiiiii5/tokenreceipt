"""Recompute the CTranslate2 panel from native rows, or from its numeric projection."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import struct


REGIMES = {"neutral": (1, 0), "stress": (0.7, 5), "greedy": (1, 1)}
SEEDS = (1701, 2701)
PATHS = ("batch", "stream")
_VISIBLE_BYTES = [*range(33, 127), *range(161, 173), *range(174, 256)]
_BPE_BYTES = {chr(byte): byte for byte in _VISIBLE_BYTES}
_BPE_BYTES.update({
    chr(256 + i): byte for i, byte in enumerate(sorted(set(range(256)) - set(_VISIBLE_BYTES)))
})


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def close(actual, expected):
    if actual is None or expected is None:
        return None
    return abs(actual - expected) <= 0.0002 + 0.00002 * abs(expected)


def hash_bytes(data):
    return None if data is None else sha256(data).hexdigest()


def bpe_bytes(piece):
    return bytes(_BPE_BYTES[letter] for letter in piece)


def vector_stats(arr, token_id, temperature, top_k, vocab_len):
    import numpy as np

    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    if len(values) < vocab_len or not 0 <= token_id < vocab_len:
        raise ValueError(f"Cannot map vector of length {len(values)} to ID {token_id} of {vocab_len}")
    if np.isnan(values).any() or not np.isfinite(values[token_id]):
        raise ValueError(f"Non-finite selected logit or NaN in vocabulary for ID {token_id}")
    logits = values[:vocab_len]
    if len(values) > vocab_len and np.max(values[vocab_len:]) > -1e9:
        raise ValueError("Unmapped vocabulary indices participate in normalization")
    peak_id = int(logits.argmax())
    peak = float(logits[peak_id])
    selected = float(logits[token_id])
    top = np.partition(logits, -top_k)[-top_k:] if top_k else logits
    behavior_log_z = peak / temperature + math.log(np.exp((top - peak) / temperature).sum())
    rank = 1 + int(np.count_nonzero(logits > selected))
    return {
        "selected_logit": selected,
        "top1_id": peak_id,
        "top1_logit": peak,
        "log_z": peak + math.log(np.exp(logits - peak).sum()),
        "behavior_log_z": behavior_log_z,
        "rank": rank,
        "positive_logits": int(np.count_nonzero(logits > 0)),
        "vector_width": len(values),
    }


def reference(stats, stage, regime):
    if stats is None:
        return None
    if stage == "reported":
        return stats["selected_logit"] - stats["log_z"]
    temperature, top_k = REGIMES[regime]
    if top_k and stats["rank"] > top_k:
        return None
    return stats["selected_logit"] / temperature - stats["behavior_log_z"]


def same_vector(a, b):
    import numpy as np

    return bool(np.array_equal(np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)))


def pair_hashes(row, vocabulary):
    token_id, observed = row["token_id"], row["token"]
    known, canonical_bytes = vocabulary[token_id] if 0 <= token_id < len(vocabulary) else (None, None)
    try:
        observed_bytes = bpe_bytes(observed)
    except KeyError:
        observed_bytes = None
    return {
        "piece": hash_bytes(observed.encode("utf-8")),
        "expected_piece": hash_bytes(known.encode("utf-8")) if known is not None else None,
        "piece_bytes": hash_bytes(observed_bytes) if canonical_bytes is not None else None,
        "expected_piece_bytes": hash_bytes(canonical_bytes),
    }


def result_matrix(raw_result, vectors):
    if raw_result is None or raw_result["logits_key"] is None:
        return None
    matrix = vectors[raw_result["logits_key"]]
    if matrix.ndim == 3 and len(matrix) == 1:
        matrix = matrix[0]
    if matrix.ndim != 2:
        raise ValueError(f"Unexpected batch result logits shape: {matrix.shape}")
    return matrix


def visible_agreement(visible, vocabulary, special_pieces):
    if visible is None:
        return None
    ids = visible["token_ids"]
    if any(not 0 <= i < len(vocabulary) for i in ids):
        return False
    by_id = [vocabulary[i][0] for i in ids]
    by_piece = visible["tokens"]

    def render(pieces):
        return b"".join(piece.encode("utf-8") if piece in special_pieces else bpe_bytes(piece) for piece in pieces)

    try:
        id_bytes, piece_bytes = map(render, (by_id, by_piece))
    except KeyError:
        return False
    expected = id_bytes.decode("utf-8", errors="replace")
    return (
        id_bytes == piece_bytes
        and expected == visible["decoded_ids_text"] == visible["decoded_tokens_text"]
        and expected.encode("utf-8").hex() == visible["decoded_ids_utf8_hex"] == visible["decoded_tokens_utf8_hex"]
    )


def project(raw, vectors, vocabulary, eos_ids):
    prefix_by_key = raw["metadata"]["forward_prefixes"]
    prompt_ids = {p["prompt_id"]: p["token_ids"] for p in raw["prompts"]}
    special_pieces = {piece for piece, piece_bytes in vocabulary if piece_bytes is None}
    projection = {"format": 1, "prompt_ids": list(prompt_ids), "eos_ids": sorted(eos_ids), "calls": []}
    for call in raw["calls"]:
        out = {k: call[k] for k in ("call_id", "path", "prompt_id", "regime", "seed")}
        out["failed"] = call["error"] is not None
        out["max_length"] = call["options"]["max_length"]
        out["length_penalty"] = call["options"].get("length_penalty", 1)
        out["return_end_token"] = call["options"].get("return_end_token", False)
        raw_result = call["result"]
        matrix = result_matrix(raw_result, vectors)
        out["final_ids"] = None if raw_result is None else raw_result["sequences_ids"][0]
        out["final_pieces"] = (
            None if raw_result is None else [
                hash_bytes(piece.encode("utf-8")) for piece in raw_result["sequences"][0]
            ]
        )
        visible = call.get("visible")
        out["final_visible_text"] = None if visible is None else hash_bytes(
            bytes.fromhex(visible["decoded_ids_utf8_hex"])
        )
        out["visible_text_agrees"] = visible_agreement(visible, vocabulary, special_pieces)
        out["final_vocab_rows"] = None if matrix is None else len(matrix)
        out["final_scores"] = None if raw_result is None else raw_result["scores"]
        out["steps"] = []
        prefix = list(prompt_ids[call["prompt_id"]])
        temperature, top_k = REGIMES[call["regime"]]
        args = call["options"]
        if (args["sampling_temperature"], args["sampling_topk"], args["sampling_topp"], args["max_length"]) != (
            temperature, top_k, 1, 12
        ):
            raise ValueError(f"Call {call['call_id']} has settings outside the frozen design")
        stream_callbacks = call.get("stream_callbacks") if call["path"] == "stream" else None
        out["stream_callbacks"] = None if stream_callbacks is None else len(stream_callbacks)
        raw_steps = call["steps"]
        for pos in range(len(raw_steps)):
            row = raw_steps[pos]
            token_id = row["token_id"]
            forward_key, logits_key = row["forward_key"], row["logits_key"]
            forward = None if forward_key is None else vectors[forward_key]
            logits = None if logits_key is None else vectors[logits_key]
            projected_row = {k: row[k] for k in (
                "step", "batch_id", "hypothesis_id", "token_id", "log_prob", "is_last"
            )}
            projected_row.update(pair_hashes(row, vocabulary))
            projected_row["prefix_matches"] = None if forward_key is None else prefix == prefix_by_key[forward_key]
            valid_id = 0 <= token_id < len(vocabulary)
            projected_row["forward"] = (
                vector_stats(forward, token_id, temperature, top_k, len(vocabulary))
                if valid_id and projected_row["prefix_matches"] else None
            )
            projected_row["same_event"] = (
                None if logits is None or not valid_id
                else vector_stats(logits, token_id, temperature, top_k, len(vocabulary))
            )
            projected_row["batch_vector_matches_callback"] = (
                None if matrix is None or logits is None or pos >= len(matrix)
                else same_vector(logits, matrix[pos])
            )
            projected_row["stream_matches_callback"] = (
                None if stream_callbacks is None or pos >= len(stream_callbacks)
                else all(row[k] == stream_callbacks[pos][k] for k in (
                    "step", "batch_id", "hypothesis_id", "token_id", "token", "log_prob", "is_last"
                ))
            )
            out["steps"].append(projected_row)
            prefix.append(token_id)
        projection["calls"].append(out)
    return projection


def tri_equal(left, right):
    return (left == right) if left is not None and right is not None else None


def conventional(call, eos):
    rows = call["steps"]
    ids = [row["token_id"] for row in rows]
    stop = bool(ids and ids[-1] in eos)
    natural = bool(ids and (stop or len(ids) == call["max_length"]))
    visible = rows[:-1] if stop and not call["return_end_token"] else rows
    byte_rows = [row for row in rows if row["expected_piece_bytes"] is not None]
    flags = {
        "count": not call["failed"] and natural and len(rows) <= call["max_length"] and all(
            row["step"] == i and row["batch_id"] == row["hypothesis_id"] == 0
            and row["is_last"] == (i == len(rows) - 1)
            and (row["token_id"] not in eos or i == len(rows) - 1)
            for i, row in enumerate(rows)
        ),
        "piece_text": None if not rows else all(row["piece"] == row["expected_piece"] for row in rows),
        "piece_bytes": None if not byte_rows else all(
            row["piece_bytes"] == row["expected_piece_bytes"] for row in byte_rows
        ),
        "visible_text": call["visible_text_agrees"],
        "literal_api_bytes": None,
        "final_ids": tri_equal(call["final_ids"], [row["token_id"] for row in visible]),
        "final_pieces": tri_equal(call["final_pieces"], [row["piece"] for row in visible]),
        "batch_vocab_count": tri_equal(call["final_vocab_rows"], len(rows)),
        "stream_callback_count": tri_equal(call["stream_callbacks"], len(rows)),
    }
    return flags, stop


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def aggregate_check(call):
    scores = call["final_scores"]
    rows = call["steps"]
    if not scores or not rows or any(row["log_prob"] is None for row in rows):
        return None
    total = 0.0
    for row in rows:
        total = fp32(total + row["log_prob"])
    expected = fp32(total / len(rows) ** call["length_penalty"])
    step_references = [reference(row["same_event"], "reported", call["regime"]) for row in rows]
    independent = None if None in step_references else math.fsum(step_references) / len(rows) ** call["length_penalty"]
    return {
        "observed": scores[0], "expected": expected, "matches": close(scores[0], expected),
        "same_event_expected": independent, "same_event_matches": close(scores[0], independent),
    }


def tally(counter, name, check):
    counter[f"{name}_{'unavailable' if check is None else 'pass' if check else 'fail'}"] += 1


def record_numeric(totals, maxima, row, source, regime):
    stats = row[source]
    score = row["log_prob"]
    expected = reference(stats, "reported", regime)
    behavior = reference(stats, "behavior", regime)
    score_matches = close(score, expected)
    tally(totals, source, score_matches)
    tally(totals, f"{source}_behavior", close(score, behavior))
    if score_matches is not None:
        maxima[source] = max(maxima.get(source, 0.0), abs(score - expected))
    if score is not None and behavior is not None:
        behavior_key = f"{source}_behavior"
        maxima[behavior_key] = max(maxima.get(behavior_key, 0.0), abs(score - behavior))
    if stats is not None:
        totals[f"{source}_rank_evaluable"] += 1
        totals[f"{source}_rank_gt1"] += stats["rank"] > 1
        totals[f"{source}_rank_gt5"] += stats["rank"] > 5
        totals[f"{source}_positive_logits_rows"] += stats["positive_logits"] > 0
        minima = f"{source}_min_log_z"
        maxima[minima] = min(maxima.get(minima, math.inf), stats["log_z"])
    return score_matches


def seed_pair_diagnostic(calls, prompt_ids):
    """Compare means on the existing two batch seeds, using both score meanings."""
    batch = {(c["prompt_id"], c["regime"], c["seed"]): c for c in calls if c["path"] == "batch"}
    pairs = []
    for prompt_id in prompt_ids:
        for regime in ("neutral", "stress"):
            left, right = (batch.get((prompt_id, regime, seed)) for seed in SEEDS)
            pair = {"prompt_id": prompt_id, "regime": regime}
            if any(c is None or c["failed"] or not c["steps"] or any(
                r["log_prob"] is None or r["same_event"] is None for r in c["steps"]
            ) for c in (left, right)):
                pair["complete"] = False
                pairs.append(pair)
                continue
            means = [math.fsum(r["log_prob"] for r in c["steps"]) / len(c["steps"]) for c in (left, right)]
            direct = [math.fsum(r["same_event"]["selected_logit"] for r in c["steps"]) / len(c["steps"])
                      for c in (left, right)]
            selected_winner = SEEDS[int(means[1] > means[0])]
            lookup_winner = SEEDS[int(direct[1] > direct[0])]
            pair.update({
                "complete": True, "draw_ids": [left["call_id"], right["call_id"]], "seed_order": list(SEEDS),
                "positions": [len(left["steps"]), len(right["steps"])],
                "selected_means": means, "lookup_means": direct, "selected_winner_seed": selected_winner,
                "lookup_winner_seed": lookup_winner, "changed_winner": selected_winner != lookup_winner,
                "selected_tie": means[0] == means[1], "lookup_tie": direct[0] == direct[1],
                "same_final_ids": tri_equal(left["final_ids"], right["final_ids"]),
                "same_visible_text": tri_equal(left["final_visible_text"], right["final_visible_text"]),
            })
            pairs.append(pair)
    complete = [pair for pair in pairs if pair["complete"]]
    return {
        "planned": len(pairs), "complete": len(complete),
        **{k: sum(pair[k] is True for pair in complete) for k in (
            "changed_winner", "selected_tie", "lookup_tie", "same_final_ids", "same_visible_text"
        )},
        "pairs": pairs,
    }


def audit(projection):
    if projection["format"] != 1 or len(set(projection["prompt_ids"])) != 6:
        raise ValueError("Expected the frozen six-prompt panel and projection format 1")
    eos = set(projection["eos_ids"])
    calls = projection["calls"]
    groups = defaultdict(Counter)
    maxima = defaultdict(dict)
    draw_details = {}
    attempted = set()
    control = None
    plan = {
        (path, prompt, regime, seed)
        for path in PATHS for prompt in projection["prompt_ids"] for regime in REGIMES
        for seed in ((SEEDS[0],) if regime == "greedy" else SEEDS)
    }
    for call in calls:
        call_id = call["call_id"]
        path, regime = call["path"], call["regime"]
        cell = path, call["prompt_id"], regime, call["seed"]
        if cell not in plan or cell in attempted or call_id in draw_details:
            raise ValueError(f"Repeated or unplanned draw {call_id}: {cell}")
        attempted.add(cell)
        group = f"{path}/{regime}"
        totals = groups[group]
        totals["attempted"] += 1
        totals["failed"] += call["failed"]
        baseline, eos_stop = conventional(call, eos)
        for check, value in baseline.items():
            tally(totals, check, value)
        totals["eos_draws"] += eos_stop
        rows = call["steps"]
        totals["positions"] += len(rows)
        totals["positions_with_literal_bytes"] += 0
        totals["derived_byte_positions"] += sum(row["expected_piece_bytes"] is not None for row in rows)
        totals["eos_positions"] += sum(row["token_id"] in eos for row in rows)
        fit = []
        hits = []
        for pos, row in enumerate(rows):
            same_ok = record_numeric(totals, maxima[group], row, "same_event", regime)
            forward_ok = record_numeric(totals, maxima[group], row, "forward", regime)
            tally(totals, "forward_prefix", row["prefix_matches"])
            tally(totals, "batch_vector_matches_callback", row["batch_vector_matches_callback"])
            tally(totals, "stream_matches_callback", row["stream_matches_callback"])
            if same_ok is False or forward_ok is False:
                hits.append({"position": pos, "same_event": same_ok, "forward": forward_ok})
            fit.append(same_ok if path == "batch" else forward_ok)
            stats = row["same_event"]
            ready = not call["failed"] and baseline["piece_bytes"] is True and all(
                value is not False for value in baseline.values()
            )
            if control is None and ready and same_ok and stats["rank"] > 1:
                replacement = stats["top1_logit"] - stats["log_z"]
                raw_expected = reference(stats, "reported", regime)
                if close(replacement, raw_expected) is False:
                    altered = deepcopy(call)
                    altered["steps"][pos]["log_prob"] = replacement
                    projected_checks, _ = conventional(altered, eos)
                    control = {
                        "call_id": call_id, "position": pos, "sampled_id": row["token_id"],
                        "replacement_source_id": stats["top1_id"],
                        "original_score": row["log_prob"], "replacement_score": replacement,
                        "expected_score": raw_expected,
                        "numeric_control_rejected": not close(replacement, raw_expected),
                        "conventional_unchanged": baseline == projected_checks,
                        "conventional": projected_checks,
                    }
        complete = bool(fit) and baseline["count"] and all(ok is not None for ok in fit)
        totals["complete_numeric_draws"] += complete
        totals["matching_numeric_draws"] += complete and all(fit)
        totals["numeric_fail_draws"] += bool(hits)
        combined = aggregate_check(call)
        tally(totals, "aggregate", combined["matches"] if combined is not None else None)
        tally(totals, "aggregate_same_event", combined["same_event_matches"] if combined is not None else None)
        draw_details[call_id] = {
            "path": path, "regime": regime, "failed": call["failed"], "positions": len(rows),
            "numeric_disagreements": hits, "conventional": baseline, "aggregate": combined,
        }
    return {
        "planned": len(plan), "attempted": len(attempted),
        "unattempted": sorted(plan - attempted), "groups": groups,
        "deviation_nats_and_log_z": maxima, "draws": draw_details,
        "score_substitution_control": control,
        "literal_vocab_lookup_diagnostic": seed_pair_diagnostic(calls, projection["prompt_ids"]),
    }


def read_raw(directory):
    import numpy as np

    raw = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    vocabulary, eos_ids = read_vocabulary(directory, raw["metadata"])
    with np.load(directory / "vectors.npz", allow_pickle=False) as vectors:
        return project(raw, vectors, vocabulary, eos_ids)


def read_vocabulary(directory, metadata):
    source = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
    pieces = source["tokenizer_tokens"]
    if pieces != source["ct2_tokens"]:
        raise ValueError("Converted and source token-ID maps disagree")
    if source["byte_decoder"] != _BPE_BYTES:
        raise ValueError("Captured and independently constructed ByteLevel decoder tables disagree")
    specials = set(source["special_token_ids"])
    eos = source["eos_token_id"]
    if eos != metadata["eos_token_id"] or eos not in specials:
        raise ValueError("The run and the pinned vocabulary must identify the same terminal EOS")
    vocabulary = [(piece, None if i in specials else bpe_bytes(piece)) for i, piece in enumerate(pieces)]
    return vocabulary, {eos}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--raw", type=Path, help="Directory containing captured native records and vectors")
    inputs.add_argument("--projection", type=Path, help="Compact numeric projection, no NumPy needed")
    parser.add_argument("--write-projection", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    projection = read_raw(args.raw) if args.raw else json.loads(args.projection.read_text(encoding="utf-8"))
    if args.write_projection:
        write_json(args.write_projection, projection)
    report = audit(projection)
    if args.output:
        write_json(args.output, report)
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
