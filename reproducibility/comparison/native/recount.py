"""Compare CTranslate2 checks on the existing anonymized numeric projection."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import struct
from zipfile import ZipFile


VOCABULARY = 49152
ARCHIVE_PREFIX = "supplement/ctranslate2/"


def score_agrees(observed, expected):
    if observed is None or expected is None:
        return None
    return abs(observed - expected) <= 0.0002 + 0.00002 * abs(expected)


def selected_score(stats):
    return None if stats is None else stats["selected_logit"] - stats["log_z"]


def compare(observed, expected):
    return None if observed is None or expected is None else observed == expected


def one_vocabulary_row(row):
    return row["elements"] == VOCABULARY and row["shape"] in ([VOCABULARY], [1, VOCABULARY])


def tally(totals, checks):
    for name, result in checks.items():
        totals[name][{True: "pass", False: "alarm", None: "abstain"}[result]] += 1


def float32(number):
    return struct.unpack("f", struct.pack("f", number))[0]


def response_checks(call, eos_ids):
    rows = call["steps"]
    n = len(rows)
    ends_in_eos = bool(rows) and rows[-1]["token_id"] in eos_ids
    visible_rows = rows[:-1] if ends_in_eos and not call["return_end_token"] else rows
    valid_count = (not call["failed"] and 0 < n <= call["max_length"]
                   and (n == call["max_length"] or ends_in_eos)
                   and all(row["step"] == i and row["is_last"] == (i == n - 1)
                           and row["batch_id"] == row["hypothesis_id"] == 0
                           and (row["token_id"] not in eos_ids or i == n - 1)
                           for i, row in enumerate(rows)))
    aggregate_agrees = None
    if n and call["final_scores"] and all(row["log_prob"] is not None for row in rows):
        accrued = 0.0
        for row in rows:
            accrued = float32(accrued + row["log_prob"])
        from_callbacks = float32(accrued / n ** call["length_penalty"])
        aggregate_agrees = score_agrees(call["final_scores"][0], from_callbacks)
    return {
        "terminal_count_and_steps": valid_count,
        "visible_text": call["visible_text_agrees"],
        "final_sequence_count": compare(len(call["final_ids"]) if call["final_ids"] is not None else None,
                                      len(visible_rows)),
        "final_vector_count": compare(call["final_vocab_rows"], n),
        "stream_callback_count_same_object": compare(call["stream_callbacks"], n),
        "final_aggregate_vs_callback_scalars": aggregate_agrees,
    }


def position_checks(call, position, row):
    same = row["same_event"]
    forward = row["forward"] if row["prefix_matches"] else None
    final_size = call["final_vocab_rows"]
    final_agrees = row["batch_vector_matches_callback"]
    final_has_row = None if final_size is None else position < final_size
    final_score = selected_score(same) if final_has_row and final_agrees else None
    final_id = None if call["final_ids"] is None or position >= len(call["final_ids"]) else call["final_ids"][position]
    final_piece = (None if call["final_pieces"] is None or position >= len(call["final_pieces"])
                   else call["final_pieces"][position])
    return {
        "canonical_piece": compare(row["piece"], row["expected_piece"]),
        "canonical_bytes_derived_from_piece": compare(row["piece_bytes"], row["expected_piece_bytes"]),
        "literal_bytes": None,
        "callback_vector_width": None if same is None else same["vector_width"] == VOCABULARY,
        "scalar_vs_callback_logsoftmax": score_agrees(row["log_prob"], selected_score(same)),
        "scalar_vs_exact_prefix_forward_logsoftmax": score_agrees(row["log_prob"], selected_score(forward)),
        "scalar_vs_literal_callback_logit": score_agrees(row["log_prob"], None if same is None else same["selected_logit"]),
        "final_id_same_origin": compare(row["token_id"], final_id),
        "final_piece_same_origin": compare(row["piece"], final_piece),
        "final_vector_available": final_has_row,
        "final_vector_equals_callback": final_agrees,
        "scalar_vs_final_logsoftmax": score_agrees(row["log_prob"], final_score),
        "iterator_vs_same_callback_object": row["stream_matches_callback"],
        "independent_next_decoder_id": None,
    }


def singleton_panel(evidence):
    totals = {}
    for call in evidence["calls"]:
        response, positions = totals.setdefault(call["path"], (defaultdict(Counter), defaultdict(Counter)))
        tally(response, response_checks(call, evidence["eos_ids"]))
        for position, row in enumerate(call["steps"]):
            tally(positions, position_checks(call, position, row))
    return {path: {"draws": response, "positions": positions} for path, (response, positions) in totals.items()}


def coverage_panel(cases):
    totals = {}
    observations = []
    for case in cases:
        group = "singleton" if case["copies"] == 1 else "split" if case["split"] else "joint"
        response, positions = totals.setdefault(group, (defaultdict(Counter), defaultdict(Counter)))
        rows = case["rows"]
        for item in range(case["copies"]):
            owned = [row for row in rows if row["item"] == item]
            tally(response, {"item_step_and_final_count": [r["step"] for r in owned] == [0, 1]
                             and all(r["final_count"] == 2 for r in owned)})
        for row in rows:
            scalar_agrees = score_agrees(row["log_prob"], row["final_chosen_logp"])
            shape_agrees = one_vocabulary_row(row)
            final_has_row = row["final_vocab"] == VOCABULARY
            tally(positions, {
                "final_id_same_origin": compare(row["token_id"], row["final_id"]),
                "final_piece_same_origin": compare(row["piece"], row["final_piece"]),
                "rendered_text_in_projection": None,
                "literal_bytes": None,
                "vector_object_presence": row["optional_present"],
                "vector_nonempty": row["elements"] > 0,
                "strict_single_callback_vector": shape_agrees,
                "scalar_vs_callback_logsoftmax": None,
                "scalar_vs_final_logsoftmax": scalar_agrees,
                "final_vector_available": final_has_row,
                "independent_next_decoder_id": None,
            })
            if not shape_agrees:
                observations.append({"group": group, "seed": case["seed"], "item": row["item"],
                                     "step": row["step"], "field": "logits", "shape": row["shape"],
                                     "elements": row["elements"], "nonempty_alarm": row["elements"] == 0,
                                     "final_vector_available": final_has_row})
    return {
        "groups": {group: {"items": response, "positions": positions} for group, (response, positions) in totals.items()},
        "localized_shape_alarms": observations,
    }


def alignment_confirmation(report):
    totals = defaultdict(Counter)
    for case in report["runs"]:
        for step in case["steps"]:
            tally(totals, {"item_routing_identifiable_from_distinct_final_values":
                           None if step["compared_final_rows"]["equal"] else True})
            for row in (step["first_callback"], step["second_callback"]):
                tally(totals, {"repeat_strict_single_callback_vector": one_vocabulary_row(row),
                               "repeat_vector_nonempty": row["elements"] > 0})
            for callback_row, final_row in zip(step["first_callback"]["rows"], step["final_rows"], strict=True):
                match = callback_row["by_final_item"][final_row["batch_id"]]
                tally(totals, {"matrix_row_equals_corresponding_final_vector":
                               callback_row["sha256"] == final_row["sha256"]
                               and match["equal"] and match["max_abs_error"] == 0})
    return totals


def decoder_witness(native, response):
    totals = defaultdict(Counter)
    callbacks = response["callback_rows"]
    assert len(native) == len(callbacks)
    assert native[0]["token_id"] == response["prompt_ids"][-1]
    first_step = len(response["prompt_ids"]) - 1
    assert all(row["absolute_step"] == first_step + i for i, row in enumerate(native))
    observed = native[1:]
    for pos, row in enumerate(callbacks):
        expected = observed[pos]["token_id"] if pos < len(observed) else None
        tally(totals, {"callback_id_vs_next_decoder_id": compare(row["token_id"], expected),
                       "final_id_vs_next_decoder_id": compare(response["final_sequence_ids"][pos], expected)})
    return {"positions": totals, "off_first_witnessed": sum(row["rank"] > 1 for row in callbacks[:len(observed)]),
            "outside_top_five_witnessed": sum(row["rank"] > 5 for row in callbacks[:len(observed)])}


def injection(evidence, frozen):
    call = next(call for call in evidence["calls"] if call["call_id"] == frozen["call_id"])
    altered = deepcopy(call)
    pos = frozen["position"]
    row = altered["steps"][pos]
    same = row["same_event"]
    original_score = row["log_prob"]
    replacement_score = same["top1_logit"] - same["log_z"]
    assert same["rank"] > 1 and same["top1_id"] == frozen["replacement_source_id"]
    assert replacement_score == frozen["replacement_score"] and original_score == frozen["original_score"]
    row["log_prob"] = replacement_score
    clean_checks = position_checks(call, pos, call["steps"][pos])
    altered_checks = position_checks(altered, pos, row)
    responses_clean = response_checks(call, evidence["eos_ids"])
    responses_altered = response_checks(altered, evidence["eos_ids"])
    recovered_same = selected_score(same)
    recovered_forward = selected_score(row["forward"])
    return {
        "call_id": call["call_id"], "position": pos, "field": "log_prob",
        "token_id": row["token_id"], "replacement_source_id": same["top1_id"],
        "original_score": original_score, "replacement_score": replacement_score,
        "clean": {"response_checks": responses_clean, "position_checks": clean_checks},
        "injected": {"response_checks": responses_altered, "position_checks": altered_checks,
                     "all_position_alarms_by_check": positions_with_alarms(altered)},
        "selected_scalar_recovery": {
            "callback": {"value": replacement_score, "within_tolerance_of_original": score_agrees(replacement_score, original_score)},
            "same_event_callback_or_final_vector": {"value": recovered_same,
                "within_tolerance_of_original": score_agrees(recovered_same, original_score)},
            "exact_prefix_forward": {"value": recovered_forward,
                "within_tolerance_of_original": score_agrees(recovered_forward, original_score)},
        },
    }


def positions_with_alarms(call):
    alarms = defaultdict(list)
    for position, row in enumerate(call["steps"]):
        for name, result in position_checks(call, position, row).items():
            if result is False:
                alarms[name].append(position)
    return alarms


def recount(read):
    evidence = read("evidence.json")
    return {
        "singleton_panel": singleton_panel(evidence),
        "callback_coverage": coverage_panel(read("batch-control.json")),
        "matrix_alignment_confirmation": alignment_confirmation(read("batch-alignment.json")),
        "independent_decoder_witness_separate_call": decoder_witness(read("witness/decoder-inputs.json"),
                                                                      read("witness/run-001/response.json")),
        "existing_score_only_control": injection(evidence, read("audit.json")["score_substitution_control"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to ctranslate2/ or to an archived evidence ZIP")
    args = parser.parse_args()
    if args.source.is_dir():
        results = recount(lambda name: json.loads((args.source / name).read_bytes()))
    else:
        with ZipFile(args.source) as archive:
            results = recount(lambda name: json.loads(archive.read(ARCHIVE_PREFIX + name)))
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
