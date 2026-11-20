"""Compare count, byte, and score checks using the redacted frozen projections."""

import argparse
from collections import Counter
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
LLAMA_TOLERANCE = 0.002
MISTRAL_TOLERANCE = 0.02


def judged(alarms, defects):
    results = dict.fromkeys(("true_alarms", "false_alarms", "misses", "true_negatives"), 0)
    names = {(True, True): "true_alarms", (True, False): "false_alarms",
             (False, True): "misses", (False, False): "true_negatives"}
    for alarm, defect in zip(alarms, defects, strict=True):
        results[names[alarm, defect]] += 1
    return results


def recovery(positions, outcomes):
    totals = {field: dict.fromkeys(("correct", "wrong", "abstained"), 0)
              for field in ("id", "selected_score", "id_and_selected_score")}
    selected_errors = []
    wrong_id_with_agreeing_score = 0
    for p, out in zip(positions, outcomes, strict=True):
        if out is None:
            for counts in totals.values():
                counts["abstained"] += 1
            continue
        id_match = p["native_id"] == out["id"]
        error = abs(p["reference_logp"] - out["logp"])
        score_match = error <= LLAMA_TOLERANCE
        selected_errors.append(error)
        wrong_id_with_agreeing_score += not id_match and score_match
        for field, correct in (("id", id_match), ("selected_score", score_match),
                               ("id_and_selected_score", id_match and score_match)):
            totals[field]["correct" if correct else "wrong"] += 1
    return dict(totals, maximum_score_error=max(selected_errors, default=None),
                wrong_id_with_agreeing_score=wrong_id_with_agreeing_score)


def compare_llama(projection):
    result = {"score_tolerance_nats": LLAMA_TOLERANCE, "widths": {}}
    for width in (0, 1, 5):
        cases = [r for r in projection["responses"] if r["width"] == width]
        positions = [p for r in cases for p in r["positions"]]
        base = {
            "responses": len(cases), "positions_with_native_ids": len(positions),
            "row_count": sum(r["reported_rows"] for r in cases),
        }
        result["widths"][str(width)] = base
        if not width:
            base["metadata_requested"] = False
            continue
        affected = [bool(r["confirmed_buffer_offsets"]) for r in cases]
        surviving = [p for p in positions if p["returned_id"] is not None]
        missing = [p for p in positions if p["returned_id"] is None]
        candidates = [c for p in surviving for c in p["candidates"]]
        wire_byte_defects = [not p["wire_bytes_match_own_id_with_terminal_eos_empty"] for p in surviving]
        case_bytes_literal = [any(p["wire_bytes_match_own_vocabulary_id"] is False for p in r["positions"])
                              for r in cases]
        case_bytes_stop_aware = [any(p["wire_bytes_match_own_id_with_terminal_eos_empty"] is False
                                     for p in r["positions"]) for r in cases]
        base.update({
            "response_alarms_relative_to_confirmed_buffering": {
                "row_count_vs_native_token_count": judged(
                    [r["reported_rows"] != r["recorded_token_count"] for r in cases], affected
                ),
                "wire_byte_arrays_joined_vs_visible": judged(
                    [not r["wire_bytes_render_visible"] for r in cases], affected
                ),
                "native_id_literal_rendering_vs_visible": judged(
                    [not r["native_ids_literal_render_visible"] for r in cases], affected
                ),
                "native_id_stop_aware_rendering_vs_visible": judged(
                    [not r["native_ids_stop_aware_render_visible"] for r in cases], affected
                ),
                "wire_bytes_vs_same_id_vocabulary_literal": judged(case_bytes_literal, affected),
                "wire_bytes_vs_same_id_vocabulary_terminal_eos_empty": judged(case_bytes_stop_aware, affected),
            },
            "surviving_row_byte_alarms": {
                "literal": judged([not p["wire_bytes_match_own_vocabulary_id"] for p in surviving],
                                  wire_byte_defects),
                "with_terminal_eos_empty": sum(wire_byte_defects),
            },
            "candidate_list_coverage": {
                "full_width_lists": sum(len(p["candidates"]) == width for p in surviving),
                "available_lists_without_native_id": sum(
                    all(c["id"] != p["native_id"] for c in p["candidates"]) for p in surviving
                ),
                "candidate_byte_items": len(candidates),
                "literal_candidate_byte_disagreements": sum(not c["bytes_match_vocabulary_literal"] for c in candidates),
                "candidate_byte_disagreements_with_eos_empty": sum(
                    not c["bytes_match_vocabulary_with_eos_empty"] for c in candidates
                ),
            },
            "localized_cases": [{
                "key": r["key"],
                "ordered_id_alignment_missing_positions": [
                    i for i, p in enumerate(r["positions"]) if p["returned_id"] is None
                ],
                "independently_confirmed_by_byte_buffer_positions": r["confirmed_buffer_offsets"],
                "stop_aware_merged_byte_positions": [
                    i for i, p in enumerate(r["positions"]) if p["wire_bytes_match_own_id_with_terminal_eos_empty"] is False
                ],
            } for r in cases if r["confirmed_buffer_offsets"]],
            "native_sampled_ids_available": len(positions),
            "selected_scalar_recomputable_for_omitted_rows": sum(p["reference_logp"] is not None for p in missing),
            "original_selected_scalar_available_for_omitted_rows": 0,
            "independent_historical_full_row_recovery": None,
            "value_checks_and_wire_recovery": {
                "outer": recovery(positions, [
                    None if p["returned_id"] is None else {"id": p["returned_id"], "logp": p["returned_logp"]}
                    for p in positions
                ]),
                "first_candidate": recovery(positions, [p["candidates"][0] if p["candidates"] else None
                                                       for p in positions]),
                "native_id_candidate_lookup": recovery(positions, [
                    next((c for c in p["candidates"] if c["id"] == p["native_id"]), None) for p in positions
                ]),
            },
        })
    return result


def compare_mistral(projection):
    pairs = projection["pairs"]
    content = [r for p in pairs for r in p["positions"]]
    comparisons = []
    for pair in pairs:
        positions = pair["positions"]
        split = {i for offsets in pair["split_scalar_offsets"] for i in offsets}
        markers = {i for i, p in enumerate(positions) if p["nonstream_replacement_bytes"]}
        differences = {i for i, p in enumerate(positions) if p["stream_bytes"] != p["nonstream_bytes"]}
        strict = {i for i in differences & markers & split
                  if positions[i]["nonstream_label"] == positions[i]["stream_label"]
                  and abs(positions[i]["nonstream_logp"] - positions[i]["stream_logp"]) <= MISTRAL_TOLERANCE}
        comparisons.append((pair, markers, differences, strict))

    affected = [bool(strict) for _, _, _, strict in comparisons]
    subset = [p for p in pairs if p["rank"] <= 2]
    unique = [(pair, markers, difference, strict) for pair, markers, difference, strict in comparisons
              if pair["reference_status"] == "unique"]
    reference_positions = [row for pair, _, _, _ in unique for row in pair["positions"]]
    ref_bad = [p["nonstream_bytes"] != p["canonical_bytes"] for p in reference_positions]
    ref_affected = [any(r["nonstream_bytes"] != r["canonical_bytes"] for r in p["positions"])
                    for p, _, _, _ in unique]
    strict_total = sum(len(strict) for _, _, _, strict in comparisons)
    all_diff = sum(len(diff) for _, _, diff, _ in comparisons)
    ref_diff = sum(len(diff) for _, _, diff, _ in unique)
    ref_repairs = sum(p["nonstream_bytes"] != p["canonical_bytes"] and p["stream_bytes"] == p["canonical_bytes"]
                      for p in reference_positions)
    wrong_repairs = sum(p["nonstream_bytes"] != p["stream_bytes"] and p["nonstream_bytes"] != p["canonical_bytes"]
                        and p["stream_bytes"] != p["canonical_bytes"]
                        for p in reference_positions)
    ref_strict = sum(len(strict) for _, _, _, strict in unique)
    known_strict_repairs = sum(
        pair["positions"][i]["stream_bytes"] == pair["positions"][i]["canonical_bytes"]
        and pair["positions"][i]["nonstream_bytes"] != pair["positions"][i]["canonical_bytes"]
        for pair, _, _, strict in unique for i in strict
    )

    def count_alarms(mode, allow_missing_stream_eos=False):
        return [p["reported_rows"][mode] != p["recorded_tokens"][mode] - (
            allow_missing_stream_eos and mode == 1 and p["stream_omitted_terminal_eos"]
        ) for p in pairs]

    return {
        "score_tolerance_nats": MISTRAL_TOLERANCE,
        "transport_panel": {
            "pairs": len(pairs), "content_positions": len(content),
            "generated_positions_per_transport": sum(p["recorded_tokens"][0] for p in pairs),
            "stream_eos_unavailable": sum(p["stream_omitted_terminal_eos"] for p in pairs),
            "response_alarms_relative_to_shared_origin_interior_byte_loss": {
                "nonstream_count": judged(count_alarms(0), affected),
                "stream_count_literal": judged(count_alarms(1), affected),
                "stream_count_allow_missing_terminal_eos": judged(count_alarms(1, True), affected),
                "nonstream_visible_replacement_character": judged([p["visible_replacement"][0] for p in pairs], affected),
                "nonstream_concatenated_byte_arrays_vs_visible": judged([p["byte_render"][0] != "exact" for p in pairs], affected),
                "nonstream_replacement_byte_present": judged([bool(markers) for _, markers, _, _ in comparisons], affected),
            },
            "stream_concatenated_arrays": dict(Counter(p["byte_render"][1] for p in pairs)),
            "stream_confirmed_incomplete_utf8_tail": sum(p["truncated_tail"][1] for p in pairs),
            "strict_interior_byte_loss_responses": sum(affected),
            "strict_interior_byte_loss_positions": strict_total,
            "all_nonstream_replacement_byte_positions": sum(len(markers) for _, markers, _, _ in comparisons),
            "nonstream_replacement_positions_outside_strict_criterion": sum(
                len(markers - strict) for _, markers, _, strict in comparisons
            ),
            "nonstream_positions_containing_replacement_bytes": sum(p["nonstream_contains_replacement_bytes"] for p in content),
            "contains_marker_vs_transport_differences": judged(
                [p["nonstream_contains_replacement_bytes"] for p in content],
                [p["nonstream_bytes"] != p["stream_bytes"] for p in content],
            ),
            "all_byte_diff_positions": all_diff,
            "byte_diff_positions_outside_strict_criterion": all_diff - strict_total,
            "display_label_disagreements": sum(p["nonstream_label"] != p["stream_label"] for p in content),
            "score_disagreements": sum(abs(p["nonstream_logp"] - p["stream_logp"]) > MISTRAL_TOLERANCE for p in content),
            "english_negative_control": {
                "responses": sum(p["stratum"] == "eng_Latn" for p in pairs),
                "positions": sum(len(p["positions"]) for p in pairs if p["stratum"] == "eng_Latn"),
                "nonstream_renderer_alarms": sum(p["stratum"] == "eng_Latn" and p["byte_render"][0] != "exact" for p in pairs),
                "paired_byte_differences": sum(len(diff) for p, _, diff, _ in comparisons if p["stratum"] == "eng_Latn"),
            },
        },
        "independent_witness_subset": {
            "responses_per_transport": len(subset),
            "unique_responses_per_transport": len(unique),
            "positions_per_transport": sum(p["recorded_tokens"][0] for p in subset),
            "witnessed_positions_per_transport": len(reference_positions),
            "abstained_positions_per_transport": sum(p["recorded_tokens"][0] for p in subset) - len(reference_positions),
            "abstention_reasons": dict(Counter(p["reference_status"] for p in subset if p["reference_status"] != "unique")),
            "nonstream_bytes_mismatch": sum(ref_bad),
            "nonstream_concatenated_byte_arrays_response_alarms": judged(
                [p["byte_render"][0] != "exact" for p, _, _, _ in unique], ref_affected
            ),
            "nonstream_visible_replacement_character_response_alarms": judged(
                [p["visible_replacement"][0] for p, _, _, _ in unique], ref_affected
            ),
            "nonstream_exact_replacement_byte_position_alarms": judged(
                [p["nonstream_replacement_bytes"] for p in reference_positions], ref_bad
            ),
            "nonstream_contains_replacement_byte_position_alarms": judged(
                [p["nonstream_contains_replacement_bytes"] for p in reference_positions], ref_bad
            ),
            "strict_paired_rule_position_alarms": judged(
                [i in strict for p, _, _, strict in unique for i in range(len(p["positions"]))],
                ref_bad,
            ),
            "stream_bytes_mismatch": sum(p["stream_bytes"] != p["canonical_bytes"] for p in reference_positions),
            "transport_byte_recovery_on_known_defects": {
                "correct": ref_repairs, "wrong": wrong_repairs, "abstained": sum(ref_bad) - ref_repairs - wrong_repairs,
                "all_pair_differences_with_independent_witness": ref_diff,
                "all_pair_differences_without_independent_witness": all_diff - ref_diff,
                "strict_diff_with_independently_confirmed_replacement": known_strict_repairs,
                "strict_diff_without_independent_witness": strict_total - ref_strict,
            },
            "identity_witness": "unique compatible path; HTTP selected-token labels expose no selected IDs",
            "selected_label_mismatches": {mode: sum(p[f"{mode}_label"] != p["canonical_label"]
                                                    for p in reference_positions) for mode in ("nonstream", "stream")},
            "selected_score_mismatches": {mode: sum(abs(p[f"{mode}_logp"] - p["reference_logp"]) > MISTRAL_TOLERANCE
                                                    for p in reference_positions) for mode in ("nonstream", "stream")},
            "maximum_selected_score_error": {mode: max(abs(p[f"{mode}_logp"] - p["reference_logp"])
                                                        for p in reference_positions) for mode in ("nonstream", "stream")},
            "independent_historical_full_row_recovery": None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE)
    args = parser.parse_args()
    llama = json.loads((args.inputs / "llama.json").read_text())
    mistral = json.loads((args.inputs / "mistral.json").read_text())
    print(json.dumps({"llama.cpp": compare_llama(llama), "mistral.rs": compare_mistral(mistral)}, indent=2))


if __name__ == "__main__":
    main()
