"""Recompute MLX alarms, row localization and recovery from numerical evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path


RECONSTRUCTION_METHODS = (
    "reported_fields",
    "canonical_literal_top5",
    "canonical_stop_aware_top5",
    "emitted_id_top5",
    "same_emission_candidate_off",
)
CUTOFF_METHODS = ("canonical_stop_aware_top5", "emitted_id_top5")
LANES = ("original", "corrected")
CONFUSION_KEYS = ("tp", "fp", "tn", "fn", "abstained_error", "abstained_clean")
RECOVERY_KEYS = ("correct", "wrong", "abstained")
RESPONSE_ONLY_METHODS = (
    "display_text_difference",
    "row_count_alarm",
    "literal_decode_alarm",
    "skip_special_decode_alarm",
    "release_render_alarm",
)


def read_rows(path):
    with path.open() as stream:
        return list(map(json.loads, stream))


def confusion(predicted, actual):
    if predicted is None:
        return "abstained_error" if actual else "abstained_clean"
    if predicted:
        return "tp" if actual else "fp"
    return "fn" if actual else "tn"


def recovery(predicted, actual):
    if predicted is None:
        return "abstained"
    return "correct" if predicted == actual else "wrong"


def counts(counter, keys):
    return {key: counter[key] for key in keys}


def response_alarm(decisions):
    if any(decisions):
        return True
    return None if None in decisions else False


def score_discrepancy(token_id, listed_score, outer_score, cutoff):
    if listed_score is not None:
        return listed_score != outer_score
    if token_id is not None and outer_score > cutoff:
        return True
    return None


def inference(method, truth, reported, candidates, literal, stop_aware):
    if method == "reported_fields":
        return reported
    if method == "same_emission_candidate_off":
        # The local validation establishes that the independent no-list observation
        # has exactly the pairs retained by the existing numerical projection.
        return truth
    token_id = {"canonical_literal_top5": literal, "canonical_stop_aware_top5": stop_aware,
                "emitted_id_top5": truth[0]}[method]
    return token_id, candidates.get(token_id)


def analyze(evidence, observations):
    summary = {}
    for lane_index, lane in enumerate(LANES):
        pair_index = lane_index + 1
        diagnostics = {method: Counter() for method in RESPONSE_ONLY_METHODS}
        alarms = {method: Counter() for method in RECONSTRUCTION_METHODS}
        localization = {method: {"id": Counter(), "score": Counter(), "id_wrong_score_equal": Counter()}
                        for method in RECONSTRUCTION_METHODS}
        restored = {method: {field: Counter() for field in ("id", "score", "pair", "pair_on_bad_rows")}
                    for method in RECONSTRUCTION_METHODS}
        answer_recovery = {method: Counter() for method in RECONSTRUCTION_METHODS}
        cutoff_scores = {method: Counter() for method in CUTOFF_METHODS}
        cutoff_responses = {method: Counter() for method in CUTOFF_METHODS}
        panel = Counter()
        for response, observed in zip(evidence, observations, strict=True):
            assert response["key"] == observed["key"]
            positions = response["positions"]
            n = len(positions)
            panel.update(responses=1, positions=n)
            decisions = {method: [] for method in RECONSTRUCTION_METHODS}
            cutoff_decisions = {method: [] for method in CUTOFF_METHODS}
            whole_pair = {method: [] for method in RECONSTRUCTION_METHODS}
            canon = observed["canonical"]
            literal, stop_aware = [canon[variant]["classes"] or [None] * n for variant in ("literal", "stop_aware")]
            affected_response = False
            response_has_wrong_score = False
            for position, canon_literal, canon_stop in zip(positions, literal, stop_aware, strict=True):
                truth = tuple(position[0])
                reported = tuple(position[pair_index])
                candidates = dict(position[3])
                candidate_cutoff = min(candidates.values())
                bad_id = truth[0] != reported[0]
                bad_score = truth[1] != reported[1]
                equal_score_wrong_id = bad_id and not bad_score
                bad_row = bad_id or bad_score
                affected_response |= bad_row
                response_has_wrong_score |= bad_score
                panel.update(id_errors=bad_id, score_errors=bad_score, pair_errors=bad_row,
                             id_wrong_score_equal=equal_score_wrong_id, score_wrong_id_equal=bad_score and not bad_id,
                             native_id_missing_from_top5=truth[0] not in candidates)

                for method in RECONSTRUCTION_METHODS:
                    token_id, score = inference(method, truth, reported, candidates, canon_literal, canon_stop)
                    predicted_id = None if token_id is None else token_id != reported[0]
                    predicted_score = None if score is None else score != reported[1]
                    if method != "reported_fields":
                        decisions[method].extend((predicted_id, predicted_score))
                        local = localization[method]
                        local["id"][confusion(predicted_id, bad_id)] += 1
                        local["score"][confusion(predicted_score, bad_score)] += 1
                        if equal_score_wrong_id:
                            local["id_wrong_score_equal"][confusion(predicted_id, True)] += 1
                    candidate_pair = None if token_id is None or score is None else (token_id, score)
                    correct_pair = recovery(candidate_pair, truth)
                    whole_pair[method].append(correct_pair)
                    restored[method]["id"][recovery(token_id, truth[0])] += 1
                    restored[method]["score"][recovery(score, truth[1])] += 1
                    restored[method]["pair"][correct_pair] += 1
                    if bad_row:
                        restored[method]["pair_on_bad_rows"][correct_pair] += 1
                    if method in CUTOFF_METHODS:
                        mismatch = score_discrepancy(token_id, score, reported[1], candidate_cutoff)
                        cutoff_decisions[method].append(mismatch)
                        cutoff_scores[method][confusion(mismatch, bad_score)] += 1

            panel["affected_responses"] += affected_response
            diagnostics["display_text_difference"][confusion(not observed["same_display"], affected_response)] += 1
            for method in RESPONSE_ONLY_METHODS[1:]:
                diagnostics[method][confusion(observed[method][lane_index], affected_response)] += 1
            for method in RECONSTRUCTION_METHODS:
                if method != "reported_fields":
                    alarms[method][confusion(response_alarm(decisions[method]), affected_response)] += 1
                pair_outcomes = whole_pair[method]
                answer = "wrong" if "wrong" in pair_outcomes else "abstained" if "abstained" in pair_outcomes else "correct"
                answer_recovery[method][answer] += 1
            for method in CUTOFF_METHODS:
                cutoff_responses[method][confusion(response_alarm(cutoff_decisions[method]), response_has_wrong_score)] += 1

        summary[lane] = {
            "panel": dict(panel),
            "response_only_alarms": {method: counts(counter, CONFUSION_KEYS) for method, counter in diagnostics.items()},
            "reconstruction": {
                method: {
                    "response_alarm": None if method == "reported_fields" else counts(alarms[method], CONFUSION_KEYS),
                    "localization": None if method == "reported_fields" else
                    {field: counts(counter, CONFUSION_KEYS) for field, counter in localization[method].items()},
                    "recovery": {field: counts(counter, RECOVERY_KEYS) for field, counter in restored[method].items()},
                    "complete_response_recovery": counts(answer_recovery[method], RECOVERY_KEYS),
                }
                for method in RECONSTRUCTION_METHODS
            },
            "score_cutoff_extension": {
                method: {
                    "response_score_alarm": counts(cutoff_responses[method], CONFUSION_KEYS),
                    "localization": {
                        "id": counts(localization[method]["id"], CONFUSION_KEYS),
                        "score": counts(cutoff_scores[method], CONFUSION_KEYS),
                        "id_wrong_score_equal": counts(localization[method]["id_wrong_score_equal"], CONFUSION_KEYS),
                    },
                    "recovery_same_as": method,
                }
                for method in CUTOFF_METHODS
            },
        }
    summary["canonical_alignment"] = {
        variant: {
            "aligned_responses": sum(row["canonical"][variant]["classes"] is not None for row in observations),
            "exact_id_sequences": sum(row["canonical"][variant]["classes"] == [p[0][0] for p in response["positions"]]
                                      for response, row in zip(evidence, observations, strict=True)),
            "unequal_length_positions": sum(len(response["positions"])
                                            for response, row in zip(evidence, observations, strict=True)
                                            if row["canonical"][variant]["classes"] is None),
        }
        for variant in ("literal", "stop_aware")
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ifeval", type=Path, help="frozen evidence/ifeval.jsonl")
    parser.add_argument("observations", type=Path, nargs="?", default=Path(__file__).with_name("observations.jsonl"))
    args = parser.parse_args()
    print(json.dumps(analyze(read_rows(args.ifeval), read_rows(args.observations)), indent=2))
