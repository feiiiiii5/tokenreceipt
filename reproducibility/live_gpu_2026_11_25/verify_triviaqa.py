"""Recompute TriviaQA metrics from the published text-free rows."""
import json
from pathlib import Path
from statistics import mean
import sys


def calibration(rows, confidence_key):
    residuals = [0.0] * 10
    brier = 0.0
    for row in rows:
        confidence = row[confidence_key]
        index = next(i for i in range(10) if confidence <= (i + 1) / 10)
        residual = confidence - row["correct"]
        residuals[index] += residual
        brier += residual**2
    return {"ece_10bin": sum(map(abs, residuals)) / len(rows), "brier": brier / len(rows)}


def verify_cohort(cohort, split_sizes):
    pilot = cohort["pilot_rows"]
    rows = cohort["confirmation_rows"]
    assert len(pilot) == split_sizes["pilot"]
    assert len(rows) == split_sizes["confirmation"]
    pilot_ranks = {row["rank"] for row in pilot}
    confirmation_ranks = {row["rank"] for row in rows}
    assert len(pilot_ranks) == len(pilot) and len(confirmation_ranks) == len(rows)
    assert len(pilot_ranks & confirmation_ranks) == split_sizes["overlap"] == 0

    expected = cohort["expected_summary"]["confirmation_buggy"]
    counts = {
        "n_questions": len(rows),
        "positions": sum(row["n_tokens"] for row in rows),
        "wrong_ids": sum(row["wrong_ids"] for row in rows),
        "wrong_scores": sum(row["wrong_scores"] for row in rows),
    }
    assert all(value == expected[key] for key, value in counts.items())
    assert abs(mean(row["correct"] for row in rows) - expected["accuracy"]) < 0.00005

    keys = {"native": "conf_native", "buggy": "conf_buggy"}
    metrics = {}
    for arm, key in keys.items():
        metrics[arm] = calibration(rows, key)
        for metric, value in metrics[arm].items():
            assert abs(value - expected["calibration"][metric][arm]) < 0.000001

    wrong = [row for row in rows if row["correct"] == 0]
    routing = {}
    for threshold in (0.80, 0.90):
        name = f"fixed_{threshold:.2f}"
        reference = expected["routing"][name]
        assert reference["tau"] == threshold and reference["wrong_total"] == len(wrong)
        counts_by_arm = {arm: sum(row[key] >= threshold for row in wrong) for arm, key in keys.items()}
        assert all(reference[f"{arm}_false_accepts"] == count for arm, count in counts_by_arm.items())
        routing[name] = {"wrong_total": len(wrong), **counts_by_arm}
    return {"pilot_items": len(pilot), **counts, "calibration": metrics, "false_accepts": routing}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("wp3a_triviaqa_routing.json")
    projection = json.loads(path.read_text(encoding="utf-8"))
    results = {
        name: verify_cohort(cohort, projection["split_sizes"])
        for name, cohort in projection["cohorts"].items()
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
