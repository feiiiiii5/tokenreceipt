"""Offline pure-stdlib verification of downstream_3b, CT2 distinct-batch, and cross-arch live audits."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median
import sys

HERE = Path(__file__).resolve().parent
D3B = HERE / "downstream_3b"
REPO = HERE.parent


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ece_10bin(confs: list[float], accs: list[int]) -> float:
    n = len(confs)
    ece = 0.0
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        if i == 0:
            idx = [j for j, c in enumerate(confs) if lo <= c <= hi]
        else:
            idx = [j for j, c in enumerate(confs) if lo < c <= hi]
        if idx:
            ece += (len(idx) / n) * abs(mean(accs[j] for j in idx) - mean(confs[j] for j in idx))
    return ece


def brier_score(confs: list[float], accs: list[int]) -> float:
    return sum((c - a) ** 2 for c, a in zip(confs, accs, strict=True)) / len(accs)


def auroc_score(confs: list[float], accs: list[int]) -> float:
    pos = [c for c, a in zip(confs, accs, strict=True) if a == 1]
    neg = [c for c, a in zip(confs, accs, strict=True) if a == 0]
    wins = 0.0
    for p in pos:
        wins += sum(1.0 for x in neg if p > x) + 0.5 * sum(1.0 for x in neg if p == x)
    return wins / (len(pos) * len(neg))


def aurc_score(confs: list[float], accs: list[int]) -> float:
    order = sorted(range(len(confs)), key=lambda i: (-confs[i], i))
    cum = 0
    risks = 0.0
    for k, idx in enumerate(order, start=1):
        cum += accs[idx]
        risks += 1.0 - (cum / k)
    return risks / len(order)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / (2 ** (n - 1)))


def verify_arc_routing() -> dict:
    pilot = load_jsonl(D3B / "arc_routing_pilot.jsonl")
    confirm = load_jsonl(D3B / "arc_routing_confirmation.jsonl")
    exp = json.loads((D3B / "arc_routing_expected.json").read_text())
    s_exp = exp["analysis_summary"]
    assert len(pilot) == 100 and len(confirm) == 250
    assert {r["rank"] for r in pilot}.isdisjoint({r["rank"] for r in confirm})

    for split_name, rows, metrics_exp in (
        ("pilot", pilot, s_exp["pilot_metrics"]),
        ("confirmation", confirm, s_exp["confirmation_metrics"]),
    ):
        assert sum(r["n_tokens"] for r in rows) == metrics_exp["positions"]
        assert sum(r["wrong_ids"] for r in rows) == metrics_exp["wrong_ids"]
        assert sum(r["wrong_scores"] for r in rows) == metrics_exp["wrong_scores"]
        assert sum(r["out_of_top5"] for r in rows) == metrics_exp["out_of_top5"]
        assert sum(r["equal_score_swaps"] for r in rows) == metrics_exp["equal_score_swaps"]
        assert sum(1 for r in rows if r["wrong_ids"] > 0) == metrics_exp["responses_with_wrong_id"]
        deltas = [r["mean_wire_lp"] - r["mean_native_lp"] for r in rows]
        assert sum(1 for d in deltas if d > 0) == metrics_exp["responses_positive_overstatement"]
        assert round(mean(deltas), 4) == metrics_exp["delta_lp_nats"]["mean"]
        assert round(median(deltas), 4) == metrics_exp["delta_lp_nats"]["median"]

        confs_nat = [math.exp(r["mean_native_lp"]) for r in rows]
        confs_bug = [math.exp(r["mean_wire_lp"]) for r in rows]
        accs = [int(r["correct"]) for r in rows]
        cal = metrics_exp["calibration_all"]
        assert round(ece_10bin(confs_nat, accs), 6) == cal["ece_10bin"]["native"]
        assert round(ece_10bin(confs_bug, accs), 6) == cal["ece_10bin"]["buggy"]
        assert round(brier_score(confs_nat, accs), 6) == cal["brier"]["native"]
        assert round(brier_score(confs_bug, accs), 6) == cal["brier"]["buggy"]
        assert round(auroc_score(confs_nat, accs), 6) == cal["auroc"]["native"]
        assert round(auroc_score(confs_bug, accs), 6) == cal["auroc"]["buggy"]
        assert round(aurc_score(confs_nat, accs), 6) == cal["aurc"]["native"]
        assert round(aurc_score(confs_bug, accs), 6) == cal["aurc"]["buggy"]

        pc = s_exp["pilot_calibration"]
        tau_map = {
            "pilot_tau_85": pc["tau_85"]["tau"],
            "pilot_tau_90": pc["tau_90"]["tau"],
            "pilot_tau_youden": pc["tau_youden"]["tau"],
            "fixed_0.70": 0.70,
            "fixed_0.80": 0.80,
            "fixed_0.90": 0.90,
        }
        for pol, pexp in metrics_exp["routing"].items():
            tau = tau_map.get(pol, pexp["tau"])
            nat_acc = [c >= tau for c in confs_nat]
            bug_acc = [c >= tau for c in confs_bug]
            assert sum(nat_acc) == pexp["native_accepted"]
            assert sum(bug_acc) == pexp["buggy_accepted"]
            b = sum((not n) and bg for n, bg in zip(nat_acc, bug_acc, strict=True))
            c = sum(n and (not bg) for n, bg in zip(nat_acc, bug_acc, strict=True))
            assert b == pexp["flips_escalate_to_accept"] and c == pexp["flips_accept_to_escalate"]
            assert round(mcnemar_exact(b, c), 6) == pexp["routing_mcnemar_p"]
            fa_nat = sum((a == 0) and n for a, n in zip(accs, nat_acc, strict=True))
            fa_bug = sum((a == 0) and bg for a, bg in zip(accs, bug_acc, strict=True))
            assert fa_nat == pexp["native_false_accepts"] and fa_bug == pexp["buggy_false_accepts"]

    ir = exp["inspect_replay"]
    assert ir["confirmation_responses"] == 250 and ir["positions"] == 10270
    assert ir["original"]["wrong_tokens"] == 1607 and ir["original"]["wrong_scores"] == 1600
    assert ir["repaired"]["wrong_tokens"] == 0 and ir["repaired"]["wrong_scores"] == 0
    return {
        "confirmation_positions": s_exp["confirmation_metrics"]["positions"],
        "confirmation_wrong_ids": s_exp["confirmation_metrics"]["wrong_ids"],
        "tau85_flips_esc_to_acc": s_exp["confirmation_metrics"]["routing"]["pilot_tau_85"]["flips_escalate_to_accept"],
        "tau85_false_accepts_nat_vs_bug": [
            s_exp["confirmation_metrics"]["routing"]["pilot_tau_85"]["native_false_accepts"],
            s_exp["confirmation_metrics"]["routing"]["pilot_tau_85"]["buggy_false_accepts"],
        ],
    }


def verify_geval() -> dict:
    exp = json.loads((D3B / "geval_expected.json").read_text())
    out = {}
    for key, fname in (("panel_a", "geval_panel_a.jsonl"), ("panel_b", "geval_panel_b.jsonl")):
        rows = load_jsonl(D3B / fname)
        e = exp[key]
        assert len(rows) == e["n_cases"]
        assert sum(r["positions"] for r in rows) == e["positions"]
        assert sum(r["wrong_ids"] for r in rows) == e["wrong_ids"]
        valid = [r for r in rows if r["parsed_both"]]
        assert len(valid) == e["valid_parsed_both"]
        assert sum(1 for r in valid if r["matched_span_orig"] is not None) == e["span_matched_orig"]
        assert sum(1 for r in valid if r["matched_span_corr"] is not None) == e["span_matched_corr"]
        assert sum(1 for r in valid if r["score_orig"] != r["score_corr"]) == e["weighted_score_changed"]
        assert sum(1 for r in valid if r["passed_orig"] != r["passed_corr"]) == e["pass_decision_flipped"]
        out[key] = {
            "n_cases": len(rows),
            "valid_both": len(valid),
            "weighted_score_changed": e["weighted_score_changed"],
            "pass_decision_flipped": e["pass_decision_flipped"],
        }
    return out


def verify_ct2_distinct_and_audit() -> dict:
    ct2 = json.loads((D3B / "ct2_distinct_batch.json").read_text())
    agg_a = ct2["aggregate"]["subpanel_a_equal_length"]
    assert agg_a["joint_item0_received_2x_vocab"] == 24
    assert agg_a["joint_item0_row0_matches_item0_final_exact"] == 24
    assert agg_a["joint_item0_row1_leaked_item1_final_exact"] == 24
    assert agg_a["joint_item1_np_asarray_null_errors"] == 24
    assert agg_a["split_exact_single_row"] == 48 and agg_a["split_empty_callbacks"] == 0
    assert ct2["aggregate"]["subpanel_b_unequal_length"]["joint_longer_item_masked_prefix_steps"] == [1, 1, 2, 2]
    assert ct2["aggregate"]["subpanel_b_unequal_length"]["split_longer_item_masked_prefix_steps"] == [0, 0, 0, 0]

    for candidate in (HERE / "tools", REPO / "tools"):
        if candidate.is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break
    from tokenreceipt_audit import replay_offline_matrix  # noqa: E402

    mat = replay_offline_matrix(HERE)
    assert mat["ct2/joint-distinct-eq"]["per_position"]["I3"] == {"fail": 48}
    assert mat["ct2/split-distinct-eq"]["per_position"]["I3"] == {"pass": 48}
    assert mat["ct2/singletons-distinct-eq"]["per_position"]["I3"] == {"pass": 48}

    la = json.loads((D3B / "live_audit_summary.json").read_text())
    assert la["smollm2_135m"]["stock_live_audit"]["width5"]["per_position"]["I1"] == {"pass": 51, "fail": 13}
    assert la["smollm2_135m"]["patched_live_audit"]["width5"]["per_position"]["I1"] == {"pass": 64}
    assert la["qwen25_3b"]["stock_live_audit"]["width5"]["per_position"]["I1"] == {"pass": 60, "fail": 4}
    assert la["qwen25_3b"]["patched_live_audit"]["width5"]["per_position"]["I1"] == {"pass": 64}
    return {
        "ct2_joint_I3": mat["ct2/joint-distinct-eq"]["per_position"]["I3"],
        "ct2_split_I3": mat["ct2/split-distinct-eq"]["per_position"]["I3"],
        "qwen3b_patched_w5_I1": la["qwen25_3b"]["patched_live_audit"]["width5"]["per_position"]["I1"],
    }


def main() -> None:
    res = {
        "arc_routing": verify_arc_routing(),
        "geval": verify_geval(),
        "ct2_and_live_audit": verify_ct2_distinct_and_audit(),
    }
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
