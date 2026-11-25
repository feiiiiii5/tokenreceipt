"""Offline pure-stdlib verification of the November 24, 2026 patch, streaming, Ollama, and 7B downstream studies.

Checks:
1. 15-check full-transport patch hash binding and archived 11-check preprint patch hash binding.
2. Cross-scale SmolLM2-135M / Qwen2.5-3B / Qwen2.5-7B live I1/I2/I3/I5 audit totals (384/384 patched).
3. Qwen2.5-7B ARC-Challenge routing, Calibration Paradox (ECE vs false-accepts), and Inspect AI 0.3.277 replay.
4. Qwen2.5-7B DeepEval 4.3.4 GEval Panels A/B score-shift and zero-flip scale contrast.
5. Qwen2.5-7B 50x4 GRPO Step-0 clipping censoring/attenuation (eps in {0.1, 0.2, 0.28}) and spurious Schulman k3 KL.
6. SGLang v0.5.23 vs vLLM v0.34.0 streaming contracts and live Ollama v0.40.17 HTTP/SSE contracts + replay_offline_matrix.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
D24 = HERE / "downstream_2026_11_24"
SUMMARY = json.loads((HERE / "live_audit_2026_11_24" / "summary.json").read_text(encoding="utf-8"))
RELEASE = json.loads((HERE / "patch" / "release.json").read_text(encoding="utf-8"))
RESULTS = json.loads((HERE / "patch" / "results.json").read_text(encoding="utf-8"))
STOCK = HERE / "releases" / "mlx-0322-server.py"
INVARIANTS = ("I1", "I2", "I3", "I5")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ece_10bin(confs: list[float], accs: list[int]) -> float:
    n = len(confs)
    ece = 0.0
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        idx = [j for j, c in enumerate(confs) if (lo <= c <= hi if i == 0 else lo < c <= hi)]
        if idx:
            ece += (len(idx) / n) * abs(mean(accs[j] for j in idx) - mean(confs[j] for j in idx))
    return ece


def brier_score(confs: list[float], accs: list[int]) -> float:
    return sum((c - a) ** 2 for c, a in zip(confs, accs, strict=True)) / len(accs)


def auroc_score(confs: list[float], accs: list[int]) -> float:
    pos = [c for c, a in zip(confs, accs, strict=True) if a == 1]
    neg = [c for c, a in zip(confs, accs, strict=True) if a == 0]
    wins = sum(sum(1.0 for x in neg if p > x) + 0.5 * sum(1.0 for x in neg if p == x) for p in pos)
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


def patched_server_hash(patch: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / "mlx_lm" / "server.py"
        target.parent.mkdir(parents=True)
        target.write_bytes(STOCK.read_bytes())
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "apply", "--whitespace=error", str(patch)], cwd=root, check=True)
        return digest(target)


def verify_release_binding() -> dict:
    assert digest(STOCK) == RELEASE["server_sha256"]
    assert digest(HERE / "patch" / "mlx-lm-0.32.2.patch") == RELEASE["patch_sha256"]
    assert SUMMARY["patch_release"] == RELEASE
    assert SUMMARY["offline_checks"] == RESULTS["offline"]
    assert len(RESULTS["offline"]) == 15
    assert patched_server_hash(HERE / "patch" / "mlx-lm-0.32.2.patch") == RELEASE["patched_server_sha256"]

    preprint = HERE / "patch" / "preprint-11-check" / "mlx-lm-0.32.2.patch"
    nov23 = json.loads((HERE / "downstream_3b" / "live_audit_summary.json").read_text(encoding="utf-8"))
    nov23_hashes = {m["patched_live_audit"]["server_sha256"] for m in nov23.values() if isinstance(m, dict) and "patched_live_audit" in m}
    preprint_server = patched_server_hash(preprint)
    assert nov23_hashes == {preprint_server}
    return {
        "patch_sha256": RELEASE["patch_sha256"],
        "patched_server_sha256": RELEASE["patched_server_sha256"],
        "offline_checks": len(RESULTS["offline"]),
        "preprint_11_check_patch_sha256": digest(preprint),
        "preprint_11_check_patched_server_sha256": preprint_server,
    }


def verify_live_audit() -> dict:
    table = {}
    patched_total = {inv: 0 for inv in INVARIANTS}
    patched_positions = 0
    for key, arms in SUMMARY["models"].items():
        stock, patched = arms["stock"], arms["patched"]
        assert stock["patch_applied"] is False and patched["patch_applied"] is True
        assert stock["server_sha256"] == RELEASE["server_sha256"]
        assert patched["server_sha256"] == RELEASE["patched_server_sha256"]
        row = {}
        for width in ("width5", "width1"):
            s, p = stock[width], patched[width]
            assert s["positions"] == p["positions"] == 64 and s["requests"] == p["requests"] == 4
            assert s["per_position"]["I5"] == {"fail": 64}
            assert s["per_position"]["I3"] == {"pass": 64}
            assert s["per_position"]["I1"].get("fail", 0) > 0
            for inv in INVARIANTS:
                assert p["per_position"][inv] == {"pass": 64}, (key, width, inv)
                patched_total[inv] += 64
            patched_positions += 64
            row[width] = {
                "stock_I1_fail": s["per_position"]["I1"]["fail"],
                "stock_I2_fail": s["per_position"]["I2"]["fail"],
                "stock_I5_fail": s["per_position"]["I5"]["fail"],
            }
        for d in patched["text_and_rows"]:
            for w in ("w5", "w1"):
                assert d[f"{w}_same_text_k0_vs_k"] and d[f"{w}_same_text_stream"]
                assert d[f"{w}_stream_logprob_rows"] == d[f"{w}_usage_tokens"]
        for d in stock["text_and_rows"]:
            assert d["w5_stream_logprob_rows"] == 0 and d["w1_stream_logprob_rows"] == 0
        levels = {lvl for arm in (stock, patched) for t in arm["memory"] for lvl in t["pressure_levels"]}
        assert max(levels) < 4
        row["max_family_rss_mib"] = max(t["family_rss_mib"] for arm in (stock, patched) for t in arm["memory"])
        row["pressure_levels"] = sorted(levels)
        table[key] = row
    assert patched_positions == 384 and all(v == 384 for v in patched_total.values())
    return {"patched_positions": patched_positions, "patched_pass": patched_total, "models": table}


def verify_arc_routing_7b() -> dict:
    pilot = load_jsonl(D24 / "arc_routing_7b_pilot.jsonl")
    confirm = load_jsonl(D24 / "arc_routing_7b_confirmation.jsonl")
    exp = json.loads((D24 / "arc_routing_7b_expected.json").read_text(encoding="utf-8"))
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
    assert ir["confirmation_responses"] == 250 and ir["positions"] == 10801
    assert ir["original"]["wrong_tokens"] == 1136 and ir["original"]["wrong_scores"] == 1134
    assert ir["repaired"]["wrong_tokens"] == 0 and ir["repaired"]["wrong_scores"] == 0
    assert all(v == 250 for v in ir["routing_decision_agreement_250"].values())
    cm = s_exp["confirmation_metrics"]
    return {
        "confirmation_positions": cm["positions"],
        "confirmation_wrong_ids": cm["wrong_ids"],
        "ece_nat_vs_bug": [cm["calibration_all"]["ece_10bin"]["native"], cm["calibration_all"]["ece_10bin"]["buggy"]],
        "tau90_false_accepts_nat_vs_bug": [
            cm["routing"]["pilot_tau_90"]["native_false_accepts"],
            cm["routing"]["pilot_tau_90"]["buggy_false_accepts"],
        ],
        "fixed80_false_accepts_nat_vs_bug": [
            cm["routing"]["fixed_0.80"]["native_false_accepts"],
            cm["routing"]["fixed_0.80"]["buggy_false_accepts"],
        ],
    }


def verify_geval_7b() -> dict:
    exp = json.loads((D24 / "geval_7b_expected.json").read_text(encoding="utf-8"))
    out = {}
    for key, fname in (("panel_a", "geval_7b_panel_a.jsonl"), ("panel_b", "geval_7b_panel_b.jsonl")):
        rows = load_jsonl(D24 / fname)
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


def verify_grpo_7b() -> dict:
    data = json.loads((D24 / "grpo_7b_step0.json").read_text(encoding="utf-8"))
    exp = data["expected_summary"]
    rollouts = data["rollouts"]
    assert len(rollouts) == exp["rollouts"] == 200
    assert {r["rank"] for r in rollouts}.isdisjoint(set(range(350)))

    total_pos = sum(r["n_tokens"] for r in rollouts)
    total_wid = sum(r["wrong_ids"] for r in rollouts)
    total_wsc = sum(r["wrong_scores"] for r in rollouts)
    assert total_pos == exp["positions"] == 8328
    assert total_wid == exp["wrong_ids"] == 847
    assert total_wsc == exp["wrong_scores"] == 844

    by_q: dict[int, list[dict]] = {}
    for r in rollouts:
        by_q.setdefault(r["rank"], []).append(r)
    assert len(by_q) == exp["questions"] == 50

    mixed_neg_ratios: list[tuple[float, bool]] = []
    mixed_pos_ratios: list[tuple[float, bool]] = []
    rollout_mean_kls: list[float] = []
    rollouts_nonzero_kl = 0
    corrupt_kls: list[float] = []
    all_kl_sum = 0.0
    mixed_count = 0

    for group in by_q.values():
        assert len(group) == 4
        rewards = [r["reward"] for r in group]
        s_rew = sum(rewards)
        is_mixed = 0.0 < s_rew < 4.0
        if is_mixed:
            mixed_count += 1
        mu = s_rew / 4.0
        for r in group:
            adv = r["reward"] - mu
            kl_sum = 0.0
            has_nonzero_kl = False
            for cp in r["corrupted_positions"]:
                ratio = math.exp(cp["native_lp"] - cp["wire_lp"])
                kl = ratio - math.log(ratio) - 1.0
                kl_sum += kl
                all_kl_sum += kl
                if kl > 0.0:
                    has_nonzero_kl = True
                if cp["score_corrupt"]:
                    corrupt_kls.append(kl)
            rollout_mean_kls.append(kl_sum / r["n_tokens"])
            if has_nonzero_kl:
                rollouts_nonzero_kl += 1

            if is_mixed:
                n_clean = r["n_tokens"] - len(r["corrupted_positions"])
                entries = [(1.0, False)] * n_clean + [
                    (math.exp(cp["native_lp"] - cp["wire_lp"]), cp["score_corrupt"])
                    for cp in r["corrupted_positions"]
                ]
                if adv < 0.0:
                    mixed_neg_ratios.extend(entries)
                elif adv > 0.0:
                    mixed_pos_ratios.extend(entries)

    assert mixed_count == exp["mixed_reward_groups"]
    sk_exp = exp["spurious_kl_step0"]
    assert rollouts_nonzero_kl == sk_exp["rollouts_nonzero_kl"]
    assert round(mean(rollout_mean_kls), 6) == sk_exp["rollout_mean_kl_avg"]
    assert round(median(rollout_mean_kls), 6) == sk_exp["rollout_mean_kl_median"]
    assert round(max(rollout_mean_kls), 6) == sk_exp["rollout_mean_kl_max"]
    assert round(mean(corrupt_kls), 6) == sk_exp["per_corrupted_position_kl_mean"]
    assert round(all_kl_sum / total_pos, 6) == sk_exp["all_positions_kl_mean"]

    for eps_str, e_exp in exp["epsilons"].items():
        eps = float(eps_str)
        neg_zeroed = [ratio for ratio, _ in mixed_neg_ratios if ratio < 1.0 - eps]
        neg_corrupt = [ratio for ratio, sc in mixed_neg_ratios if sc]
        pos_atten = [ratio for ratio, _ in mixed_pos_ratios if ratio < 1.0]
        assert len(mixed_neg_ratios) == e_exp["neg_adv"]["positions"]
        assert len(neg_zeroed) == e_exp["neg_adv"]["hard_zeroed"]
        assert len(neg_corrupt) == e_exp["neg_adv"]["score_corrupt"]
        assert len(mixed_pos_ratios) == e_exp["pos_adv"]["positions"]
        assert len(pos_atten) == e_exp["pos_adv"]["attenuated"]

    return {
        "rollouts": len(rollouts),
        "positions": total_pos,
        "wrong_scores": total_wsc,
        "eps02_neg_hard_zeroed": f"{exp['epsilons']['0.2']['neg_adv']['hard_zeroed']}/{exp['epsilons']['0.2']['neg_adv']['positions']}",
        "eps02_pos_attenuated": f"{exp['epsilons']['0.2']['pos_adv']['attenuated']}/{exp['epsilons']['0.2']['pos_adv']['positions']}",
        "rollouts_nonzero_kl": rollouts_nonzero_kl,
    }


def verify_sglang_vllm_and_ollama() -> dict:
    for candidate in (HERE / "tools", REPO / "tools"):
        if candidate.is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break
    from tokenreceipt_audit import replay_offline_matrix  # noqa: E402

    mat = replay_offline_matrix(HERE)
    sg = json.loads((D24 / "sglang_vllm_streaming.json").read_text(encoding="utf-8"))
    for arm_name, exp_arm in sg["arms"].items():
        assert mat[arm_name]["per_position"] == exp_arm["all_widths"]["per_position"], arm_name
    assert mat["vllm/stream"]["per_position"]["I3"] == {"pass": 246}
    assert mat["vllm/stream"]["per_position"]["I5"] == {"pass": 246}
    assert mat["vllm/stream"]["per_position"]["I4"] == {"pass": 144, "fail": 102}
    assert mat["sglang/plain-stream"]["per_position"]["I3"] == {"fail": 78, "pass": 168}
    assert mat["sglang/plain-stream"]["per_position"]["I5"] == {"fail": 228, "pass": 18}
    assert mat["sglang/parser-stream"]["per_position"]["I5"] == {"pass": 246}
    assert mat["sglang/patched-stream"]["per_position"]["I5"] == {"pass": 246}

    ol = json.loads((D24 / "ollama_live_audit.json").read_text(encoding="utf-8"))
    for arm_name, exp_arm in ol["summaries"].items():
        assert mat[arm_name]["per_position"] == exp_arm["per_position"], arm_name
    assert mat["ollama/v1-t08-k5"]["per_position"]["I3"] == {"fail": 21, "pass": 403}
    assert mat["ollama/v1-t08-k5"]["per_position"]["I5"] == {"abstain": 21, "pass": 403}
    assert mat["ollama/v1-t00-k1"]["per_position"]["I2"] == {"abstain": 39, "pass": 349}

    return {
        "sglang_plain_I3": mat["sglang/plain-stream"]["per_position"]["I3"],
        "sglang_patched_I5": mat["sglang/patched-stream"]["per_position"]["I5"],
        "vllm_I4": mat["vllm/stream"]["per_position"]["I4"],
        "ollama_t08_k5_I2": mat["ollama/v1-t08-k5"]["per_position"]["I2"],
        "ollama_t00_k1_I3": mat["ollama/v1-t00-k1"]["per_position"]["I3"],
    }


def main() -> None:
    print(
        json.dumps(
            {
                "release": verify_release_binding(),
                "live_audit": verify_live_audit(),
                "arc_routing_7b": verify_arc_routing_7b(),
                "geval_7b": verify_geval_7b(),
                "grpo_7b_step0": verify_grpo_7b(),
                "sglang_vllm_ollama": verify_sglang_vllm_and_ollama(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
