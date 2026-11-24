"""Offline pure-stdlib verification of the November 24 full-transport patch live audit.

Checks that the released 15-check patch matches the audited patch, recomputes the
cross-scale I1/I2/I3/I5 totals from live_audit_2026_11_24/summary.json, and confirms
that the archived 11-check preprint patch reproduces the patched-server hash used by
the November 23 live audits in downstream_3b/.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
SUMMARY = json.loads((HERE / "live_audit_2026_11_24" / "summary.json").read_text(encoding="utf-8"))
RELEASE = json.loads((HERE / "patch" / "release.json").read_text(encoding="utf-8"))
RESULTS = json.loads((HERE / "patch" / "results.json").read_text(encoding="utf-8"))
STOCK = HERE / "releases" / "mlx-0322-server.py"
INVARIANTS = ("I1", "I2", "I3", "I5")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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


def main() -> None:
    print(json.dumps({"release": verify_release_binding(), "live_audit": verify_live_audit()}, indent=2))


if __name__ == "__main__":
    main()
