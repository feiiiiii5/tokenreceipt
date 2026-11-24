"""Verify the replay, CTranslate2, and frozen comparison evidence offline."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(*args, capture=False):
    return subprocess.run([sys.executable, "-B", *args], cwd=ROOT, check=True,
                          capture_output=capture)


def main():
    reports = (
        ("replay/results.json", ("replay/analyze.py",)),
        ("ctranslate2/audit.json", ("ctranslate2/verify.py", "--projection", "ctranslate2/evidence.json")),
    )
    for expected, command in reports:
        result = run(*command, capture=True)
        if result.stdout != (ROOT / expected).read_bytes():
            raise ValueError(f"Recomputed report differs from {expected}")
        print(f"{expected} matches", flush=True)

    run("replay/counterexample.py")
    run("replay/capture/test_wire_recovery.py")
    run("ctranslate2/check_controls.py")
    run("comparison/verify.py")
    run("verify_2026_11_23.py")
    run("verify_2026_11_24.py")


if __name__ == "__main__":
    main()
