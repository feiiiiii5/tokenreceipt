"""Acquire and grade a same-process decoder-input trace using the shared study lock."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
DECODER_SYMBOL = (
    "_ZN11ctranslate26layers18TransformerDecoderclExRKNS_11StorageViewERNSt3__113unordered_map"
    "INS5_12basic_stringIcNS5_11char_traitsIcEENS5_9allocatorIcEEEES2_NS5_4hashISC_EEN"
    "S5_8equal_toISC_EENSA_INS5_4pairIKSC_S2_EEEEEEPS2_SN_"
)
TRACE_PREFIX = "CT2_DECODER "


def grade(output):
    response = json.loads((output / "response.json").read_text())
    trace = [
        json.loads(line[len(TRACE_PREFIX):])
        for line in (output / "lldb.log").read_text().splitlines()
        if line.startswith(TRACE_PREFIX)
    ]
    (output / "decoder_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trace)
    )
    rows = response["callback_rows"]
    prompt_ids = response["prompt_ids"]
    expected_steps = list(range(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(rows)))
    if [row["x1_step"] for row in trace] != expected_steps:
        raise RuntimeError("Wrong number or order of incremental decoder inputs")
    if not rows or trace[0]["token_id"] != prompt_ids[-1]:
        raise RuntimeError("First incremental call did not use the final prompt token")
    if [row["step"] for row in rows] != list(range(len(rows))):
        raise RuntimeError("Unexpected callback step sequence")
    if not rows[-1]["is_last"] or any(row["is_last"] for row in rows[:-1]):
        raise RuntimeError("Unexpected callback stop framing")
    final = response["final_sequence_ids"]
    comparisons = [
        {
            "callback_step": row["step"], "callback_id": row["token_id"],
            "final_id": final[row["step"]] if row["step"] < len(final) else None,
            "native_decoder_input": later["token_id"], "rank": row["rank"],
        }
        for row, later in zip(rows[:-1], trace[1:], strict=True)
    ]
    matches = sum(
        row["callback_id"] == row["final_id"] == row["native_decoder_input"]
        for row in comparisons
    )
    summary = {
        "prompt_anchor_id": trace[0]["token_id"],
        "callback_steps": len(rows), "native_decoder_inputs": len(trace),
        "witnessed_generated_ids": len(comparisons), "matching_generated_ids": matches,
        "witnessed_off_first_ids": sum(row["rank"] > 1 for row in comparisons),
        "last_callback_row_without_a_next_input": rows[-1], "comparisons": comparisons,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if matches != len(comparisons):
        raise RuntimeError("Reported ID disagrees with the next decoder input")
    return summary


def main():
    parser = argparse.ArgumentParser()
    for key in ("site", "model", "tokenizer", "output"):
        parser.add_argument(f"--{key}", required=True, type=lambda value: Path(value).resolve())
    args = parser.parse_args()

    with Path("/tmp/native-engine-review.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Shared research lock is already held")
        args.output.mkdir()
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(args.site), "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        })
        command = [
            "xcrun", "lldb", "--batch",
            "-o", "settings set target.disable-aslr false",
            "-o", f'command script import "{HERE / "lldb_hook.py"}"',
            "-o", f"breakpoint set --shlib libctranslate2.4.9.0.dylib "
                  f"--skip-prologue false --name {DECODER_SYMBOL}",
            "-o", "breakpoint command add --python-function lldb_hook.record_decoder_input 1",
            "-o", "run", "-o", "breakpoint list --verbose",
            "-f", sys.executable, "--", "-B", str(HERE / "infer.py"),
            str(args.model), str(args.tokenizer), str(args.output / "response.json"),
        ]
        frozen_files = [HERE / name for name in ("run.py", "infer.py", "lldb_hook.py")]
        record = {
            "launched_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": command, "site": str(args.site), "timeout_s": 120,
            "sha256": {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in frozen_files
            },
        }
        (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        with (args.output / "lldb.log").open("w") as log:
            child = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                status = child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise
        if status:
            raise SystemExit(f"LLDB failed with exit status {status}. See {args.output / 'lldb.log'}")
        summary = grade(args.output)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
