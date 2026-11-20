"""Collect the frozen IFEval panel and three repeats against local MLX-LM 0.32.2."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener, urlopen


HERE = Path(__file__).resolve().parent
IFEVAL_URL = ("https://raw.githubusercontent.com/google-research/google-research/"
              "7117df5f368e1301e34bd014be9c93e24a4e73c6/instruction_following_eval/data/input_data.jsonl")
IFEVAL_HASH = "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
SERVER_HASH = "8d475058966cf90afcf8cb857f218c2b9dbc01ad1c506d9726de655a90d80723"
REPLAY_ARMS = ("forward-default", "reverse-default", "forward-zero")
HTTP = build_opener(ProxyHandler({}))


def source_panel(local_source):
    if local_source:
        data = local_source.read_bytes()
    else:
        with urlopen(IFEVAL_URL, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != IFEVAL_HASH:
        raise ValueError("IFEval corpus differs from the pinned source")
    population = {row["key"]: row["prompt"] for row in map(json.loads, data.splitlines())}
    keys = json.loads((HERE / "keys.json").read_bytes())
    if len(population) != 541 or len(keys) != 64 or len(set(keys)) != 64:
        raise ValueError("Expected the frozen 64-key panel from 541 IFEval prompts")
    return [(key, population[key]) for key in keys]


def payloads(panel):
    for key, prompt in panel:
        for draw in range(4):
            yield key, draw, dict(model="default_model", messages=[dict(role="user", content=prompt)],
                                  temperature=0.8, top_p=1.0, top_k=0, max_tokens=24, logprobs=True,
                                  top_logprobs=5, seed=73803 + 10 * key + draw, stream=False)


def verify_environment(model):
    installed = {}
    for line in (HERE / "requirements.txt").read_text().splitlines():
        name, pin = line.split("==")
        installed[name] = version(name)
        if installed[name] != pin:
            raise ValueError(f"Expected {name}=={pin}, found {installed[name]}")
    server_file = distribution("mlx-lm").locate_file("mlx_lm/server.py")
    model_files = json.loads((HERE / "model.json").read_bytes())["sha256"]
    for path, expected in [(server_file, SERVER_HASH), *((model / n, h) for n, h in model_files.items())]:
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Wrong SHA-256 for {path.name}")
    return installed


@contextmanager
def service(output, model, arm):
    receipts = output / arm
    receipts.mkdir()
    env = {**os.environ, "CAPTURE_OUTPUT": str(receipts),
           "CAPTURE_NATIVE": "1" if arm == "original" else "0",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    command = [sys.executable, "-I", "-B", str(HERE / "worker.py"), "--model", str(model),
               "--port", "0", "--host", "127.0.0.1", "--log-level", "WARNING"]
    if arm == "forward-zero":
        command.extend(("--prompt-cache-size", "0"))
    with (output / f"{arm}.log").open("x") as log, subprocess.Popen(
            command, cwd=receipts, env=env, stdout=log, stderr=subprocess.STDOUT) as server:
        try:
            port = receipts / "port"
            timeout = time.monotonic() + 60
            while not port.exists():
                if server.poll() is not None or time.monotonic() >= timeout:
                    raise RuntimeError(f"Server did not start. See {arm}.log")
                time.sleep(0.1)
            base = f"http://127.0.0.1:{int(port.read_text())}"
            if arm != "original":
                with HTTP.open(base + "/v1/models", timeout=15) as response:
                    response.read()
            yield base + "/v1/chat/completions", receipts
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def capture(output, model, requests, arm):
    filename = "captures.jsonl.gz" if arm == "original" else f"{arm}.jsonl.gz"
    order = reversed(requests) if arm == "reverse-default" else requests
    with service(output, model, arm) as (url, receipts), gzip.open(output / filename, "xt", encoding="utf-8") as out:
        for key, draw, original in order:
            payload = original.copy()
            headers = {"Content-Type": "application/json"}
            label = f"{key}-0.8-{draw}"
            if arm == "original":
                headers["X-Study-Id"] = label
            else:
                del payload["top_logprobs"]
            request = Request(url, json.dumps(payload).encode(), headers)
            with HTTP.open(request, timeout=60) as response:
                wire = json.load(response)
            row = dict(key=key, draw=draw, request=payload, wire=wire)
            if arm == "original":
                native = receipts / f"{label}.native.json"
                serializer = receipts / f"{label}.serializer.json"
                record = json.loads(serializer.read_bytes())
                if wire != record["upstream"]:
                    raise ValueError(f"HTTP response differs from the same-call witness for {label}")
                row.update(native=json.loads(native.read_bytes()), plain=record["same_call_no_candidates"],
                           source=record["pre_serializer"])
                native.unlink()
                serializer.unlink()
            print(json.dumps(row), file=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new output directory for raw journals")
    parser.add_argument("--model", type=Path, help="local directory of the pinned checkpoint")
    parser.add_argument("--ifeval-file", type=Path, help="use a local copy of the pinned IFEval JSONL")
    parser.add_argument("--check-panel", action="store_true", help="verify corpus and keys without running MLX")
    args = parser.parse_args()
    if not args.check_panel and (args.output is None or args.model is None):
        parser.error("--output and --model are required for capture")
    requests = list(payloads(source_panel(args.ifeval_file)))
    if args.check_panel:
        print(f"Verified {len(requests) // 4} prompts and {len(requests)} request seeds")
        return
    model = args.model.resolve()
    installed = verify_environment(model)
    output = args.output.resolve()
    output.mkdir()
    (output / "installed.json").write_text(json.dumps(installed) + "\n")
    for arm in ("original", *REPLAY_ARMS):
        print(f"Collecting {arm}", flush=True)
        capture(output, model, requests, arm)


if __name__ == "__main__":
    main()
