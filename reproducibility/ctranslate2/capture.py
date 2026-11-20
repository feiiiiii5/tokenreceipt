"""Capture the frozen public CTranslate2 4.9.0 generation and prefix-forward panel."""
import argparse
import fcntl
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import ctranslate2
import numpy as np
from transformers import AutoTokenizer


PROMPTS = (
    "The book on the table was",
    "In a quiet garden, the",
    "0, 1, 1, 2, 3,",
    'She said: "I think',
    "Le café est",
    "春天来了，",
)
REGIMES = (
    ("neutral", 1.0, 0, (1701, 2701)),
    ("stress", 0.7, 5, (1701, 2701)),
    ("greedy", 1.0, 1, (1701,)),
)
GENERATOR_OPTIONS = dict(device="cpu", compute_type="float32", inter_threads=1, intra_threads=1)


def file_sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, data, pretty=False):
    with path.open("w", encoding="utf8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2 if pretty else None, allow_nan=False)
        output.write("\n")


def to_finite_json(value):
    if value is not None and not np.isfinite(value):
        return repr(value)
    return value


def new_generator(model):
    return ctranslate2.Generator(str(model), **GENERATOR_OPTIONS)


def capture_vocabulary(model, tokenizer):
    native = json.loads((model / "vocabulary.json").read_text())
    native_config = json.loads((model / "config.json").read_text())
    all_ids = range(len(tokenizer))
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = list(byte_values)
    remainder = [value for value in range(256) if value not in byte_values]
    byte_decoder = dict(zip(map(chr, codepoints + list(range(256, 256 + len(remainder)))),
                            byte_values + remainder))
    return {
        "ct2_tokens": native,
        "tokenizer_tokens": tokenizer.convert_ids_to_tokens(list(all_ids)),
        "byte_decoder": byte_decoder,
        "special_token_ids": tokenizer.all_special_ids,
        "eos_token_id": native.index(native_config["eos_token"]),
    }


def save_vector(vectors, key, value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if not array.size:
        return None
    vectors[key] = array
    return key


def capture_step(item, vectors, vector_prefix):
    return {
        "step": item.step,
        "batch_id": item.batch_id,
        "hypothesis_id": item.hypothesis_id,
        "token_id": item.token_id,
        "token": item.token,
        "log_prob": to_finite_json(item.log_prob),
        "is_last": item.is_last,
        "logits_key": save_vector(vectors, vector_prefix, item.logits),
        "forward_key": None,
    }


def capture_visible(tokenizer, token_ids, tokens):
    decoding = dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
    across_ids = tokenizer.decode(token_ids, **decoding)
    across_tokens = tokenizer.convert_tokens_to_string(tokens)
    return {
        "token_ids": token_ids,
        "tokens": tokens,
        "decoded_ids_text": across_ids,
        "decoded_tokens_text": across_tokens,
        "decoded_ids_utf8_hex": across_ids.encode("utf8").hex(),
        "decoded_tokens_utf8_hex": across_tokens.encode("utf8").hex(),
    }


def capture_call(model, path, prompt, regime, temperature, topk, seed, tokenizer, vectors):
    call_id = f"{path}-{prompt['prompt_id']}-{regime}-{seed}"
    options = dict(max_length=12, min_length=0, sampling_temperature=temperature,
                   sampling_topk=topk, sampling_topp=1.0)
    if path == "batch":
        options.update(beam_size=1, include_prompt_in_result=False, return_end_token=False,
                       return_scores=True, return_logits_vocab=True)
    else:
        options["return_log_prob"] = True
    record = dict(call_id=call_id, path=path, prompt_id=prompt["prompt_id"], regime=regime,
                  seed=seed, options=options, steps=[], stream_callbacks=[], result=None,
                  visible=None, error=None)

    def append_steps(item):
        rows = record["steps"] if path == "batch" else record["stream_callbacks"]
        rows.append(capture_step(item, vectors, f"{call_id}-callback-{len(rows)}"))
        return False

    # The release initializes its thread-local RNG at first use on each worker.
    # Reset the seed and allocate a fresh worker for each public generation call.
    ctranslate2.set_random_seed(seed)
    generator = new_generator(model)
    try:
        if path == "batch":
            result, = generator.generate_batch([prompt["tokens"]], callback=append_steps, **options)
            record["result"] = {
                "sequences_ids": result.sequences_ids,
                "sequences": result.sequences,
                "scores": [to_finite_json(value) for value in result.scores],
                "logits_key": save_vector(vectors, f"{call_id}-result", result.logits),
            }
            token_ids, = result.sequences_ids
            tokens, = result.sequences
        else:
            for item in generator.generate_tokens(prompt["tokens"], callback=append_steps, **options):
                rows = record["steps"]
                rows.append(capture_step(item, vectors, f"{call_id}-stream-{len(rows)}"))
            token_ids = [row["token_id"] for row in record["steps"]]
            tokens = [row["token"] for row in record["steps"]]
        record["visible"] = capture_visible(tokenizer, token_ids, tokens)
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        del generator
    return record


def capture_prefixes(forward_engine, call, prompt_ids, vectors, key_by_prefix, forward_prefixes):
    prefix = list(prompt_ids)
    for row in call["steps"]:
        signature = tuple(prefix)
        key = key_by_prefix.get(signature)
        if key is None:
            # Before reading position t's output, run a separate full pass using
            # the original prompt followed by exactly the exposed IDs before t.
            native = np.asarray(forward_engine.forward_batch([prefix], return_log_probs=False))
            if native.dtype != np.float32 or native.ndim != 3 or native.shape[:2] != (1, len(prefix)):
                raise ValueError(f"Unexpected native forward shape/dtype: {native.shape} {native.dtype}")
            key = f"forward-{len(key_by_prefix)}"
            key_by_prefix[signature] = key
            forward_prefixes[key] = prefix.copy()
            vectors[key] = native[0, -1].copy()
        row["forward_key"] = key
        prefix.append(row["token_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with Path("/tmp/native-engine-review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        protocol_directory = Path(__file__).resolve().parent
        frozen = json.loads((protocol_directory / "freeze.json").read_text())
        manifest = protocol_directory / "model-source.json"
        if (file_sha256(protocol_directory / "PROTOCOL.md") != frozen["protocol_sha256"]
                or file_sha256(manifest) != frozen["model_manifest_sha256"]):
            raise ValueError("Protocol or frozen model manifest has changed.")
        if ctranslate2.__version__ != "4.9.0":
            raise ValueError(f"Unexpected engine version: {ctranslate2.__version__}")
        output = args.output
        output.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        vocabulary = capture_vocabulary(args.model, tokenizer)
        write_json(output / "vocabulary.json", vocabulary)
        metadata = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "capture_sha256": file_sha256(Path(__file__)),
            "engine": ctranslate2.__version__,
            "generator_options": GENERATOR_OPTIONS,
            "seed_scope": "fresh generator worker after set_random_seed for each call",
            "forward_comparison": "separate generator forward on prompt plus prior exposed IDs",
            "forward_prefixes": {},
            "eos_token_id": vocabulary["eos_token_id"],
            "eos_token": vocabulary["ct2_tokens"][vocabulary["eos_token_id"]],
            "model_config": json.loads((args.model / "config.json").read_text()),
            "vector_storage_dtype": "float32",
            "planned_calls_by_path": {"batch": 30, "stream": 30},
        }
        prompts = []
        for index, prompt in enumerate(PROMPTS, 1):
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompts.append(dict(prompt_id=index, text=prompt, token_ids=token_ids,
                                tokens=tokenizer.convert_ids_to_tokens(token_ids),
                                live_text_utf8_hex=prompt.encode("utf8").hex()))
        calls, vectors, key_by_prefix = [], {}, {}
        with (output / "capture.jsonl").open("x", encoding="utf8") as journal:
            for path in ("batch", "stream"):
                for prompt in prompts:
                    for regime, temperature, topk, seeds in REGIMES:
                        for seed in seeds:
                            record = capture_call(args.model, path, prompt, regime, temperature,
                                                  topk, seed, tokenizer, vectors)
                            calls.append(record)
                            journal.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                            journal.flush()
                            print(len(calls), record["call_id"], len(record["steps"]),
                                  record["error"] or "ok", flush=True)
            forward_engine = new_generator(args.model)
            for record in calls:
                prompt_ids = prompts[record["prompt_id"] - 1]["token_ids"]
                try:
                    capture_prefixes(forward_engine, record, prompt_ids, vectors, key_by_prefix,
                                     metadata["forward_prefixes"])
                except Exception as error:
                    record["forward_error"] = f"{type(error).__name__}: {error}"
                    print(record["call_id"], record["forward_error"], flush=True)
            del forward_engine
        np.savez_compressed(output / "vectors.npz", **vectors)
        metadata["elapsed_seconds"] = time.monotonic() - start
        write_json(output / "run.json", dict(metadata=metadata, prompts=prompts, calls=calls), pretty=True)


if __name__ == "__main__":
    main()
