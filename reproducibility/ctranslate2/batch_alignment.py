"""Identify the rows present in two-item callbacks from the released public API."""
import fcntl
import hashlib
import json
import sys
from pathlib import Path

import ctranslate2 as ct
import numpy as np
from tokenizers import Tokenizer


CONFIGURATIONS = (
    ("The book on the table was", 1701, 1, 1.0),
    ("In a quiet garden, the", 2701, 5, 0.7),
)


def digest(vector):
    return hashlib.sha256(np.asarray(vector, dtype="<f4").tobytes()).hexdigest()


def summarize(vector):
    return dict(shape=list(vector.shape), elements=vector.size, sha256=digest(vector))


def compare(first, second):
    if first.shape != second.shape:
        return dict(equal=False, max_abs_error=None)
    with np.errstate(invalid="ignore"):
        difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
    difference[first == second] = 0.0
    max_abs_error = float(np.max(difference))
    return dict(equal=bool(np.array_equal(first, second)),
                max_abs_error=max_abs_error if np.isfinite(max_abs_error) else str(max_abs_error))


def run(model, tokenizer, prompt, seed, topk, temperature):
    ct.set_random_seed(seed)
    engine = ct.Generator(model, device="cpu", compute_type="float32", inter_threads=1, intra_threads=1)
    pieces = tokenizer.encode(prompt, add_special_tokens=False).tokens
    callbacks = {}

    def collect(step):
        vector = step.logits
        shape = list(vector.shape) if vector is not None else None
        size = int(np.prod(shape)) if shape else 0
        array = np.asarray(vector).copy() if size else None
        callbacks.setdefault(step.step, []).append(
            (dict(batch_id=step.batch_id, token_id=step.token_id, token=step.token,
                  log_prob=step.log_prob, shape=shape, elements=size), array)
        )
        return False

    options = dict(beam_size=1, max_batch_size=0, max_length=2, sampling_temperature=temperature,
                   sampling_topk=topk, include_prompt_in_result=False, return_end_token=True,
                   return_scores=True, return_logits_vocab=True)
    final = engine.generate_batch([pieces, pieces], callback=collect, **options)
    summary = []
    for position, observations in sorted(callbacks.items()):
        final_arrays = [np.asarray(result.logits[0][position], dtype=np.float32) for result in final]
        first, first_array = observations[0]
        second, _ = observations[1]
        first["rows"] = []
        if first_array is not None:
            for index, vector in enumerate(first_array):
                first["rows"].append(dict(index=index, **summarize(vector),
                                          by_final_item=[compare(vector, v) for v in final_arrays]))
        summary.append(dict(position=position, first_callback=first, second_callback=second,
                            final_rows=[dict(batch_id=index, token_id=final[index].sequences_ids[0][position],
                                             token=final[index].sequences[0][position], **summarize(vector))
                                        for index, vector in enumerate(final_arrays)],
                            compared_final_rows=compare(*final_arrays)))
    return dict(prompt=prompt, seed=seed, options=options, steps=summary,
                distinguishable_steps=sum(not row["compared_final_rows"]["equal"] for row in summary))


if __name__ == "__main__":
    if ct.__version__ != "4.9.0":
        raise RuntimeError("This confirmation requires the released CTranslate2 4.9.0 wheel.")
    model_path, tokenizer_path = sys.argv[1:]
    tokenizer = Tokenizer.from_file(tokenizer_path)
    with Path("/tmp/native-engine-review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runs = [run(model_path, tokenizer, *configuration) for configuration in CONFIGURATIONS]
    print(json.dumps(dict(engine=ct.__version__, hash_format="SHA-256 over contiguous little-endian float32",
                          runs=runs), ensure_ascii=False, indent=2, allow_nan=False))
