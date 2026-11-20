"""Capture the frozen first prompt's native callback and final result under LLDB."""

import hashlib
import json
import sys
from pathlib import Path

import ctranslate2
import numpy as np
from transformers import AutoTokenizer


def main():
    model, tokenizer_source, output = map(Path, sys.argv[1:])
    native_library = Path(ctranslate2.__file__).parent / ".dylibs/libctranslate2.4.9.0.dylib"
    with native_library.open("rb") as file:
        native_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    expected_sha256 = "18d0cd66270146b7b07436e5f1eae10641d0dbe17ae4d3b5e43cd246fdd89e23"
    if native_sha256 != expected_sha256:
        raise RuntimeError(f"Unexpected CTranslate2 native library hash {native_sha256}")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), local_files_only=True)
    prompt = "The book on the table was"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
    rows = []

    def callback(step):
        vector = np.asarray(step.logits).reshape(-1)
        rows.append({
            "step": step.step, "token_id": step.token_id, "token": step.token,
            "log_prob": step.log_prob, "is_last": step.is_last,
            "rank": 1 + int(np.count_nonzero(vector > vector[step.token_id])),
            "argmax_id": int(np.argmax(vector)),
        })
        return False

    seed = 1701
    ctranslate2.set_random_seed(seed)
    generator = ctranslate2.Generator(
        str(model), device="cpu", compute_type="float32", inter_threads=1, intra_threads=1,
    )
    result = generator.generate_batch(
        [prompt_tokens], beam_size=1, max_length=12,
        sampling_temperature=1, sampling_topk=0, sampling_topp=1,
        include_prompt_in_result=False, return_scores=True,
        return_logits_vocab=True, callback=callback,
    )[0]
    evidence = {
        "wheel_version": ctranslate2.__version__, "native_dylib_sha256": native_sha256,
        "seed": seed, "prompt": prompt, "prompt_ids": prompt_ids,
        "callback_rows": rows, "final_sequence_ids": result.sequences_ids[0],
        "final_pieces": result.sequences[0], "final_score": result.scores[0],
    }
    output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
