"""Compare one native batch result with both forward-pass conventions in a fresh process."""
import argparse
import json
from pathlib import Path
import ctranslate2 as ct
import numpy as np
from tokenizers import Tokenizer


def vector(storage):
    return np.asarray(storage).reshape(-1).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("tokenizer", type=Path)
    args = parser.parse_args()
    assert ct.__version__ == "4.9.0"
    generator = ct.Generator(str(args.model), device="cpu", compute_type="float32",
                             inter_threads=1, intra_threads=1)
    prompt = Tokenizer.from_file(str(args.tokenizer)).encode("The book on the table was", add_special_tokens=False)
    callbacks = []

    def observe(row):
        callbacks.append((row.token_id, row.token, row.log_prob, vector(row.logits)))
        return False

    ct.set_random_seed(1701)
    result, = generator.generate_batch([prompt.tokens], beam_size=1, max_length=2,
                                        sampling_topk=5, sampling_temperature=0.7,
                                        include_prompt_in_result=False, return_scores=True,
                                        return_logits_vocab=True, callback=observe)
    token, piece, selected, scores = callbacks[0]
    raw, normalized = (np.asarray(generator.forward_batch([prompt.ids], return_log_probs=flag))[0,-1,:].astype(np.float64)
                       for flag in (False, True))
    z = float(np.logaddexp.reduce(scores))
    final = vector(result.logits[0][0])
    summary = dict(token_id=token, piece=piece, selected_score=selected, looked_up=float(scores[token]),
                   logsumexp=z, positive_entries=int(np.count_nonzero(scores > 0)), vocabulary=int(scores.size),
                   max_raw_difference=float(np.max(np.abs(raw-scores))),
                   max_normalized_difference=float(np.max(np.abs(normalized-scores))),
                   max_difference_after_normalization=float(np.max(np.abs((scores-z)-normalized))),
                   selected_difference_after_normalization=float(abs(selected-(scores[token]-z))),
                   callback_vs_final_max_difference=float(np.max(np.abs(final-scores))),
                   selected_id_matches_final=result.sequences_ids[0][0]==token,
                   generator_help=next(line.strip() for line in ct.Generator.generate_batch.__doc__.splitlines() if line.strip().startswith('return_logits_vocab:')),
                   step_help=ct.GenerationStepResult.logits.__doc__,
                   final_help=ct.GenerationResult.logits.__doc__)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
