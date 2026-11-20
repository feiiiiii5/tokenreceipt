"""Check optional score-vector coverage for two identical prompts in one caller's batch."""
import json
import sys
from pathlib import Path
import ctranslate2 as ct
import numpy as np
from tokenizers import Tokenizer


def run(model, tokenizer, prompt, copies, split, seed, k, temp):
    ct.set_random_seed(seed)
    engine = ct.Generator(model, device="cpu", compute_type="float32", inter_threads=1, intra_threads=1)
    pieces = tokenizer.encode(prompt, add_special_tokens=False).tokens
    records = []

    def callback(step):
        blob = step.logits
        records.append(dict(item=step.batch_id, step=step.step, token_id=step.token_id,
                            piece=step.token, log_prob=step.log_prob,
                            optional_present=blob is not None,
                            shape=list(blob.shape) if blob is not None else None,
                            elements=int(blob.num_elements()) if callable(getattr(blob,'num_elements',None))
                            else (int(np.prod(blob.shape)) if blob is not None and len(blob.shape) else 0)))
        return False

    final = engine.generate_batch([pieces] * copies, beam_size=1, max_batch_size=split, max_length=2,
                                  sampling_temperature=temp, sampling_topk=k, include_prompt_in_result=False,
                                  return_end_token=True, return_scores=True, return_logits_vocab=True,
                                  callback=callback)
    for row in records:
        result = final[row['item']]
        pos = row['step']
        vec = np.asarray(result.logits[0][pos]).reshape(-1).astype(np.float64)
        idx = result.sequences_ids[0][pos]
        row.update(final_id=idx, final_piece=result.sequences[0][pos], final_vocab=vec.size,
                   final_chosen_logp=float(vec[idx] - np.logaddexp.reduce(vec)),
                   final_count=len(result.sequences_ids[0]))
    return dict(copies=copies,split=split,seed=seed,topk=k,prompt=prompt,rows=records)


if __name__=='__main__':
    assert ct.__version__=='4.9.0'
    model, tok = sys.argv[1:]
    tokenizer = Tokenizer.from_file(tok)
    result = [run(model,tokenizer,'The book on the table was',copies,split,1701,1,1.0)
              for copies,split in [(1,0),(2,0),(2,1)]]
    result.extend(run(model,tokenizer,'In a quiet garden, the',2,split,2701,5,0.7) for split in (0,1))
    print(json.dumps(result,indent=2))
