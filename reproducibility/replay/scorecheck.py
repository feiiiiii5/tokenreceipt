"""Compare two ordinary, aligned MLX-LM 0.32.2 chat responses from the same request and seed.

Original requests return five true top scores; candidate-free repeats return
selected IDs and scores at the same raw model stage. A compatible repeat can
still differ from the original draw. Input is one complete JSON body per file.
"""
import argparse
import json
from pathlib import Path


def compatible(candidates, token_id, score):
    if len(candidates) != 5:
        raise ValueError('Expected five distinct candidate IDs')
    if token_id in candidates:
        return score == candidates[token_id]
    return score <= min(candidates.values())


def check(original, repeat):
    first, second = (body['choices'][0] for body in (original, repeat))
    rows, selected = (choice['logprobs']['content'] for choice in (first, second))
    display = (first['message']['content'] == second['message']['content'] and
               first['finish_reason'] == second['finish_reason'] and
               len(rows) == len(selected) == original['usage']['completion_tokens'] == repeat['usage']['completion_tokens'] and bool(rows))
    if not display:
        return False
    return all(compatible({p['id']:p['logprob'] for p in row['top_logprobs']}, pos['id'], pos['logprob'])
               for row, pos in zip(rows, selected, strict=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('repeat', type=Path)
    args = parser.parse_args()
    print(json.dumps({'compatible':check(json.loads(args.original.read_bytes()), json.loads(args.repeat.read_bytes()))}))
