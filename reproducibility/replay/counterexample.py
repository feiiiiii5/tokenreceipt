"""A compatible candidate-free repeat can reverse all the information needed to rank A vs B."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from math import log
from pathlib import Path
from types import SimpleNamespace
from scorecheck import check


def example(supplement):
    spec = spec_from_file_location('serializer_fixture', supplement/'regression/check.py')
    fixture = module_from_spec(spec)
    spec.loader.exec_module(fixture)
    serialize = fixture.load_serializer(supplement/'releases/mlx-0322-server.py')
    vocab = json.loads((supplement/'provenance/ambiguity-vocab.json').read_bytes())['token_ids']
    meta = SimpleNamespace(request_id='test',system_fingerprint='test',object_type='chat.completion',requested_model='test',created=0,stream=False)
    alts = [dict(id=vocab[t],token=t,logprob=log(m/256)) for t,m in zip('EFGHI',(64,52,43,35,28))]
    # These are the same two normalized distributions and shared draws as regression/ambiguity.py.
    original_scores, repeat_scores = (13,9,7,4), (9,13,7,4)
    answers = []
    for t, a, b in zip('ABCD',original_scores,repeat_scores,strict=True):
        args = dict(prompt_token_count=1,completion_token_count=1,tokens=[vocab[t]])
        original = serialize(meta,t,'length',**args,token_logprobs=[log(a/256)],top_tokens=[alts])
        repeat = serialize(meta,t,'length',**args,token_logprobs=[log(b/256)],top_tokens=[])
        answers.append(check(original,repeat))
    assert all(answers)
    return dict(compatible=answers,original_winner='ABCD'[original_scores.index(max(original_scores))],
                repeated_winner='ABCD'[repeat_scores.index(max(repeat_scores))])


if __name__ == '__main__':
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    print(json.dumps(example(root)))
