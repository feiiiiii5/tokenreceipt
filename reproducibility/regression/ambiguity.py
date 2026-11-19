"""A normalized two-world witness for nonidentifiability of a four-way ranking."""
from fractions import Fraction
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
import json
from math import log
from pathlib import Path
from types import SimpleNamespace


def witness(artifact):
    path = artifact / "regression/check.py"
    spec = spec_from_file_location("regression", path)
    regression = module_from_spec(spec)
    spec.loader.exec_module(regression)
    serialize = regression.load_serializer(artifact / "releases/mlx-0322-server.py")
    tokenizer = json.loads((artifact / "provenance/ambiguity-vocab.json").read_bytes())
    vocab = tokenizer["token_ids"]
    vocabulary_size = tokenizer["vocabulary_size"]
    order = "EFGHIABCD"
    shared_draws = [Fraction(i, 256) for i in (225, 240, 248, 253)]
    fields = SimpleNamespace(request_id="fixed", system_fingerprint="fixed",
                             object_type="chat.completion", requested_model="fixed", created=0, stream=False)
    worlds = []
    packets = []
    for hidden in ((13,9,7,4), (9,13,7,4)):
        masses = [Fraction(i,256) for i in (64,52,43,35,28,*hidden)]
        per_remaining_id = Fraction(1,256*(vocabulary_size-len(order)))
        assert sum(masses) + (vocabulary_size-len(order))*per_remaining_id == 1
        assert min(masses[:5]) > max(*masses[5:],per_remaining_id)
        candidates = [dict(id=vocab[t],token=t,logprob=log(p)) for t,p in zip(order[:5],masses[:5])]
        body, scores, chosen = [], [], []
        for u in shared_draws:
            index = next(i for i in range(len(order)) if sum(masses[:i+1]) > u)
            t = order[index]
            s = log(masses[index])
            chosen.append(t)
            scores.append(s)
            body.append(serialize(fields, t, "length", prompt_token_count=1, completion_token_count=1,
                                  tokens=[vocab[t]],token_logprobs=[s],top_tokens=[candidates]))
        assert chosen == list("ABCD")
        worlds.append(dict(chosen=chosen,scores=scores,winner=chosen[scores.index(max(scores))]))
        packets.append(body)
    assert packets[0] == packets[1] and worlds[0]["winner"] != worlds[1]["winner"]
    digest = sha256(json.dumps(packets[0],sort_keys=True).encode()).hexdigest()
    return dict(worlds=worlds,identical_complete_response_objects=len(packets[0]),response_set_sha256=digest)


if __name__ == "__main__":
    print(json.dumps(witness(Path(__file__).resolve().parents[1]),indent=2))
