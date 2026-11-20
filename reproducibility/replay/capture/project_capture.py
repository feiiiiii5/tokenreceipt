"""Project locally captured journals into the numerical replay analysis format."""
import argparse
import gzip
import json
from pathlib import Path
import sys

from capture import REPLAY_ARMS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scorecheck import check


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        yield from map(json.loads, source)


def view(record):
    packet = record["wire"]
    choice = packet["choices"][0]
    return choice, choice["logprobs"]["content"], packet["usage"]


def pair(positions):
    return [(row["id"], row["logprob"]) for row in positions]


def project(original, repeat, lane, group):
    expected_request = original["request"].copy()
    assert expected_request.pop("top_logprobs") == 5 and expected_request == repeat["request"]
    first, reported, first_usage = view(original)
    second, selected, second_usage = view(repeat)
    native = original["native"]
    plain = original["plain"]["choices"][0]["logprobs"]["content"]
    assert pair(native) == pair(plain) == pair(original["source"])
    assert len(reported) == len(native) == first_usage["completion_tokens"]
    candidate_matches = []
    for rich in reported:
        candidates = rich["top_logprobs"]
        assert len(candidates) == len({row["id"] for row in candidates}) == 5
        assert pair([rich]) == pair(candidates[:1])
    for rich, thin in zip(reported, selected):
        scores = dict(pair(rich["top_logprobs"]))
        candidate_matches.append([scores.get(thin["id"]), min(scores.values())])
    display = first["message"]["content"] == second["message"]["content"]
    finish = first["finish_reason"] == second["finish_reason"]
    counts = [first_usage["completion_tokens"], second_usage["completion_tokens"]]
    cache = [info["prompt_tokens_details"]["cached_tokens"] for info in (first_usage, second_usage)]
    replay_scores = [pos["logprob"] for pos in selected]
    eligible = bool(native) and display and finish and len(selected) == len(native) == counts[0] == counts[1]
    allowed = eligible and all(score == known if known is not None else score <= cutoff
                               for score, (known, cutoff) in zip(replay_scores, candidate_matches, strict=True))
    assert allowed == check(original["wire"], repeat["wire"])
    return dict(lane=lane, group=group, draw=repeat["draw"], same_text=display, same_finish=finish,
                same_ids=[pos["id"] for pos in native] == [pos["id"] for pos in selected],
                usage=counts, cache=cache, original_scores=[pos["logprob"] for pos in native],
                repeat_scores=replay_scores, reported_scores=[pos["logprob"] for pos in reported],
                listed_score_and_cutoff=candidate_matches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    originals = {(record["key"], record["draw"]): record for record in rows(args.captures / "captures.jsonl.gz")}
    keys = json.loads(Path(__file__).with_name("keys.json").read_bytes())
    assert list(originals) == [(key, draw) for key in keys for draw in range(4)]
    groups = {key: index for index, key in enumerate(keys)}
    with gzip.open(args.output, "xt", encoding="utf-8") as output:
        for arm in REPLAY_ARMS:
            order = reversed(originals) if arm == "reverse-default" else iter(originals)
            for row, key in zip(rows(args.captures / f"{arm}.jsonl.gz"), order, strict=True):
                assert (row["key"], row["draw"]) == key
                projected = project(originals[key], row, f"prospective-{arm}", groups[key[0]])
                print(json.dumps(projected, separators=(",", ":")), file=output)


if __name__ == "__main__":
    main()
