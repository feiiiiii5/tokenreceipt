"""Grade a frozen numeric projection. The decision rule sees only the observable fields."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean
import sys

POLICIES = ('display','display+cache','display+candidates','display+candidates+cache')


def summarize(rows):
    totals = {policy: dict(requests=0, accepted=0, wrong_pair=0, wrong_scores=0,
                           rejected_exact_eligible=0, listed_contradictions=0, cutoff_only_contradictions=0,
                           groups=0, issued=0, wrong_winner=0,
                           corrected_winner=0) for policy in POLICIES}
    groups = defaultdict(dict)
    for rec in rows:
        old, new = rec['original_scores'], rec['repeat_scores']
        n, m = len(old), len(new)
        assert n and m and len(rec['reported_scores']) == n
        display = (rec['same_text'] and rec['same_finish'] and n == m == rec['usage'][0] == rec['usage'][1])
        cache = rec['cache'][0] == rec['cache'][1]
        # For each aligned repeat ID, the projection carries the original cutoff
        # and its exact score when listed. Candidate-free scores come from new.
        matches = rec['listed_score_and_cutoff']
        assert len(matches) == min(n,m)
        listed_error = any(known is not None and score != known
                           for score, (known, _) in zip(new, matches))
        cutoff_error = any(known is None and score > cutoff
                           for score, (known, cutoff) in zip(new, matches))
        agrees = not (listed_error or cutoff_error)
        accepts = (display, display and cache, display and agrees, display and agrees and cache)
        exact_scores = old == new
        exact_pair = exact_scores and rec['same_ids']
        for policy, allowed in zip(POLICIES, accepts, strict=True):
            counts = totals[policy]
            counts['requests'] += 1
            if 'candidates' in policy and display and ('cache' not in policy or cache):
                counts['listed_contradictions'] += listed_error
                counts['cutoff_only_contradictions'] += cutoff_error and not listed_error
            counts['accepted'] += allowed
            counts['wrong_pair'] += allowed and not exact_pair
            counts['wrong_scores'] += allowed and not exact_scores
            counts['rejected_exact_eligible'] += display and exact_pair and not allowed
        key, draw = rec['group'], rec['draw']
        assert draw not in groups[key]
        groups[key][draw] = (accepts, mean(old), mean(new), mean(rec['reported_scores']))
    for group in groups.values():
        assert set(group) == set(range(4))
        original, repeated, reported = [max(range(4), key=lambda i: group[i][col]) for col in (1,2,3)]
        for i, policy in enumerate(POLICIES):
            counts = totals[policy]
            counts['groups'] += 1
            if all(row[0][i] for row in group.values()):
                counts['issued'] += 1
                counts['wrong_winner'] += repeated != original
                counts['corrected_winner'] += repeated == original != reported
    return totals


def audit(path):
    lanes = defaultdict(list)
    with gzip.open(path,'rt') as stream:
        for line in stream:
            rec = json.loads(line)
            lanes[rec['lane']].append(rec)
    return {lane:summarize(rows) for lane, rows in lanes.items()}


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else here/'observations.jsonl.gz'
    print(json.dumps(audit(path), indent=2))
