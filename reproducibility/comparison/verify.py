"""Recompute all four comparison reports from the public evidence."""
import subprocess
import sys
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    supplement = here.parent
    jobs = [
        ('common-results.json', ['compare.py', supplement]),
        ('mlx/result.json', ['mlx/analyze.py', supplement / 'evidence/ifeval.jsonl']),
        ('others/results.json', ['others/compare.py']),
        ('native/results.json', ['native/recount.py', supplement / 'ctranslate2']),
    ]
    for expected, args in jobs:
        actual = subprocess.check_output([sys.executable, '-B', *map(str, args)], cwd=here)
        if actual != (here/expected).read_bytes():
            raise ValueError(f'Recomputed report differs from {expected}')
        print(f'{expected} matches')


if __name__ == '__main__':
    main()
