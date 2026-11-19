"""Recompute the primary SST-2 endpoint and paired bootstrap intervals."""
import json
from pathlib import Path

import numpy as np


EVIDENCE = Path(__file__).resolve().parent / "evidence"


def crossed(matrix, labels, draws):
    rng = np.random.default_rng(79037)
    nseeds, nitems = matrix.shape
    classes = [np.flatnonzero(labels == label) for label in (0, 1)]
    results = np.empty(draws)
    for start in range(0, draws, 250):
        end = min(start + 250, draws)
        size = end - start
        seed_weights = rng.multinomial(nseeds, np.repeat(1 / nseeds, nseeds), size=size) / nseeds
        item_weights = np.zeros((size, nitems))
        for group in classes:
            n = len(group)
            item_weights[:, group] = rng.multinomial(n, np.repeat(1 / n, n), size=size) / nitems
        results[start:end] = ((seed_weights @ matrix) * item_weights).sum(axis=1)
    return np.quantile(results, [.025, .975])


def main():
    evidence = json.loads((EVIDENCE / "sst2-expected.json").read_bytes())
    with np.load(EVIDENCE / "sst2-endpoints.npz", allow_pickle=False) as data:
        labels = data["labels"]
        seeds = data["seeds"]
        initial, selected, outer = (np.exp(data[key]) for key in ("initial", "selected", "outer"))
    shape = (evidence["n_seeds"], evidence["n_heldout"])
    assert initial.shape == selected.shape == outer.shape == shape == (128, 384)
    assert len(set(seeds)) == shape[0] and np.bincount(labels).tolist() == [192, 192]
    draws = evidence["n_bootstrap"]
    sample = np.random.default_rng(79031).integers(shape[0], size=(draws, shape[0]))
    names = ("fault_minus_reference_expected", "reference_learning_expected", "fault_learning_expected")
    out = {}
    for name, matrix in zip(names, (outer - selected, selected - initial, outer - initial), strict=True):
        expected = evidence["contrasts"][name]
        per_seed = matrix.mean(axis=1)
        mean = float(matrix.mean())
        seed_interval = np.quantile(per_seed[sample].mean(axis=1), [.025, .975])
        crossed_interval = crossed(matrix, labels, draws)
        np.testing.assert_allclose(per_seed, expected["per_seed"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(mean, expected["mean"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(seed_interval, expected["seed_only_interval_95"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(crossed_interval, expected["crossed_interval_95"], rtol=0, atol=1e-12)
        out[name] = {"mean": mean, "seed_only_interval_95": seed_interval.tolist(),
                     "crossed_interval_95": crossed_interval.tolist()}
    for key, matrix in zip(("initial", "selected_final", "outer_final"), (initial, selected, outer), strict=True):
        np.testing.assert_allclose(matrix.mean(), evidence["expected_correctness"][key], rtol=0, atol=1e-12)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
