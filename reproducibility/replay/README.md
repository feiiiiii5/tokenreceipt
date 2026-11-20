# Can the archived alternatives rule out a candidate-free repeat?

From the `reproducibility` directory with Python 3.11 or newer, run:

```sh
python3 -B replay/analyze.py | cmp - replay/results.json
python3 -B replay/counterexample.py
```

`analyze.py` reads 2,688 paired observations in `observations.jsonl.gz`. Six arms reuse one historical 80-prompt panel (320 draws per arm), and three arms reuse a prospectively held-out 64-prompt panel (256 draws per arm). The experimental unit for a ranking is a prompt with four draws. Treat each arm separately. `PROTOCOL.md` records selection, order and outcomes. See [capture/README.md](capture/README.md) for the pinned inputs and live rerun. The original transcript, prompt and native ID sequences stay off the distributed numerical projection.

For each pair the projection records a panel-local prompt index (`group`), draw index, transcript and finish-reason comparisons, completion counts, cached prompt-token counts, original reported scores, and scores on the candidate-free repeat. For each aligned repeat ID, `listed_score_and_cutoff` gives its archived score (or `null` when absent) and the smallest archived candidate score. The original native scores and the native-versus-repeat ID equality flag grade results only. Mean score ranks four draws and the earliest draw wins an exact tie. `corrected_winner` counts issued rankings in which the repeated winner equals the native winner and the archived outer-score winner differs. `listed_contradictions` and `cutoff_only_contradictions` explain rejections of otherwise eligible pairs by policies that check candidates.

`scorecheck.py original.json repeat.json` checks a pair of raw, complete responses from the pinned MLX-LM 0.32.2 plain-text endpoint under this protocol. A failure rules out reusing the repeat as an exact account of the original. A pass means its scores are compatible with the visible constraints. `counterexample.py` uses the released serializer on the two normalized distributions from the paper: four compatible repeats change the best draw from A to B. The five alternatives must be genuine top scores from the same numerical stage for either equality or the unlisted-score ceiling to apply.
