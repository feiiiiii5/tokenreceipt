# MLX-LM 0.32.2 comparison on frozen IFEval responses

The panel contains 64 requests to SmolLM2-135M-Instruct at temperature 0.8, a 24-position cap, and five candidates. Each request supplies one emission trace, the original object before correction, the corrected HTTP object, and a same-event candidate-disabled serialization. These are 64 paired requests and 1,488 paired positions. Before correction, 528 IDs and 521 selected scores were wrong, including seven wrong IDs with numerically equal scores. The corrected control has 1,488 correct pairs. Scores use exact equality on the normalized log probabilities captured before the sampler applies temperature. The corrected control restores the same sampled values and retains the same candidate list, display and usage.

## Response-level checks

| Check | Original alarms / 64 | Corrected false alarms / 64 | Input |
| --- | ---: | ---: | --- |
| Paired display equality | 0 | 0 | Text from the two same-event objects |
| Count versus reported completion usage | 0 | 0 | Probability row count and usage from one object |
| Literal tokenizer decoding of the reported IDs | 64 | 4 | Text, IDs and tokenizer |
| Tokenizer decoding with special tokens suppressed | 64 | 0 | Text, IDs and tokenizer |
| Re-render under the release's EOS convention | 64 | 0 | Text, IDs, finish reason, EOS configuration and release detokenizer |

The released BPE rendering check recreates the stock plain-text output from the native ID sequence in all 64 controls. It omits terminal EOS text on a stop, flushes buffered text, and checks the consistency of any supplied EOS ID with the recorded finish. The simpler special-token-suppression check was a post-hoc control in the original experiment. Both rendering variants have response-level resolution. The four false alarms in the literal tokenizer check occur on the natural stops whose terminal IDs appear in the metadata while the display omits the EOS text.

## Position-level comparisons

A localization is a claim about an original outer ID or numerical score at a specific position. Canonical methods infer IDs by encoding the displayed text. With the stop convention, they append the configured EOS ID on a stop. They abstain on every position when the resulting length differs from the reported row count. The candidate-assisted methods look up the requested ID in that position's five candidates. An absent candidate leaves the score and pair unknown, even on positions where the corrected outer record already provides the true score. The candidate-disabled method compares the recorded same-event ID and score against the outer fields. Its inputs share the producer and sampling event with the native trace.

| Method | True ID errors localized / 528 (equal-score subset / 7) | True score errors localized / 521 | Original pairs correct / wrong / abstained, all 1,488 positions | Original pairs correct / wrong / abstained, 528 erroneous positions |
| --- | ---: | ---: | ---: | ---: |
| Read the outer fields | — | — | 960 / 528 / 0 | 0 / 528 / 0 |
| Literal canonical IDs and top five | 503 (6) | 301 | 1,220 / 0 / 268 | 307 / 0 / 221 |
| Stop-aware canonical IDs and top five | 518 (6) | 309 | 1,261 / 0 / 227 | 315 / 0 / 213 |
| Emitted IDs and top five | 528 (7) | 317 | 1,284 / 0 / 204 | 324 / 0 / 204 |
| Same-event candidate-disabled pairs | 528 (7) | 521 | 1,488 / 0 / 0 | 528 / 0 / 0 |

All unlocalized errors in the table's diagnostic rows were abstentions. All those methods made zero false localization claims on the original clean slots and corrected controls. The stop-aware canonicalizer aligned in 63/64 responses and empirically recovered every ID in those responses. Its 227 pair abstentions comprise 24 positions in the unequal-length response and 203 candidate omissions among the aligned positions. Emitted IDs reduce these losses to the 204 omissions. Both stop-aware canonical and emitted-ID lookups fully recover 9/64 original responses. Their clean-control pair recovery totals equal their original totals because the two objects supply identical displays and candidates. The full same-event check recovers all 64 responses.

`result.json` reports false claims and abstentions separately, along with standalone ID and numeric score recovery. Score recovery compares the scalar. Exact pair recovery also requires the correct ID. A reconstruction's response alarm fires when any local ID or score claim finds a discrepancy. When all available claims agree and at least one slot is unknown, that reconstruction abstains at response level. A `null` method result indicates a capability outside the read-through baseline. The prompt set and prior analysis were already known. The measured canonical ID agreement applies to this panel. Multiple ID sequences can render the same text.

## Score detection using the candidate cutoff

After the initial exact-lookup comparison, we added the one-sided cutoff already used in the earlier manuscript for archival ranking and repeat validation. When a supplied ID appears in the five candidates, compare its exact listed score. For an absent ID, a reported score strictly above the lowest candidate score is inconsistent with the supplied ID. A score at or below the cutoff leaves equality undetermined. The rule requires five genuine top scores at the same stage. For the canonical method the conclusion also depends on the inferred ID.

| ID source | Original score-error alarms / 64 | Original ID errors localized / 528 | Original score errors localized / 521 | Corrected score false alarms, responses / positions |
| --- | ---: | ---: | ---: | ---: |
| Stop-aware canonical IDs | 63 | 518 | 512 | 0 / 0 |
| Emitted IDs | 64 | 528 | 521 | 0 / 0 |

The cutoff identifies 203 additional wrong scores with canonical IDs and 204 with emitted IDs. The nine scores missed by the canonical method all occur in its unaligned response. On the corrected control, each method has nine fully checked responses and 55 responses with at least one undetermined score. ID localization and exact pair recovery retain their counts from the previous table. The bound certifies a discrepancy in the unlisted cases while leaving the replacement score undetermined. `result.json` stores these results under `score_cutoff_extension` and links to the corresponding exact reconstruction for recovery counts.

## Offline replay

From the `reproducibility` directory with Python 3.11 or newer:

```sh
python3 -B comparison/mlx/analyze.py evidence/ifeval.jsonl | diff -u comparison/mlx/result.json -
```

`evidence/ifeval.jsonl` is the existing frozen projection (SHA-256 `88aa0f0bca069125f55e0ef718d3f4fc5e121ae2ee9d5ee7cc4dc46a6fd38a98`). It stores the native, original and corrected pairs and five alternatives, with token ID equivalence classes local to each position. `observations.jsonl` adds display equality, row count and renderer mismatch flags, plus canonical per-position ID classes and encoded lengths. `-1` indicates a canonical ID unequal to every already projected ID at that position. `null` marks an unaligned sequence. The records contain no transcript, prompt, vocabulary IDs, absolute paths or request metadata. The renderer flags are derived from retained text and IDs, so the distributed replay checks their counts and joins. It cannot independently repeat rendering from those flags.

## Rebuilding the additional observations on the retained deposit

`project.py` uses `tokenizers==0.22.2`, the pinned SmolLM2 tokenizer, and the `StreamingDetokenizer`, `_byte_decoder`, and `BPEStreamingDetokenizer` definitions from released MLX-LM 0.32.2 `mlx_lm/tokenizer_utils.py`. It verifies tokenizer and release source hashes, the same-event response identifiers, native and candidate-disabled equality, raw classes and numerical equality behind the frozen projection, and the unchanged candidate lists. Both styles of the earlier tokenizer control independently reproduce their frozen outputs using `experiments/scripts/text_oracle.py`. Released serializer replay with `experiments/scripts/check.py` independently reproduces the frozen panel and checks the actual correction. The source inputs are `experiments/data/ifeval/{key}.{native,before,http,counterfactual}.json`. No model generation is required.

```sh
python3 -B comparison/mlx/project.py /path/to/experiments evidence/ifeval.jsonl /path/to/mlx_lm/tokenizer_utils.py \
  | diff -u comparison/mlx/observations.jsonl -
```
