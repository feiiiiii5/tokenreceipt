# CTranslate2 oracle comparison

This comparison reuses the frozen CTranslate2 4.9.0 CPU/float32 panel, its existing source-directed callbacks, the separate decoder-input observation, and the original score-only injection. See the [comparison protocol](../PROTOCOL.md) and [recomputed counts](results.json).

## Clean singleton panel

| Check | Batch, 30 calls and 360 positions | Stream, 30 calls and 360 positions |
|---|---:|---:|
| Rendered-text agreement and terminal count | 30/30 each | 30/30 each |
| Canonical pieces and derived bytes | 360/360 each | 360/360 each |
| Directly exposed bytes | 360 abstain | 360 abstain |
| Width of callback vocabulary values | 360/360 | 360 abstain |
| Selected scalar agrees with same-step log-softmax | 360/360 | 360 abstain |
| Selected scalar agrees with exact-prefix forward log-softmax | 360/360 | 360/360 |
| Public final ID, piece, and vector agree with callback | 360/360 each | 360 abstain |
| Independent next-decoder ID observation | 360 abstain | 360 abstain |

The forward check separately computes the score conditional on the prior exposed IDs. The same-step and final vectors share the underlying generation values. The stream iterator and its callback shared one object and agreed at 360/360 positions. The batch projection preserves flattened width and full-vector equality flags. All original singleton calls reached the 12-position cap. The comparison uses the reported full-vocabulary log-softmax scale, including the temperature/top-k lane. Direct selected-logit lookup would incorrectly flag 360/360 clean vector-enabled positions as selected-scalar mismatches.

## Live callback coverage and final results

Each joint call had two identical prompts and two positions per item. The two joint configurations yielded four affected positions with a `2 × 49,152` matrix in item 0 and four with an empty field in item 1.

| Live coverage check | Joint, 8 affected positions | Split, 8 clean positions | Singleton, 2 clean positions |
|---|---:|---:|---:|
| Vocabulary field object exists | 0 alarms, 8 misses | 0 false alarms | 0 false alarms |
| Vocabulary field has elements | 4 alarms at item 1, 4 misses | 0 false alarms | 0 false alarms |
| Exactly one vocabulary row for the callback item | 8 alarms at the affected `logits` slots | 0 false alarms | 0 false alarms |

All four joint items passed step/final counts and all eight positions passed public callback-to-final ID and piece equality. The short-run projection holds pieces and IDs with their final counterparts. It supplies no separate rendered-text or literal-byte measurement. The live per-item callback vector could supply a selected-score reference at 0/8 joint slots under the one-row access rule. The split and singleton callbacks have the correct live shape. Their captured summaries carry no selected callback logit or normalizer, so callback-side numerical grading abstains there too.

Final results supplied all eight joint rows, IDs and corresponding projected normalized selected scores. These recovered the selected scalar within tolerance of its original callback value at 8/8 positions, with ID equality from the shared source. Whole-vector recovery here means availability at the final API. A separate alignment run repeated the two joint configurations. The nonempty check flagged 4/8 positions and the strict shape check flagged 8/8 positions again. Eight matrix rows matched the corresponding final row. The two final rows were identical on all four paired steps, so the values alone leave item routing unidentifiable at 4/4 steps. The packaged source audit identifies the order and the two delivery paths. Both ID fields share an origin.

## Original score-only injection and decoder observation

The existing copied injection changes `batch-1-neutral-1701` position 0, keeping ID 2268 and substituting the top-one score for ID 253. Text, count, label, derived bytes, width and final ID checks still pass. The batch aggregate flags the response without localizing a position. Same-event log-softmax, the corresponding final vector and exact-prefix forward each flag only `log_prob` at position 0. The other 11 positions pass these checks. Same-event normalization recovers −4.7270727 and exact-prefix forward recovers −4.7270661, both within tolerance of the clean scalar −4.7270393. The injected scalar is −1.7539415. The injection supplies a label for scalar localization and recovery.

The separate instrumented call compares public IDs with the inputs actually consumed at the next decoder step. Eleven IDs agree, including eight off-first and six outside the top five. At the cap, the final position has no next-step input. This observation supplies 11 determinate ID checks and one abstention for that call.

## Repeat offline

From the `reproducibility` directory with Python 3.11 or newer:

```sh
python3 -B comparison/native/recount.py ctranslate2 | diff -u comparison/native/results.json -
```

The driver reads `evidence.json`, `audit.json`, `batch-control.json`, `batch-alignment.json`, `witness/decoder-inputs.json`, and `witness/run-001/response.json` from that directory. Computation uses the prior projection of logits, log-normalizers, shape measurements and equality evidence. It uses absolute tolerance `0.0002 + 0.00002 * abs(reference)` nats. Full-vector recomputation requires the local capture described in the original CTranslate2 package.
