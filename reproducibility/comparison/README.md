# What can a check detect, locate or recover?

To verify Tables 3–4, run the command below from the `reproducibility` directory with Python 3.11 or newer. It uses the standard library and the frozen numerical records.

```sh
python3 -B comparison/verify.py
```

This regenerates the shared checker's results and the access-specific MLX, llama.cpp/mistral.rs and CTranslate2 reports, comparing each to its archived output. [Original comparison rule](PROTOCOL.md) · [Protocol date and digest](freeze.json) · [Retrospective cutoff and marker elaboration](CUTOFF-NOTE.md) · [Shared checker](compare.py) · [Shared results](common-results.json)

The shared checker accepts an expected and reported quantity at one aligned position. Each adapter identifies its scope and score tolerance. A count or extent check establishes coverage. Identity, numerical value and byte comparisons require a reference of the corresponding type. Missing values yield abstentions. A group with one definite contradiction raises an alarm even if other positions are unknown. A group with no contradiction passes only if its positions are all determined. The top-set bound runs only when the adapter supplies a genuine same-stage top set and an ID. The exact recovery score witness is used only to grade a lookup. A different ID may have an equal score, so the full pair and numerical recovery counts are reported separately in the MLX analysis.

| Adapter | Source of reference | Unit and access |
|---|---|---|
| MLX-LM | Generation trace and candidate-disabled serializer output on exactly the same draw | Position; local token-equivalence classes. Corrected controls share that draw. Canonical-ID runs infer identity from output text with the stop convention. |
| llama.cpp | Recorded native token IDs, an independently aligned response and separate full-prefix C API scores | Native token position at one alternative-list width. The three omitted rows at width five admit a prefix score calculation but lack an original reported score. |
| mistral.rs | A unique PyTorch path compatible with the observed response | Byte equality and numerical tolerance on the prespecified 14-response subset per mode. An ambiguous or absent path leaves the field ungraded. |
| CTranslate2 | Same-step vector in singleton batch, conditional prefix calculation for stream, or result-side vector in the joint tests | Generated token or per-item callback. The final arrays share a generation with the callback. The separate decoder-input observation witnesses 11 IDs. |

In the shared output, `differ` indicates a contradiction to the supplied reference, while `abstain` also covers fields outside an adapter's inputs. The two MLX canonical groups compare IDs to the inferred sequence. The MLX-specific report scores those inferences against the native trace, including false alarms and errors on which a method abstained. An extent discrepancy localizes a missing row only if position alignment is independently available. The method-specific replays retain comparisons such as whole-text rendering, exact versus contains-replacement heuristics, empty versus wrong-width arrays, and full-sequence recovery.

Table 3 draws on the [llama.cpp and mistral.rs coverage, bytes and score analysis](others/README.md) and the [CTranslate2 shape, values and independent ID witness](native/README.md). Table 4 draws on [MLX response alarms, position-level errors, bounds and recovery](mlx/README.md).

The benchmark projections retain numerical values, position-local equality classes and measured text or byte comparisons. Reconstructing the original text, tokenization or byte values requires the retained captures. The CTranslate2 report separately uses the controlled native observations, including a short decoder witness generated from a study-authored prompt. The offline command regrades the published projections.
