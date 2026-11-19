# TokenReceipt

[When a Language Model Server Reports a Different Token](paper/preprint/tokenreceipt.pdf) · Yufeiyang Chen

A language model samples one token, but the inference server can attach a different token and probability to it in the API response. TokenReceipt tests this on the same sampling event, traces the error into answer selection and policy updates, and provides a patch and regression test for MLX-LM.

Across 64 prompts on one tested model, the API reported a different token at 528 of 1,488 positions. On 2,108 held-out science questions, scoring the same four answers using the sampled or reported probabilities selected different answer texts on 846 questions. In a controlled online sentiment task the affected learner lost 1.90 percentage points of expected correctness across 128 training seeds. A second workload showed changes to decoder updates, with no established loss in task performance.

- Read the [signed paper](paper/preprint/tokenreceipt.pdf) or [build the LaTeX source](paper/preprint/BUILD.md).
- Run the [offline replication](reproducibility/README.md) for the release census, patch, answer-selection results, and learning endpoint. The [materials inventory](reproducibility/MATERIALS.md) identifies the sources, included data, and reconstruction steps.
- Use the [wire recovery tool](tools/README.md) to test how much ranking information can be recovered from saved server responses.

The public data contain numerical observations and within-position equality classes. Source text for restricted or unclear-license datasets can be reconstructed from the upstream sources using the documented selection rules. See the [licenses and source notices](reproducibility/licenses/).
