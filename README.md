# TokenReceipt

[When a Language Model Server Reports a Different Token](paper/preprint/tokenreceipt.pdf) · Yufeiyang Chen

A language model samples one token, but the inference server can attach a different token and probability to it in the API response. TokenReceipt tests this on the same sampling event, traces the error into answer selection and policy updates, and provides a patch and regression test for MLX-LM.

On 64 prompts, MLX-LM reported a different token at 528 of 1,488 positions and a different score at 521. Rendering the reported IDs flagged all 64 faulty responses and none of their corrected controls. Re-encoding the text with the server's stop convention and checking the top-five score cutoff located 512 of the 521 wrong scores. With the actual sampled IDs, the cutoff located all 521, but the top five still omitted 204 exact scores. The same-event record recovered all 1,488 token–score pairs.

In a held-out test of CTranslate2 4.9.0, all 360 selected scores on each of its streaming and batch paths passed the predefined check. A follow-up test on two-item batches found a different fault. At each tested step the first callback received both vocabulary rows and the second received none. Checking the size of each callback's vector located all eight affected callbacks. Tables 3 and 4 compare what the checks can flag, locate and recover across MLX-LM, llama.cpp, mistral.rs and CTranslate2.

On 2,108 held-out science questions, scoring the same four answers using the sampled or reported probabilities selected different answer texts on 846 questions. In a controlled online sentiment task the affected learner lost 1.90 percentage points of expected correctness across 128 training seeds. A second workload showed changes to decoder updates, with no established loss in task performance.

- Read the [signed paper](paper/preprint/tokenreceipt.pdf) or [build the LaTeX source](paper/preprint/BUILD.md).
- Run the [offline replication](reproducibility/README.md) for the release census, patch, answer-selection results and learning endpoint. The [repeated-request study](reproducibility/replay/), [CTranslate2 test](reproducibility/ctranslate2/) and [four-engine comparison](reproducibility/comparison/) include their protocols and replay instructions. The [materials inventory](reproducibility/MATERIALS.md) identifies the sources, included data and reconstruction steps.
- Use the [wire recovery tool](tools/README.md) to test how much ranking information can be recovered from saved server responses.

The public data contain numerical observations and within-position equality classes. Source text for restricted or unclear-license datasets can be reconstructed from the upstream sources using the documented selection rules. See the [licenses and source notices](reproducibility/licenses/).
