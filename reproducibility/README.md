# Reproducing the TokenReceipt results

This public package accompanies *When a Language Model Server Reports a Different Token*. It includes runnable component tests, numerical projections of recorded experiments, and instructions for reconstructing selected upstream inputs. The November 19 revision adds offline evidence for the replay study, the prospective CTranslate2 test and the frozen comparisons in Tables 3–4.

## Offline replay

Run these commands here with Python 3.11 or later and Git. They use the included pinned sources and do not need model weights, acceleration hardware, or provider credentials.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B regression/check.py --source releases/mlx-0313-server.py --mode demonstrate
python3 -B regression/check.py --source releases/mlx-0322-server.py --mode demonstrate
python3 -B releases/audit.py regression/check.py > /tmp/tokenreceipt-release-audit.json
diff -u releases/results.json /tmp/tokenreceipt-release-audit.json
python3 -B patch/verify.py releases/mlx-0322-server.py
python3 -B regression/ambiguity.py
python3 -B verify.py
python3 -B verify_extended.py
python3 -B verify_2026_11_23.py
python3 -B verify_2026_11_24.py
```

On Linux you can use `sha256sum -c SHA256SUMS`. The two `--mode demonstrate` commands print a failed invariant for the known substitution and pass the control. Demonstration mode exits successfully only when it reproduces both. Run the checker without `--mode demonstrate` to require the invariant on a trusted upstream checkout.

| Replay | Expected result and evidence |
| --- | --- |
| Twelve released MLX-LM wheels | On the isolated candidate-absent fixture, the first four releases retain the sampled pair and the eight releases beginning with 0.30.6 replace it. The candidate-omitted and first-candidate controls pass for every release. [Wheel and source hashes](releases/releases.json) |
| Two score assignments | The same four complete serializer outputs are compatible with distinct underlying winners. The script uses the original vocabulary size and only the nine IDs that the construction needs. [Expected witness](evidence/ambiguity.json) |
| Release patch | Git applies the patch to exactly MLX-LM 0.32.2 and verifies the patched module hash. Fifteen targeted tests verify the sampled fields, error handling on missing `token_strings`/length mismatches/empty candidate tuples, four streaming cases (exact sampled rows, multibyte chunk buffering, stop-token flush, no duplicate rows on length finish), and three controls that compare the complete serialized response. The 11-check revision evaluated in the preprint is kept in [`patch/preprint-11-check/`](patch/preprint-11-check/). Separately, numerical records from 10 previous HTTP requests recompute 23 wrong selected IDs before and zero after across 85 positions with candidates. [Patch and expected counts](patch/) |
| IFEval prompts | With five alternatives, the 64 recorded responses contain 528 wrong selected IDs and 521 wrong selected scores in 1,488 positions. The sampled ID is absent in 204 candidate lists. The paired repair restores all 1,488 selected ID–score pairs. [Per-prompt counts](evidence/ifeval-expected.json) |
| ARC-Easy answer selection | With 2,108 held-out questions, ordering by the recorded scores changes the selected text on 846 questions. The sets of highest-scoring texts are disjoint on 784. Reference and reported scores give 470 and 446 correct answers. First parsable answer gives 501. [Tie and accuracy comparisons](evidence/arc-expected.json) |
| Multilingual byte loss | The saved per-response counts reproduce 72 affected pairs out of 84 and 708 affected positions belonging to 359 completed Unicode characters. The seven language strata share twelve source indices. [Per-pair counts](evidence/multilingual-summary.json) |

The IFEval and patch projections replace actual token IDs with equality classes reset at each position. This preserves within-position comparisons and exact scores without preserving decodable token sequences. The ARC projection stores four exact mean scores per source ID, within-question text equality classes, whether an answer parsed, and whether it was correct. These projections allow numerical recomputation. The regression, patch and ambiguity scripts separately execute the pinned serializer on controlled inputs.

## Full-transport patch live audit (November 24)

`python3 -B verify_2026_11_24.py` (also run by `verify_extended.py`) binds the 15-check patch to its audited hashes, confirms that the archived 11-check preprint patch reproduces the patched-server hash of the November 23 audits, and recomputes the live audit in [`live_audit_2026_11_24/`](live_audit_2026_11_24/README.md). On SmolLM2-135M, Qwen2.5-3B-Instruct-4bit and Qwen2.5-7B-Instruct-4bit, stock MLX-LM 0.32.2 failed `I1` at 2–20 of 64 positions per width and omitted every streaming logprob row (`I5`, 64/64). The patch passed `I1`, `I2`, `I3` and `I5` at all 384 patched positions without changing the generated text.

## Replay, CTranslate2, and Tables 3–4

`python3 -B verify_extended.py` checks the new reports and controls from the included files. The entry points below also let you inspect the outputs separately.

| Evidence | What the offline check recomputes |
| --- | --- |
| [Replay and cache checks](replay/README.md) | Four policies on 2,688 paired observations in nine settings, including three repeat settings on 64 new IFEval prompts. The score check rejects 32 wrong-score repeats accepted by display matching in the two prospective stress arms. The two-distribution counterexample changes the winning draw while all four repeats pass. |
| [CTranslate2 4.9.0](ctranslate2/README.md) | The frozen CPU panel of 30 batch and 30 streaming calls, with 360 selected scores per path. Separate controls cover vocabulary-score normalization, the 4/8 nonempty versus 8/8 wrong-width callback alarms, eight matching final vectors, and 11 independent next-decoder input observations in a separate call. |
| [Independent engine and MLX comparisons](comparison/README.md) | Table 3 grades llama.cpp row counts and score availability, mistral.rs byte baselines, and CTranslate2 shape and score controls. Table 4 separates MLX response alarms, wrong-ID and wrong-score localization, exact recovery, and abstention. With true IDs and the genuine top-five cutoff, all 521 wrong scores can be located and 204 of 1,488 selected scores remain unrecoverable by lookup. |
| [Downstream Qwen2.5-3B, DeepEval, & CT2 distinct-batch](downstream_3b/README.md) | Recomputes the November 23 additions offline via `verify_2026_11_23.py` (also invoked by `verify_extended.py`): `Qwen2.5-3B-Instruct-4bit` ARC-Challenge confidence routing (`N=100` pilot + `N=250` confirmation + Inspect AI `0.3.277` replay + `7B->3B` memory-gate trace), `deepeval==4.3.4` `GEval.measure` (`N=36` + `N=60`, `6/94` pass flips), CTranslate2 `4.9.0` equal- and unequal-length distinct-prompt batch cross-row leakage (`24/24` equal-len, `[1,1,2,2]` unequal-len) plus `tokenreceipt_audit` matrix checks, and SmolLM2-135M / Qwen2.5-3B live hardened-patch I1–I5 audits. |

The replay and comparison projections carry scores, equality classes and measured render or byte outcomes. The CTranslate2 projection carries scores, ranks, normalization terms and comparisons from six study-authored prompts. Its optional live recipe can regenerate full vectors and repeat the decoder-input observation. The offline checks regrade the supplied observations. Rerendering original benchmark responses or recalculating the original full-vocabulary arrays requires the corresponding retained records or a new capture.

## SST-2 learning endpoint

With NumPy installed, run `python3 -B verify_learning.py`. The saved archive contains numeric labels, seed IDs and correct-action log probabilities at the initial and final policies. It contains no review text, token sequences, example IDs, or model parameters. With 128 training seeds and 384 held-out items, the expected accuracy changes by −1.8974 percentage points, with a seed-only 95% bootstrap interval of [−2.6566, −1.1583] and a crossed seed/item interval of [−2.8136, −1.0159]. The script recomputes the endpoint, seed means, and both intervals. This stage starts from recorded policy outputs. [Endpoint values and intervals](evidence/sst2-expected.json)

## Rebuild upstream inputs locally

Give each command a separate output directory outside the repository. It downloads from the pinned sources as needed and checks the recorded SHA-256 before reading the data. The ARC-Easy and SST-2 commands require PyArrow. The remaining commands use the standard library.

```sh
python3 -B rebuild_inputs.py ifeval /tmp/tokenreceipt-ifeval
python3 -B rebuild_inputs.py arc /tmp/tokenreceipt-arc
python3 -B rebuild_inputs.py flores /tmp/tokenreceipt-flores
python3 -B rebuild_inputs.py sst2 /tmp/tokenreceipt-sst2
python3 -B rebuild_inputs.py indra /tmp/tokenreceipt-indra
```

The IFEval and ARC-Easy commands recover their evaluation panels. The FLORES-200 command recovers the 84 language–index pairs. The SST-2 command reconstructs the deduplicated candidate pools in the frozen order. Final training and validation selection also uses the pinned SmolLM2 tokenizer, within-class quotas, and normalized 56-token prompt exclusions. The INDRA command checks the published row file and recomputes AUROC from it in place. The source material and reconstructed text remain in the specified output directory.

## Materials and reproduction limits

[Inventory](INVENTORY.md) maps the distributed files to their verification commands. [Materials and licenses](MATERIALS.md) identifies the third-party sources. [Source receipt](evidence/source-receipt.json) records hashes of selected inputs to the public projections and of the 2,203-entry companion research manifest. If you hold a lawful local copy of that deposit, `python3 -B export_evidence.py /path/to/deposit /path/to/new-projections` regenerates the IFEval, patch, ARC and SST-2 projections using NumPy.

The source receipt contains hashes of the original capture data. Inspect AI and the decoder-training experiment are represented by summaries. The vLLM and SGLang component findings are reported in the manuscript. Live repetition instructions here cover the 64-prompt replay and CTranslate2. Training runs and the complete original benchmark captures remain in the research archive.
