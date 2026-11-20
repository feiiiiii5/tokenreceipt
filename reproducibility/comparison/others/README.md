# Count, byte and score checks on frozen llama.cpp and mistral.rs responses

Run `python3 -B compare.py` next to `llama.json` and `mistral.json` with Python 3.10 or later. The command reproduces `results.json` without inference. These are retrospective comparisons on previously collected responses. Candidate widths and transport arms share prompts and reference evidence. The tables use each original panel's units.

## llama.cpp native non-streaming responses

Each positive candidate width has 12 responses and 280 generated positions. Three rows are missing in the same response at both widths. Width zero intentionally requested no metadata and retains 280 native sampled IDs. We use it to document availability and identify the identical sequence used in the C API replay.

| Check at each positive width | Affected response alarms | False alarms on the 11 controls | Localization and access |
|---|---:|---:|---|
| Row count against native sampled-ID count | 1/1 | 0/11 | Three missing rows, with their positions unresolved. Uses the native response alone. |
| Concatenated probability-row bytes against visible text | 0/1 | 0/11 | The surviving row bytes preserve the complete text. Uses the native response alone. |
| Literal vocabulary pieces of native IDs against visible text | 0/1 | 2/11 | The two positive alarms are rendered terminal EOS controls. Needs the checkpoint tokenizer. |
| EOS-aware rendering of native IDs against visible text | 0/1 | 0/11 | The native ID sequence already carries the whole generated text. |
| Each outer row's bytes against its literal vocabulary piece | 1/1 | 2/11 | Three combined byte spans and two empty terminal EOS rows disagree with literal vocabulary bytes. |
| Same byte check with the endpoint's terminal EOS convention | 1/1 | 0/11 | Three combined-byte rows at positions 4, 8 and 20 after alignment, with 274 other returned rows accepted. |

The response-level count gives the deficit. Aligning the outer IDs as an ordered subsequence of the native IDs identifies the three exact missing positions (3, 7 and 19, zero based). A separate tokenizer byte-buffer calculation confirms the boundaries, including the following rows' combined bytes. All 277 returned candidate lists have the requested width. Candidate byte comparisons with EOS rendered empty accept all 277 items at width one and all 1,385 at width five. Literal vocabulary bytes disagree on 3 and 10 candidate items respectively, all for EOS.

The C API replay shares the serving model kernels and conditions on the recorded native IDs. It checks the raw full-vocabulary log-softmax at the selected ID, using 0.002 nats. The 277 scored rows at each width all match the native IDs and the computed scores (maximum error 0.000269 nats). For width five, the wire's outer ID/score pair gives 277 correct values and 3 abstentions on the 280 positions. First-candidate substitution gives 188 correct, 89 wrong, 3 abstained. Looking up the native ID in the candidate list gives 259 correct, 0 wrong and 21 abstained. The latter comprises 18 available lists without the sampled ID and three omitted lists. At width one, native-ID candidate lookup gives 188 correct, 0 wrong and 92 abstentions. The outer and first-candidate counts are the same at both widths. Every observed first-candidate ID error also changes the selected scalar beyond tolerance.

Native IDs and their canonical byte pieces are available for the three gaps. The replay supplies three recomputed raw selected scores. The original wire has no scalar to verify those three as historical values. Recovery of the full missing historical rows is unavailable.

## mistral.rs multilingual transport panel

The paired comparison uses 84 prompts with 1,336 aligned content positions. Each transport has 1,338 generated positions. The two extra positions are terminal EOS rows omitted by streaming. The earlier frozen conservative rule identifies 708 positions on 72 non-English responses where byte arrays differ, the non-streaming bytes are exactly U+FFFD, the streamed piece belongs to a completed Unicode scalar spanning rows, and the displayed label and score agree. The 12 English requests give 184 content-position controls with zero paired byte differences.

| Check on the 84 pairs | Responses flagged out of 72 affected | Responses flagged out of 12 controls |
|---|---:|---:|
| Non-streaming count against recorded usage | 0/72 | 0/12 |
| Streaming count against recorded usage, literal | 0/72 | 2/12 |
| Streaming count allowing the frozen terminal-EOS omission | 0/72 | 0/12 |
| U+FFFD in visible non-streaming text | 22/72 | 0/12 |
| Concatenated non-streaming bytes against visible text | 72/72 | 0/12 |
| At least one non-streaming row containing U+FFFD bytes | 72/72 | 0/12 |

The 22 visible U+FFFD alarms also appear in streaming and correspond to incomplete final UTF-8 at the length cap. Streaming byte concatenation yields 62 exact renders and 22 renders that match with replacement decoding of the incomplete tail. The count alarms on the two English streaming controls refer to actual missing terminal records and are false alarms for the table's interior byte-loss target. The extra terminal positions remain unavailable for transport comparison.

At the position level, checking that bytes equal `ef bf bd` flags 727 positions, including 19 outside the frozen conservative rule. Checking that bytes contain `ef bf bd` flags the same 768 positions as the transport byte-difference check. Sixty paired differences fall outside the frozen rule. The equality of marker locations and transport differences measures agreement between these checks on this panel, with the two serializers sharing one backend. The marker can also occur as a correctly encoded literal replacement character, as observed in the earlier forced control outside this panel.

The separately frozen reference subset has 14 responses per transport. A unique compatible PyTorch path exists for 11, or 176 of 224 positions per transport. All three no-solution abstentions ended in truncated UTF-8. The reference reconstructed the path from visible text, finish reason and token count, using checkpoint logits and canonical bytes. Scoring uses the temperature-scaled full-vocabulary log-softmax before top-k truncation and a 0.02-nat tolerance. The response provides token label strings. ID equality for the outer record is unavailable as a separate HTTP value.

| Check against the independent witness on 11 determinate responses | Correct alarms | False alarms | Misses |
|---|---:|---:|---:|
| Non-streaming concatenation, responses | 9/9 | 0/2 | 0 |
| Visible U+FFFD, responses | 0/9 | 0/2 | 9 |
| Non-streaming byte field equals `ef bf bd`, positions | 80/85 | 0/91 | 5 |
| Non-streaming byte field contains `ef bf bd`, positions | 85/85 | 0/91 | 0 |

The five missed by exact equality include an initial space before the replacement bytes. The independent forward comparison confirms 85 non-streaming byte errors and zero streamed byte errors. The displayed label and selected scalar agree with the reference on all 176 witnessed positions in both transports (maximum score error 0.007618 nats). The cheap containment check identifies the slots in this subset. The independently reconstructed path establishes the canonical byte values and rules out an observed label or score mismatch in those slots. The aligned streaming bytes substitute the independently verified canonical value at all 85 defective positions, with zero wrong substitutions or abstentions among known defects. The other 683 of 768 paired differences have no independent label. Whole historical row recovery is unavailable.

## Prior tests and access requirements

- MURPHY reports count checks against native IDs or chat usage, native streaming identity against the same build's stable non-streaming trace, selected and candidate bytes against a GGUF vocabulary decoded with `special=false`, and numeric comparison across builds on matching aligned histories. The article leaves absolute numerical accuracy and independent chat token identity open. We applied the count and vocabulary-byte ideas to the saved native llama.cpp panel. [MURPHY article, Checks and Limitations](https://huggingface.co/blog/bytkim/llama-cpp-logprob-loss#checks) · [Published measurement code](https://github.com/bytkim/MURPHY-BENCH/blob/3864cb24e3b9ac2620619901a17efd6c429f2e02/murphy/core/measure.py) · [Published vocabulary reader](https://github.com/bytkim/MURPHY-BENCH/blob/3864cb24e3b9ac2620619901a17efd6c429f2e02/scripts/vocabulary_readback.cpp)
- Liu et al., TOSEM §10.2, proposes deterministic repeated outputs, length and repetition checks, visible Unicode validity and U+FFFD, semantic evaluation and cross-backend metamorphic testing. These are proposed checks grounded in the observed bug taxonomy. The precise selected-token ID and conditional numeric score require reference observations beyond those proposals. Our visible U+FFFD baseline uses only response text. The byte-row replacement check instead reads the HTTP metadata field. [Paper, §10.2, article pages 28–29](https://chenzhenpeng18.github.io/papers/TOSEM26-LLMInference.pdf#page=28) · [ACM record](https://doi.org/10.1145/3788873)

## Evidence projection

`llama.json` retains request keys, row counts, ordered positions, candidate scores, native/outer/candidate ID equality and replay reference scores. IDs are position-local equivalence classes. It also carries tokenizer-derived byte equality, text-render equality and the positions confirmed by the frozen byte-buffer calculation. `mistral.json` retains stratum, corpus row index, rank, count, finish, score, byte/label equality, marker indicators, concatenation outcomes, and the offset spans of complete UTF-8 scalars. Byte and label values use separate equality classes for each position. The projection retains no prompt content, response text, literal byte arrays, original token IDs or tokens. The independent score and equality labels are populated only where the frozen model-based reference has a unique path.

For a local archival copy, `python3 -B project.py /path/to/frozen-workspace --output /path/to/projections` regenerates both projections from the original raw records. Source paths within the workspace and content digests are listed under `source_sha256` in each projection. These include `experiments/independent-engine/inference-001/requests.jsonl`, `experiments/independent-engine/inference-001/witness.tsv`, `experiments/data/tokenizer.json`, and `experiments/generality/multilingual-bytes/run-001/{requests,witness}.jsonl`. The raw archive is required to rerender text, decode literal bytes and verify that derivation against the original wire. The distributed projections suffice to reproduce the comparative tallies and verify the values and abstentions reported here.
