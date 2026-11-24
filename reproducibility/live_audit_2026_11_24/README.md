# Full-transport patch live audit (November 24)

`summary.json` is a numerical projection of a live `I1`–`I5` audit of stock `mlx-lm==0.32.2` and the 15-check patch in [`../patch/`](../patch/). It contains no generated text and no filesystem paths. `python3 -B ../verify_2026_11_24.py` recomputes the totals below and checks that the recorded patch and patched-server hashes match [`../patch/release.json`](../patch/release.json).

Protocol: three checkpoints (`HuggingFaceTB/SmolLM2-135M-Instruct`, `mlx-community/Qwen2.5-3B-Instruct-4bit`, `mlx-community/Qwen2.5-7B-Instruct-4bit`), four prompts (`en-sky`, `en-code`, `vi-greet`, `zh-poem`), `max_tokens=16`, `temperature=0.8`, prompt cache disabled. Each width (`top_logprobs` 5 and 1) is compared against same-event native draws and a paired `stream=True` request, giving 64 positions per model, arm and width. A memory-pressure guard ran throughout; no RED events occurred (the 7B runs reached YELLOW, peak process-family RSS 4,071 MiB).

| Model | Stock `I1` fail (w5 / w1) | Stock `I2` fail (w5 / w1) | Stock `I5` fail | Patched `I1`, `I2`, `I3`, `I5` |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 20 / 17 | 18 / 15 | 64 / 64 | 64/64 pass at both widths |
| Qwen2.5-3B-Instruct-4bit | 7 / 10 | 7 / 10 | 64 / 64 | 64/64 pass at both widths |
| Qwen2.5-7B-Instruct-4bit | 2 / 3 | 2 / 3 | 64 / 64 | 64/64 pass at both widths |

Stock streaming responses carried no logprob rows (`I5`). Across 384 patched positions, every invariant passed, and the generated text matched between `top_logprobs` 0, the requested width, and streaming.

Stock failure counts depend on sampling, so they differ from the separate November 23 run in [`../downstream_3b/live_audit_summary.json`](../downstream_3b/live_audit_summary.json). That earlier run used the 11-check patch evaluated in the preprint, which corrected `I1`/`I2` but left streaming `I5` unchanged.
