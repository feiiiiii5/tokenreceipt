# Public reproducibility inventory

`SHA256SUMS` lists the digest of each distributed file. Run the commands from this directory with Python 3.11 or later. [Setup and scope](README.md) · [Sources and terms](MATERIALS.md)

| Material | Included evidence and checks |
| --- | --- |
| `regression/`, `releases/`, `patch/` | Pinned MLX-LM code, synthetic fixtures, release audit and exact patch verification. See the [offline commands](README.md#offline-replay). |
| `evidence/`, `provenance/` | IFEval, patch HTTP, ARC-Easy and multilingual counts via `python3 -B verify.py`. SST-2 recorded learning endpoints and paired intervals via `python3 -B verify_learning.py` with NumPy. Other archived experiments are represented by summaries and provenance where included. |
| `rebuild_inputs.py`, `export_evidence.py` | Rebuild chosen inputs from pinned upstream sources or regenerate projections from a separately obtained research deposit. [Input procedures](README.md#rebuild-upstream-inputs-locally) |
| `replay/` | Four filters on nine frozen settings, with numerical inputs, a two-world counterexample, wire-recovery tests and an optional live recipe for the new 64-prompt panel. [Replay documentation](replay/README.md) |
| `ctranslate2/` | Frozen prospective protocol, per-position numeric observations, same-call decoder witness, batch-shape and normalization diagnostics, source pins, and optional live scripts. The six prompt strings and short native witness are study-authored examples. [CTranslate2 documentation](ctranslate2/README.md) |
| `comparison/` | A shared checker and access-specific analysis for MLX-LM, llama.cpp, mistral.rs and CTranslate2 read the included numerical files. Run `python3 -B comparison/verify.py` to compare all four reports to their archived results. [Adapters and grading](comparison/README.md) |
| `downstream_3b/`, `verify_2026_11_23.py` | Pure-stdlib numerical verification of `Qwen2.5-3B-Instruct-4bit` ARC-Challenge routing (`N=100` + `N=250` + Inspect AI replay), `deepeval==4.3.4` G-Eval (`N=36` + `N=60`), CTranslate2 distinct-prompt cross-row leakage, and SmolLM2-135M / Qwen2.5-3B live hardened-patch I1–I5 audits. [Downstream 3B documentation](downstream_3b/README.md) |
| `live_audit_2026_11_24/`, `downstream_2026_11_24/`, `verify_2026_11_24.py`, `patch/preprint-11-check/` | Pure-stdlib verification of the 15-check full-transport patch, the SmolLM2-135M / Qwen2.5-3B / Qwen2.5-7B 384-position live audit, `SGLang v0.5.23` vs `vLLM v0.34.0` streaming (`246` pos/arm), live `Ollama v0.40.17` HTTP/SSE audits, `Qwen2.5-7B-Instruct-4bit` ARC-Challenge routing, Inspect AI replay, DeepEval 4.3.4 Panels A/B, and `50 x 4` GRPO Step-0 projections. [Live audit](live_audit_2026_11_24/README.md) · [Nov 24 projections](downstream_2026_11_24/README.md) |
| `live_gpu_2026_11_25/` | SGLang/vLLM live HTTP/SSE summaries and TriviaQA routing for one live MLX-LM cohort and two controlled score-injection cohorts. [Projections and verification](live_gpu_2026_11_25/README.md) |
| `verify_extended.py` | Runs the replay, CTranslate2, comparison, `verify_2026_11_23.py`, and `verify_2026_11_24.py` offline checks together without model weights or optional Python dependencies. |
| `licenses/` | Upstream MIT and Apache-2.0 notices for the included MLX-LM source, the preprint’s SGLang `serving_chat.py` excerpt, and CTranslate2 API descriptions. |
