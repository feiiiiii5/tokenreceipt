# Redacted Public Projections & Audit Records (2026-11-24)

This directory contains redacted, text-free or study-authored numerical projections verifying all November 24, 2026 cross-engine and `Qwen2.5-7B-Instruct-4bit` experiments without redistributing third-party benchmark prose, raw proprietary transcripts, or local environment paths.

## Contents

- `arc_routing_7b_pilot.jsonl`, `arc_routing_7b_confirmation.jsonl`, `arc_routing_7b_expected.json`: Text-free per-item projections (`N=100` pilot ranks `0..99`, `N=250` disjoint confirmation ranks `100..349`) for `Qwen2.5-7B-Instruct-4bit` (`ARC-Challenge`), capturing wrong ID/score counts, calibration paradox (`ECE` drop vs wrong-answer false-accept surges), and `Inspect AI 0.3.277` full-corpus replay parity.
- `geval_7b_panel_a.jsonl`, `geval_7b_panel_b.jsonl`, `geval_7b_expected.json`: Text-free per-case projections (`N=36` + `N=60`) for `deepeval==4.3.4` `GEval.measure` under `Qwen2.5-7B-Instruct-4bit`, documenting scale attenuation (`2/95` score shifts, `0/95` pass flips vs `6/94` on `3B`).
- `grpo_7b_step0.json`: Text-free per-rollout and corrupted-position projection for `50 x 4 = 200` `Qwen2.5-7B-Instruct-4bit` GRPO trajectories (`8,328` positions across disjoint ARC-Challenge ranks `350..399`), enabling exact pure-stdlib recomputation of group advantages, Step-0 PPO clipping censoring ($\hat{A}<0$) and attenuation ($\hat{A}>0$) across $\epsilon \in \{0.1, 0.2, 0.28\}$, and spurious Schulman $k_3$ KL divergence.
- `sglang_vllm_streaming.json`: Contract replay slots (`vllm/stream`, `sglang/plain-stream`, `sglang/parser-stream`, `sglang/patched-stream`) over 12 synthetic multilingual/ASCII trajectories $\times$ 3 candidate widths (`246` positions/arm), isolating SGLang v0.5.23's empty-delta streaming drop (`78/246`) and vLLM v0.34.0's deferred UTF-8 fragment convention (`I4`).
- `ollama_live_audit.json`: Live `Ollama v0.40.17` (`smollm2:135m` GGUF) HTTP/SSE audit slots across OpenAI-compatible (`/v1/chat/completions`, `24` study-authored English + multilingual prompts across `T in {0.8, 0.0}` and `k in {1, 5}`) and native (`/api/chat`, `8` prompts) endpoints, isolating `I3` stop/UTF-8 coalescing row shortfalls and `k=5` stochastic numeric drift (`T=0.8`).

## Verification

Run from `reproducibility/`:

```sh
python3 -B verify_2026_11_24.py
python3 -B verify_extended.py
```
