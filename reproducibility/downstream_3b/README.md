# Numerical Projections for Cross-Architecture Audit, 3B Downstream Routing, DeepEval G-Eval, & CTranslate2 Distinct-Batch

This directory stores strictly redacted numerical projections (no ARC-Challenge question text, no DeepEval case prompts/answers, no raw generated completions, and no filesystem paths) so `../verify_2026_11_23.py` can recompute all downstream endpoints offline:

- `arc_routing_pilot.jsonl` (`100` items) and `arc_routing_confirmation.jsonl` (`250` items) from `Qwen2.5-3B-Instruct-4bit` on disjoint hash-ranked splits of `allenai/ai2_arc` `ARC-Challenge/test`, plus `arc_routing_expected.json` (including Inspect AI `0.3.277` confirmation replay totals and the `Qwen2.5-7B` vs `3B` hardware-gate summaries).
- `geval_panel_a.jsonl` (`36` cases) and `geval_panel_b.jsonl` (`60` cases) from `deepeval==4.3.4` `GEval.measure` on `Qwen2.5-3B-Instruct-4bit`, plus `geval_expected.json`.
- `ct2_distinct_batch.json`: numerical summary and stripped callback-slot records from `ctranslate2==4.9.0` equal-length and unequal-length distinct-prompt batch diagnostics.
- `live_audit_summary.json`: per-position and per-request I1–I5 contract-audit results across `SmolLM2-135M-Instruct` and `Qwen2.5-3B-Instruct-4bit` under both stock `mlx-lm==0.32.2` and the 11-check preprint patch ([`../patch/preprint-11-check/`](../patch/preprint-11-check/)); the streaming-capable 15-check revision is audited in [`../live_audit_2026_11_24/`](../live_audit_2026_11_24/README.md).
