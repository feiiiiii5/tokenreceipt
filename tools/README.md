# Serving-Boundary Audit and Wire-Recovery Tools

Two standalone, zero-third-party-dependency Python 3.11+ tools live in `tools/`:

## 1. Cross-Engine Serving-Boundary Contract Auditor (`tokenreceipt_audit.py`)

Checks **Invariants I1–I5** (`selected-ID identity`, `score/stage consistency`, `per-item/per-step extent`, `UTF-8 byte fidelity`, `stream vs non-stream consistency`) with strict three-valued outcomes (`pass`, `fail`, `abstain`):

```sh
# Audit any live OpenAI-compatible HTTP endpoint
python3 -B tools/tokenreceipt_audit.py probe-http --base-url http://127.0.0.1:8080 --top-k 5

# Run the unified I1-I5 suite over frozen cross-engine projections in reproducibility/
python3 -B tools/tokenreceipt_audit.py replay-matrix --repro-dir reproducibility

# Run unit & offline-matrix tests
python3 -B tools/test_tokenreceipt_audit.py
```

## 2. Generalized Wire-Recovery Client (`wire_recovery.py`)

Tests whether saved non-streaming responses determine the draw with the highest mean sampled-token log probability using arbitrary HuggingFace `tokenizer.json` files (ByteLevel or Metaspace), inferred or explicit EOS IDs, and uniform candidate width $k \ge 1$. Returns the zero-based draw index or `null` (`None`) on ambiguity or excluded fields (`reasoning`, `reasoning_content`, `tool_calls`, usage mismatch).

```sh
python3 -B tools/wire_recovery.py --tokenizer path/to/tokenizer.json draw-0.json draw-1.json draw-2.json draw-3.json
python3 -B tools/test_wire_recovery.py
```
