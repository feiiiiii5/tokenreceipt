# November 25 numerical projections

`wp3a_triviaqa_routing.json` contains 150 pilot and 400 disjoint confirmation items per cohort from TriviaQA `rc.nocontext` validation. The projection contains ranks, correctness labels, confidence scores and token error counts. The protocol specified exact or normalized alias matching. The implemented grader also accepts whole-phrase containment of eligible aliases. Confirmation ECE increased in all three cohorts, contrary to the protocol's expected decrease. Question text, aliases, generated answers and token sequences stay in the research archive. [Dataset card](https://huggingface.co/datasets/mandarjoshi/trivia_qa/tree/main/rc.nocontext) · [TriviaQA source and copyright notice](https://nlp.cs.washington.edu/triviaqa/index.html)

The MLX-LM 0.32.2 cohort uses the cached `mlx-community/Qwen2.5-7B-Instruct-4bit` checkpoint and compares native sampled scores with the stock HTTP serialization of the same event at `top_logprobs=5`. The two Tesla T4 cohorts use live Transformers generation in NF4 with FP16 compute. They simulate the `k=1` substitution by taking the top-1 score at each sampled position. The Inspect AI record summarizes an offline SDK replay of the MLX captures.

Run from this directory with Python 3.10 or later.

```sh
python3 -B verify_triviaqa.py
```

The script recalculates the split sizes, token counts, 10-bin ECE, Brier scores and wrong-answer false-accept counts at 0.80 and 0.90 from the published rows. False-accept rates use the total number of incorrect answers as the denominator.

`wp1a_sglang_live.json` and `wp1b_vllm_live.json` summarize black-box HTTP/SSE probes against live CUDA servers on a T4. The stored SGLang row-count totals include all 60 requests per arm. The stock totals differ by 47 rows. Among the 58 same-text pairs per arm, stock SGLang omits 49 rows across 21 requests and changes two routing decisions at −0.35 nats. The patched arm records zero omissions and zero changes on its 58 same-text pairs. Recount with `jq` from this directory.

```sh
jq '[.wp3b_plain, .wp3b_patched] | map([.rows[] | select(.same_text)] | {
  pairs: length,
  requests_with_omissions: (map(select(.dropped_rows > 0)) | length),
  omitted_rows: (map(.dropped_rows) | add),
  routing_changes: (map(select((.ns_mean_lp >= -0.35) != (.stream_mean_lp >= -0.35))) | length)
})' wp1a_sglang_live.json
```

On vLLM 0.34.0, the Qwen panel recorded 382/382 positions passing row coverage (I3), one score consistency failure (I2), and 278 I5 passes with 104 abstentions. TinyLlama passed I1–I5 on 146/146 positions. These live black-box summaries do not carry a native tokenizer-byte reference or raw packets for strict per-token replay.
