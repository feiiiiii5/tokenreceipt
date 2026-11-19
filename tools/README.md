# Recover an archived ranking

`wire_recovery.py` tests whether four saved responses determine which draw had the highest mean sampled-token log probability. It returns the winner's zero-based draw index, using the earliest draw to break ties. `null` means the tool abstained.

## Input contract

- Use the stock MLX-LM 0.32.2 non-streaming `POST /v1/chat/completions` endpoint with the pinned SmolLM2-135M-Instruct tokenizer. The command verifies the tokenizer file by SHA-256. [Pinned tokenizer](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac/tokenizer.json)
- Supply four raw response JSON objects for the same prompt and settings in their original draw order. Use plain text with no tools, custom stop sequences, hidden reasoning, or custom rendering. The response must account for every emitted token, including EOS on a `stop` finish.
- Use `logprobs: true` and `top_logprobs: 5`. Each row must contain the five highest raw pre-sampling scores for that position, with unique IDs. The bound uses the smallest listed score as the maximum possible score of any unlisted ID. The response alone cannot establish that the alternatives came from the intended generation or that the server used the expected request settings.

## Run

Use Python 3.11 or newer with its standard library. Supply the tokenizer separately.

```sh
python3 -B tools/wire_recovery.py --tokenizer path/to/tokenizer.json \
    draw-0.json draw-1.json draw-2.json draw-3.json
```

The tool abstains on responses carrying `reasoning`, `reasoning_content`, or `tool_calls`, as well as responses whose probability rows disagree with completion usage. Visible replacement characters leave the affected response's score interval unconstrained.

## Checks

```sh
python3 -B tools/test_wire_recovery.py
```

The unit cases exercise a certifiable group and verify abstention for each excluded field and a mismatched token count. The public entry point agreed exactly with the original implementation on all 9,232 response intervals and all 2,308 decisions in the frozen panels.

| Panel | Decisions checked | Winners certified | Incorrect certifications |
| --- | ---: | ---: | ---: |
| ARC confirmation | 2,108 | 0 | 0 |
| IFEval development | 120 | 2 | 0 |
| IFEval held out | 80 | 1 | 0 |
