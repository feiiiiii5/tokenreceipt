# Rerun the 64-prompt replay experiment

Use Apple Silicon macOS with Python 3.11 and Metal. The scripts fetch the pinned IFEval corpus from its upstream source, verify its SHA-256 and recover the 64 prompts in the order in [keys.json](keys.json). Each prompt receives four original draws. The three candidate-free replay passes use separate fresh services with the same seeds. The recorded run used Python 3.11. They run in forward order with the default prompt cache, reverse order with the default cache, and forward order with the prompt cache disabled.

From this directory, create an environment and download the pinned SmolLM2-135M-Instruct checkpoint. It requires about 300 MB of local storage. [Checkpoint and file hashes](model.json)

```sh
python3.11 -m venv /tmp/replay-env
. /tmp/replay-env/bin/activate
python -m pip install -r requirements.txt
hf download HuggingFaceTB/SmolLM2-135M-Instruct --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --local-dir /tmp/replay-model
```

If you already downloaded the model at that revision, give the collector its directory. If you retained the pinned IFEval JSONL from the public input reconstruction, pass `--ifeval-file PATH` to read it locally. Use `--check-panel` to check the corpus and keys without loading MLX.

```sh
python -B capture.py --model /tmp/replay-model --output /tmp/replay-captures
python -B project_capture.py /tmp/replay-captures /tmp/replay-new-observations.jsonl.gz
python -B ../analyze.py /tmp/replay-new-observations.jsonl.gz
```

The collector verifies the installed release and model files before serving. It uses only loopback HTTP. In the original pass, a same-event observer records the sampled ID and score and calls the original formatter again on the same emission with the alternatives removed. The three repeat passes capture ordinary stock responses. The local raw journals contain prompt and response text. The projection retains scores, equality flags and the fields required to recompute the replay decisions. Exact counts in a new run may vary with device and numerical runtime.

## Conservative ranking from archived responses

Use [wire_recovery.py](wire_recovery.py) to try to identify the highest-scoring draw without a repeat request or native trace. It takes four raw MLX-LM 0.32.2 chat-completions response objects in draw order and returns a zero-based index or `null` for abstention.

```sh
python -B wire_recovery.py --tokenizer /tmp/replay-model/tokenizer.json draw-0.json draw-1.json draw-2.json draw-3.json
python -B test_wire_recovery.py
```

Use non-streaming plain-text responses from the pinned tokenizer and stock renderer with no custom stops, tools or hidden reasoning. The original requests must use `logprobs: true` and `top_logprobs: 5`. Every emitted position must have a probability row, including the EOS on a stop finish. The five distinct candidate IDs at each position must come from the five highest raw scores at the original generation. The archived response cannot establish that provenance. The method uses the smallest listed score as the upper bound for an unlisted ID and keeps all compatible token segmentations when computing a winner.
