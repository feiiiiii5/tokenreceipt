# Prospective CTranslate2 test

[The frozen protocol](PROTOCOL.md) and [its timestamp and digests](freeze.json) precede inspection of the target implementation and data collection. The panel used the public CTranslate2 4.9.0 native CPU interface, beam size one and the same pinned float32 model as the protocol. Each of the batch and streaming paths returned 360 steps across 30 calls. Every selected scalar passed the prespecified tolerance. The stream forwards its callback; the two presentations of one stream share a source. [Implementation and upstream history](SOURCE.md)

The additional normalization and batch-coverage diagnostics were prompted by the source. The batch option and callback documentation describe log probabilities. The measured vocabulary field holds logits, and using its entries directly changed five of twelve two-answer rankings. In each joint two-item call, the first callback got the whole batch matrix and the second an empty field at each step. The final results provided a complete vector for each item. The matched split controls gave each callback one vector. [Normalization plan](DIAGNOSTIC-PLAN.md) · [Batch-coverage plan](BATCH-CONTROL.md) · [Numeric results](audit.json) · [Native field readings](confirmation.json) · [Batch comparison](batch-control.json) · [Full-row confirmation](batch-alignment.json)

## Offline verification

From this directory, run the following with Python 3. The projection contains per-position scores, ranks, normalization terms and equality hashes for comparative text, pieces and bytes. These suffice to regrade the observations and the injected substitution control. Recomputing normalization terms from all vocabulary entries requires the full live capture below.

```sh
python3 -B verify.py --projection evidence.json | cmp - audit.json
python3 -B check_controls.py
```

The controls command checks the fresh normalization reading, callback coverage, eight full-row comparisons, final-score comparisons and the 11 next-decoder ID observations against the original callback and sequence. The [decoder witness](witness/README.md) describes an optional repeat of that same-call observation with the exact wheel on arm64 macOS.

## Live repetition

The [release file](https://pypi.org/project/ctranslate2/4.9.0/#ctranslate2-4.9.0-cp311-cp311-macosx_11_0_arm64.whl), the [eight original model files](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac) and the [environment and file digests](provenance.json) identify the measured runtime. Use CPython 3.11 and the package versions listed there. Set CT2_WHEEL to the downloaded wheel file, CT2_CHECKPOINT to the checkpoint directory, and CT2_WORK to a fresh directory. Run from this directory. The preparation command verifies the wheel and all checkpoint files before conversion. The packaged launchers use an archive-relative model manifest and a neutral shared lock name.

```sh
mkdir -p "$CT2_WORK"
python -m pip install --no-index --no-deps --target "$CT2_WORK/site" "$CT2_WHEEL"
export PYTHONPATH="$CT2_WORK/site"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -B prepare_model.py --checkpoint "$CT2_CHECKPOINT" --manifest model-source.json \
  --wheel "$CT2_WHEEL" --converted "$CT2_WORK/converted" --provenance "$CT2_WORK/provenance.json"
python -B capture.py --model "$CT2_WORK/converted" --tokenizer "$CT2_CHECKPOINT" --output "$CT2_WORK/raw"
python -B verify.py --raw "$CT2_WORK/raw" --write-projection "$CT2_WORK/evidence.json" --output "$CT2_WORK/audit.json"
python -B confirm.py "$CT2_WORK/converted" "$CT2_CHECKPOINT/tokenizer.json" > "$CT2_WORK/confirmation.json"
python -B batch_control.py "$CT2_WORK/converted" "$CT2_CHECKPOINT/tokenizer.json" > "$CT2_WORK/batch-control.json"
python -B batch_alignment.py "$CT2_WORK/converted" "$CT2_CHECKPOINT/tokenizer.json" > "$CT2_WORK/batch-alignment.json"
```

The acquisition journal, final sequences, vocabulary map and all 677 original full-width arrays stay in the caller's output directory. They can be regenerated without network access after obtaining the release and checkpoint. The archived panel yielded 287 distinct forward-pass prefixes. The six study prompts are included in the protocol.
