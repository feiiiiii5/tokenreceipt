# Same-generation decoder-input witness

The witness uses the first frozen prompt in a call separate from the main panel. The original CTranslate2 4.9.0 ARM64 wheel runs one CPU float32 `generate_batch` call at seed 1701, temperature 1, top-k zero, top-p 1 and a 12-position cap. The decoder breakpoint, callback and final result belong to the same process and generation. [Frozen protocol](../PROTOCOL.md)

## Result

The debugger stopped 13 times at the incremental decoder entry. The first call prefills the prompt with a null output pointer and is skipped. The first recorded input is the last prompt token, ID 436. Eleven subsequent decoder inputs agree exactly with both the preceding callback ID and its final sequence ID. Eight are off-first and six have rank above five. For example, the first generated ID was 2268 at rank 12. The next decoder call consumed ID 2268. After the 12-position cap the decoder has no input corresponding to the last sampled ID, 314. That final position has no independent next-input observation. [Recorded decoder steps and input IDs](decoder-inputs.json) · [Callback and final result](run-001/response.json)

## Probe

The debugger reads the one-token INT32 CPU `StorageView` passed to the incremental overload of `TransformerDecoder::operator()`, before the decoder consumes it. The breakpoint is set at the overload's entry, module file address `0xc6d04` in the wheel identified in the source audit. In the arm64 call, `x1` is the absolute decoder step, `x2` points to the ID view, and `x4` is the output pointer. A null `x4` marks the no-output prefill in this run. The wheel's `StorageView::data<int>() const` at file address `0x16a464` loads the underlying pointer at view offset `0x18`. The probe validates the entry address, INT32 tag and scalar input size and reads the ID through that pointer. It changes no source or wheel files and resumes the process immediately after each observation. The pointer read witnesses the ID consumed by the next decoder call. Callback score and rank are captured separately through the normal Python API. [Static source audit](../SOURCE.md) · [Decoder entry and use](https://github.com/OpenNMT/CTranslate2/blob/a22ba1c63ee35d1a0d25db149aa2a66d36a6a732/src/layers/transformer.cc#L636-L667) · [StorageView layout](https://github.com/OpenNMT/CTranslate2/blob/a22ba1c63ee35d1a0d25db149aa2a66d36a6a732/include/ctranslate2/storage_view.h#L258-L266)

To repeat from the CTranslate2 study directory, use CPython 3.11 on arm64 macOS with the exact wheel, Transformers, NumPy and Xcode Command Line Tools. Provide the pinned tokenizer and the main panel's float32 converted model. The runner uses the same shared research lock and records its source digests before launch. The runner writes the full debugger transcript and command to the supplied output directory.

```sh
python -B witness/run.py \
  --site /path/to/extracted/wheel \
  --model /path/to/converted/model \
  --tokenizer /path/to/pinned/tokenizer \
  --output /path/to/local/witness-repeat
```
