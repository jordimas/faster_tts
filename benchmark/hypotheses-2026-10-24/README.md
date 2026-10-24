# CPU INT8 memory hypotheses — 2026-10-24

Preparation was done in parallel; measurements ran sequentially to avoid contention. All runs used 8 threads, the same text, 1 warmup, 1 timed synthesis, no service RAM caps, and no allocator thresholds. Total time is model loading + warmup + timed synthesis, excluding interpreter imports and one-time checkpoint export. Peak RSS spans all three phases.

| Variant | Load (s) | Warmup (s) | Timed synthesis (s) | Total (s) | Total change | Peak RSS (GiB) | Peak reduction | Loaded RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 42.838 | 71.793 | 75.272 | 189.908 | +0.0% | 68.384 | 0.0% | 58.606 |
| saved-int8 | 27.542 | 75.064 | 74.926 | 177.536 | -6.5% | 35.090 | 48.7% | 34.426 |
| reuse-expert | 42.140 | 75.661 | 76.604 | 194.410 | +2.4% | 44.694 | 34.6% | 34.948 |

Both hypotheses preserved audio bytes and generated-code bytes exactly in warmup and timed runs (SHA256 values in JSON reports). All outputs ended with EOS and contained 2.786 seconds of audio. This validates these particular inputs, not general model quality.

## Saved INT8 checkpoint

The original checkpoint has FP8 experts and FP32 sensitive weights. The exporter loads native CT2 INT8 weights once, writes the converted tensors and scales, and records a standard INT8 checkpoint. Subsequent loads skip quantization. No changes to the installed CT2 runtime are required. The resulting checkpoint is `/tmp/faster-tts-saved-int8`; companion assets link to the original cache, so this experimental directory is not a standalone distributable.

One-time export cost: **75.873 s**, **74.082 GiB peak RSS**, **34.248 GiB of new tensor payload**. Counting export, first use took 253.409 s. Its conversion cost pays back after about 5.0 loads based on measured loading time alone. This estimate is sensitive to disk caching and run-to-run variation.

## In-place expert storage

The INT8-only patch writes quantized expert bytes directly into each original FP8 expert allocation after decoding into the separate reusable FP32 scratch buffer. It avoids allocating a replacement byte buffer and freeing the original for every expert. It reduces peak and retained RAM, but does not eliminate conversion work. In this measurement it did not improve total time.

The experimental runtime is isolated at `/tmp/ct2-reuse-expert/venv/bin/python`; neither the main CT2 source nor the installed production wheel was changed. Its bundled OpenBLAS/GOMP libraries match the baseline byte-for-byte; RUY uses unchanged original build objects. All six focused mixed-precision tests passed. The exporter also passed a tiny native save/load round-trip test.

## Limits and evidence

One timed run per variant is insufficient to establish small speed differences. File-cache state was not reset; these are local warm-cache workflow measurements, not cold-disk guarantees. The exporter adds a one-time disk-space and memory cost. The two changes were measured independently, not combined. No global allocator settings, API changes, or automatic benchmark queue were introduced. GPU benchmarks were not started.

Baseline is the installed wheel for CT2 commit `613c119cb5ecdd9fb25cab470e991b521fa93890`, SHA256 `58b521b74f288e01d8e94d2ce96bd412d2e1af38dc2dc39e730d889f8bed18eb`. The source checkpoint revision is `9aba67191f7f44b915c63ae3bfa08dde5eafdd84`.

- [Structured comparison](comparison.json)
- [Measurement script](measure.py)
- [Baseline measurement](baseline.json) and [log](baseline.log)
- [Saved checkpoint measurement](saved-int8.json), [log](saved-int8.log), and [exporter](saved-int8/README.md)
- [In-place measurement](reuse-expert.json), [log](reuse-expert.log), and [patch/build details](reuse-expert/README.md)
