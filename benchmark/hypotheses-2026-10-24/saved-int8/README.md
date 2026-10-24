# Saved INT8 checkpoint experiment

`export_int8.cc` loads the current FP8 checkpoint with native CT2 CPU INT8 conversion, then serializes the resulting native tensors into a standard INT8 checkpoint. It uses the native quantization operations exactly; no NumPy approximation or alternate quantizer is involved. Tensor payloads are serialized through const references, avoiding a second full model copy. The exporter does not change repository source or the installed wheel.

The exporter keeps original alias relationships and generated quantization scales. Other checkpoint assets are linked to the source snapshot. The output `config.json` uses `quantization_policy=standard`, `matrix_dtype=int8`, and removes `fp8_format`. Full binary tensor payloads are written once to a new directory; it refuses an existing output directory.

Build:

```bash
c++ -std=c++17 -O2 -I/home/jupyter/CTranslate2/include \
  benchmark/hypotheses-2026-10-24/saved-int8/export_int8.cc \
  -L/home/jupyter/CTranslate2/build-cuda \
  -Wl,-rpath,/home/jupyter/CTranslate2/build-cuda -lctranslate2 \
  -o /tmp/faster-tts-export-int8
```

Small native export/load/export fixture passed, including FP8 input, regular FP32 quantization, codec tensors, generated scales, and a weight alias:

```bash
.venv/bin/python benchmark/hypotheses-2026-10-24/saved-int8/check_export.py
```

One-time export (do not run concurrently with benchmark measurements):

```bash
/tmp/faster-tts-export-int8 \
  /tmp/faster-tts-hf-cache/hub/models--jordimas--drama-3-ct2/snapshots/9aba67191f7f44b915c63ae3bfa08dde5eafdd84 \
  /tmp/faster-tts-saved-int8
```

It reports one-time conversion wall time, peak process RSS, payload size, tensor count, and alias count. Benchmark the saved directory separately with the parent's measurement script; report recurring load+synthesis cost separately from one-time conversion cost.

## One-time conversion result

Export completed successfully, with no concurrent benchmark measurement:

- Wall time: 75.8727 s.
- Peak process RSS: 74.0815 GiB.
- Tensor payload: 36,773,589,540 bytes (about 34.25 GiB).
- 63,004 tensors, no aliases in this source checkpoint.
- Output checkpoint: `/tmp/faster-tts-saved-int8`.

The conversion service is inactive with `Result=success`. Its standard INT8 checkpoint is ready for recurring load/synthesis measurement. One-time conversion is reported separately and should be added when comparing the first-use total cost.
