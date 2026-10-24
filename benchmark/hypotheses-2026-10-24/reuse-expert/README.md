# Reuse original expert buffer experiment

Patch: [reuse-expert.patch](reuse-expert.patch), applied to an isolated copy of `src/models/drama3.cc` from CT2 commit `613c119c`. Other objects are linked unchanged from the existing build. The main CT2 source and installed benchmark environment are untouched.

For INT8 output only, dequantize the source FP8 expert into the existing reusable FP32 scratch, then quantize from that scratch directly into the original INT8 byte container. Keep the original variable registered, add the INT8 scale, and remove the obsolete FP8 inverse scale. Float and INT16 conversion paths are unchanged.

Aliasing is safe for this experiment: the source FP8 byte tensor is no longer read after complete dequantization into separate FP32 scratch; output has the same dtype and shape, so `StorageView::resize_as` preserves its allocation; checkpoint variable aliases are registered after weight preparation; mapped CPU expert loading rejects INT8 requests. The `const_cast` removes the read-only accessor view of an actually mutable model-owned `StorageView` during model preparation, without changing public API or object layout.

Validation: all six existing `Drama3MixedPrecisionTest.*` tests passed, including compute-type conversion numerical checks and GPU projection checks. No full-model performance measurement was started by this agent.

Run the parent measurement with `/tmp/ct2-reuse-expert/venv/bin/python` and `PYTHONPATH=/home/jupyter/faster_tts`. Its local CT2 package loads its copied, replaced native library; remaining dependencies come from the main venv through a `.pth` file. `/proc/self/maps` confirmed the isolated native library path. Bundled OpenBLAS, libgomp, gfortran, and quadmath libraries are copied unchanged from the baseline wheel; the patched library selects the same bundled filenames and uses `$ORIGIN` RPATH. RUY is statically linked from the unchanged original objects. [dependencies.txt](dependencies.txt) records the resolved native dependencies. Build artifacts are under `/tmp/ct2-reuse-expert/`; [build.json](build.json) records provenance.
