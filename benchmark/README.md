## Performance benchmarking

Run the standalone script from the repository root:

```bash
uv run python benchmark/benchmark.py drama-3 --threads 8 --runs 3
uv run python benchmark/benchmark.py drama-3 --device cuda --compute-type float16
```

Use `--text` for your own input, or pass a local model directory instead of
`drama-3`. The script loads once, warms up once, then reports synthesis time,
audio duration, latency, real-time factor (RTF), device, and peak process RAM
(RSS, in GiB). It performs one timed run by default; use `--runs` for more
runs and their mean latency and sample standard deviation.
Peak RSS includes loading, warmup, and timed runs; it excludes filesystem cache
outside the process. The script uses Python's resource module (Linux/macOS).
Model download progress bars are hidden by default; use `--show-progress` to
enable them. Warnings and errors remain visible.
RTF below 1 means faster than real
time. Loading and warmup are excluded. Keep text and model fixed when comparing
settings; check audio durations because different precisions can change output.
Without `--device`, it benchmarks CPU followed by every GPU visible to CTranslate2
(including ROCm GPUs exposed through `cuda`). Each device runs in a separate
process, and a failed device does not prevent the remaining devices from running.
Use `--device cpu` or `--device cuda --device-index 0` to select a single device.
`--device auto` selects one GPU when available, otherwise CPU.
Without `--compute-type`, it benchmarks every compute type reported as supported
by CTranslate2 for the selected device. Specify a type to benchmark only that one.
Each compute type runs in a separate process so model memory and native allocator
caches are released before the next type starts. Failed types are reported and
the sweep continues, exiting with status 1 if any failed. A single type can still
exceed available memory; Drama 3 GPU inference also keeps CPU float32 weights.

Use `--check quality` (or `--check-quality`) to transcribe every timed output
with faster-whisper's multilingual `medium` model and compare it with `--text`:

```bash
uv run --with faster-whisper python benchmark/benchmark.py drama-3 --check quality
```

The checker writes temporary mono PCM16 WAV files and removes them after checking.
It runs after the performance summary in a separate CPU int8 process, so ASR
loading, transcription, and memory are excluded from synthesis timing and peak
RSS. The first check downloads the Whisper model if it is not cached. A single
final line reports `WER: 12.50% (1/8) [cuda:0 float16 default]`
(total word errors / total reference words), labeled with the device, compute
type, and memory profile for each benchmark configuration. This pools errors and
reference word counts across runs (equivalent to mean WER because every run uses
the same text).
WER is word-level edit distance divided by
reference word count; lower is better, and insertions can make it exceed 100%.
Scoring normalizes Unicode, case, whitespace, and punctuation, retaining internal
apostrophes. Numbers and abbreviations are not expanded, so equivalent spoken
forms can still count as errors. Word-based scoring is less useful for languages
without spaces between words.

WER is an intelligibility/content regression check: it can expose dropped,
repeated, or mispronounced words, but ASR mistakes also affect the score. It does
not measure naturalness, prosody, voice similarity, or audio artifacts. Compare
fixed texts and ASR settings across configurations, use a varied set of longer
texts, and listen to representative outputs alongside the scores.

For GPU runs, the script defaults `CT2_CUDA_CACHING_ALLOCATOR_CONFIG` to
`2,6,24,209715200`. When the CUB allocator is active (the ROCm default), this
rounds allocations to powers of two instead of powers of four, reducing padding
for cached float16/bfloat16 expert weights. Explicit environment settings take
precedence; the setting has no effect on the asynchronous CUDA allocator.
It limits unused cached blocks to 200 MiB, not total GPU memory. Live expert
weights can still exceed available memory. On integrated GPUs, GPU allocations
also consume system RAM and are not included in the reported process RSS.

### Memory configurations

CPU runs use default mapped-weight loading where supported. There is no CPU
process cap or `--cpu-memory-gb` setting. Weight quantization, prompt/history size,
`max_frames`, and `prefill_chunk_size` control individual allocations, without a
guaranteed total RAM ceiling. Optional `CT2_DRAMA_CPU_MLOCK=1` keeps mapped expert
pages resident, subject to the OS locked-memory allowance.

Each GPU benchmarks `default` (uncapped expert cache) and `limited`
(`CT2_DRAMA3_GPU_EXPERT_CACHE_GB=4`) in separate processes. This limits cached
expert weights, not total VRAM or system RAM. Other allocations require extra
memory. Profile settings override an inherited GPU expert-cache setting.

Use `--memory-profile default` or `--memory-profile limited` to select one GPU
profile. `--gpu-expert-cache-gb` overrides the limited cache budget. An explicit
CPU `limited` profile is reported as skipped. Loading and warmup remain outside
the timing; failures are reported and the sweep continues.

### Local CTranslate2 wheel

The uv dependency source currently points to
`dist/ctranslate2-4.8.3+rocm.1e66308c.mkl.1-cp314-cp314-linux_x86_64.whl`,
built from CT2 commit `1e66308cb916dd6e5f4bd074848a6ac17b2adf96` with
ROCm, MKL, and oneDNN. It bundles a freshly built native library and Python
3.14 bindings. This is a host-specific Linux wheel, not a manylinux release.
Build provenance and hashes are recorded in the adjacent `.build.json` file.
Run `uv sync` before benchmarking to install the referenced wheel.
