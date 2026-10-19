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

For GPU runs, the script defaults `CT2_CUDA_ALLOCATOR` to `cub_caching` to avoid
intermittent ROCm asynchronous memory-pool teardown crashes after synthesis.
This is a benchmark setting; the native library's global default is unchanged.
Explicit allocator overrides are honored.
The script defaults `CT2_CUDA_CACHING_ALLOCATOR_CONFIG` to
`2,6,24,209715200`. When the CUB allocator is active, this
rounds allocations to powers of two instead of powers of four, reducing padding
for cached float16/bfloat16 expert weights. Explicit environment settings take
precedence; the setting has no effect on the asynchronous CUDA allocator.
It limits unused cached blocks to 200 MiB, not total GPU memory. Live expert
weights can still exceed available memory. On integrated GPUs, GPU allocations
also consume system RAM and are not included in the reported process RSS.

### Memory configurations

CPU runs load weights into allocated RAM, without a process memory cap.
The default FP8 checkpoint works directly; no separate INT8 model is required.
Compute type, prompt/history size, `max_frames`, and `prefill_chunk_size` affect
individual allocations, without guaranteeing a total RAM ceiling. The benchmark
does not invoke systemd or configure cgroups.

Each GPU runs both `limited` and uncapped (`default`) profiles by default, in
separate processes, with all selected compute types for each profile.
The `limited` profile sets `CT2_DRAMA3_GPU_EXPERT_CACHE_GB=12`. This limits cached
expert weights, not total VRAM or system RAM. Other allocations require extra
memory. Profile settings override an inherited GPU expert-cache setting.

Use `--memory-profile limited` to run only the capped configuration, or
`--memory-profile default` to run only an uncapped GPU expert cache.
`--gpu-expert-cache-gb` overrides the limited cache budget. An explicit
CPU `limited` profile is reported as skipped. Loading and warmup remain outside
the timing; failures are reported and the sweep continues.

### Official Hugging Face / PyTorch comparison

Run `benchmark_hf.py` with a Python environment containing the official Fish
Speech dependencies (including PyTorch). It uses the local official checkpoint
and source tree; override them with the model argument and `--source`, or
`HF_MODEL` and `HF_SOURCE`.

From the faster_tts repository root, comparable commands are:

```bash
python benchmark/benchmark_hf.py --device cuda --compute-type float32 --runs 3
uv run python benchmark/benchmark.py drama-3 --device cuda --compute-type float32 --memory-profile default --runs 3
```

Both use the same default text, eight CPU threads, one full synthesis warmup,
and the same timing/RTF/peak RSS summary. Single runs show only the summary.
HF retains both the language model and CPU float32 codec across runs; loading,
warmup, WAV writes, and optional quality checking are outside the timer. GPU
work is synchronized at timing boundaries. Sentence splitting, bounded chunk
history, seed 0 per chunk, 512 total frames, temperature 0.7, top-k 30, top-p 0.7,
and prefill chunks of 16 match the CT2 API defaults. `--check quality` uses the
same separate CPU Whisper checker and requires faster-whisper in that environment.

Without `--device`, HF runs CPU and every PyTorch-visible GPU; without
`--compute-type`, it runs bfloat16, float16, and float32 in separate processes.
Failures are reported while the remaining configurations continue. These types
select floating-point weights/activations. CPU runs retain the sample runner's
FP8 expert storage and dequantize selected experts. GPU runs follow the official
hardware check: supported NVIDIA GPUs retain FP8; ROCm expands expert weights
to the selected floating-point type, requiring substantially more memory. HF has no corresponding CT2 INT8 modes or expert-cache cap,
so HF GPU runs are uncapped only. Compare with CT2's `--memory-profile default`.
Matching sampling settings do not imply identical random samples or audio
lengths. Peak RSS measures process RAM and excludes GPU allocations in both scripts.

### Measured CPU and ROCm results (2026-10-19)

Hardware: AMD Ryzen AI MAX+ PRO 395 with Radeon 8060S, 16 CPU cores / 32
threads; 122.8 GiB system RAM. The integrated GPU is `gfx1151`; ROCm PyTorch
reports 61.4 GiB GPU-addressable memory, shared with system RAM. Ubuntu
26.04.1 LTS, Linux 7.0.0-38-generic. Both benchmarks use eight CPU threads.

Settings: `Hello! Welcome to our application.`, one complete warmup followed
by three timed runs per configuration. The table shows uncapped configurations
for this shared CPU/GPU memory system. All configurations and both backends
run sequentially. Model/codec loading and warmup are excluded from latency;
peak RSS includes them. Waveform decoding runs on CPU float32 for both
backends. No ASR quality checks were enabled. Lower latency and RTF are better;
RTF below 1 is faster than real time. Audio duration is the mean across runs.

Software: CTranslate2 4.8.3 (local ROCm build, Python 3.14); PyTorch and
torchaudio 2.8.0+rocm6.4 (HIP 6.4.43482, Python 3.13). CT2 wheel SHA-256:
`043a4be5e50271848f978e2cc67fec6d2ccfb2ae19241f91b29e52c6acf532e3`.
CT2 uses `jordimas/drama-3-ct2` revision
`9aba67191f7f44b915c63ae3bfa08dde5eafdd84` (`experts_fp8_sensitive_float32`).
HF uses the local official Drama 3 checkpoint and Fish Speech source revision
`541566932abbaebcb0a40e562a1f846738383f38`.

| Backend | Device | Compute type | Mean (s) | Std (s) | Audio (s) | RTF | Peak RSS (GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| CT2 | CPU | float32 | 132.833 | 4.873 | 2.508 | 52.969 | 45.276 |
| CT2 | CPU | int8 | 27.404 | 0.050 | 2.554 | 10.729 | 44.754 |
| CT2 | CPU | int8_float32 | 27.783 | 0.297 | 2.554 | 10.877 | 44.755 |
| CT2 | ROCm GPU | bfloat16 | 27.443 | 0.490 | 2.508 | 10.943 | 46.613 |
| CT2 | ROCm GPU | float16 | 29.970 | 0.183 | 3.111 | 9.632 | 46.880 |
| CT2 | ROCm GPU | float32 | 47.344 | 1.130 | 2.508 | 18.879 | 46.810 |
| CT2 | ROCm GPU | int8 | 12.735 | 0.049 | 3.158 | 4.033 | 45.592 |
| CT2 | ROCm GPU | int8_bfloat16 | 30.674 | 2.358 | 3.158 | 9.714 | 48.739 |
| CT2 | ROCm GPU | int8_float16 | 42.816 | 2.350 | 5.248 | 8.159 | 49.505 |
| CT2 | ROCm GPU | int8_float32 | 13.102 | 0.338 | 3.158 | 4.149 | 45.591 |
| HF / PyTorch | CPU | bfloat16 | 117.856 | 0.192 | 3.158 | 37.321 | 45.622 |
| HF / PyTorch | CPU | float16 | 188.062 | 0.450 | 3.158 | 59.553 | 45.614 |
| HF / PyTorch | CPU | float32 | 123.234 | 1.773 | 2.508 | 49.141 | 52.484 |

The same sampling settings can produce different audio across backends and
precisions. Compare RTF alongside latency; these measurements do not establish
equivalent audio quality. Peak RSS excludes GPU allocations; on this integrated
GPU, those allocations also consume shared system RAM.

Commands (from this directory, using the environments described above):

```bash
../.venv/bin/python -u benchmark.py --memory-profile default --runs 3
/tmp/faster-tts-hf-rocm/bin/python -u benchmark_hf.py /home/jordi/sc/audio/models/drama-3 --runs 3
```

### Local CTranslate2 wheel

The uv dependency source currently points to
`../../python/dist/amd-lzNjsoc1/cpython-314/ctranslate2-4.8.3-cp314-cp314-linux_x86_64.whl`,
a local Python 3.14 Linux build with ROCm support. The measured-results section
records the installed wheel hash. This is a host-specific wheel, not a manylinux
release. Run `uv sync` before benchmarking to install the referenced wheel.

### JSON results

Both benchmark scripts automatically write JSON in the current working directory:
`benchmark.py` writes `ct2.json`, and `benchmark_hf.py` writes `hf.json`. Each
invocation overwrites the existing file while retaining normal console output.
The full console output (including worker errors) is also saved to `ct2.log` or
`hf.log` in the current working directory, overwriting the previous log.
Use `--output-json PATH` to override the destination:

```bash
../.venv/bin/python benchmark.py --memory-profile default --runs 3
/tmp/faster-tts-hf-rocm/bin/python benchmark_hf.py /home/jordi/sc/audio/models/drama-3 --runs 3
```

Each invocation replaces the destination with a JSON document containing
`schema_version`, backend/version, host information, and a `results` array.
Host information includes the CPU model, logical CPU count, and total system
RAM. Each result also includes `hardware.cpu` and, for GPU runs, `hardware.gpu`
with the selected GPU name, runtime (CUDA/ROCm), and total device memory.
Unavailable hardware fields are null with a query error when applicable.
Device, memory-profile, and compute-type sweeps merge into this array, updating
the file after each configuration. Use separate files for separate invocations.

Each successful result contains model, device/index, `compute_type`, memory
profile/cache budget, thread count, text, warmup/run counts, per-run
`time_seconds`, `audio_seconds`, `rtf`, and `finish_reason`, plus a `summary`:

- `mean_time_seconds` and `std_time_seconds` (sample standard deviation).
- `mean_audio_seconds` and `rtf` (total synthesis time / total audio duration).
- `peak_rss_gib` (process RAM including loading/warmup, excluding GPU allocations).

Failures and skips have `status: "failed"` or `"skipped"` and no summary;
failed workers include an exit code when available. Consult console output for
failure details. Reports describe synthesis performance; optional ASR checking
continues to print its scores separately. JSON numbers retain full measurement
precision. The [saved HF](hf.json) and
[saved CT2](ct2.json) reports were imported from the existing
logs and therefore retain only the precision those logs recorded.
