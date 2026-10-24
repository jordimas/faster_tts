# faster_tts

`faster_tts` is a Python text-to-speech library built on CTranslate2, currently
supporting **Drama 3 on CPU and GPU**. Its goal is to provide simple Python
access to Drama 3 and future
CTranslate2 (CT2) text-to-speech engines.

Key features:

- ✅ Automatic model downloading and caching from Hugging Face, plus local checkpoints
- ✅ Plain-text synthesis with speaker tags and prompt preparation
- ✅ Sentence splitting with history conditioning and joined audio
- ✅ Automatic or explicit GPU expert-cache budgets
- ✅ Command-line synthesis with WAV output

Requires CTranslate2 **4.8.4 or higher**.

## Quick start

```python
from faster_tts import Model

model = Model()
result = model.synthesize("Hello! Welcome to our application.")

audio = result.audio
sample_rate = result.sample_rate
```

`Model()` downloads or reuses the cached
[`jordimas/drama-3-ct2`](https://huggingface.co/jordimas/drama-3-ct2) checkpoint.
Reuse the model across calls. Audio is a NumPy array with the returned sample rate.

`Model(path_or_hub_id)` selects the engine from the converted checkpoint's
`model_type`. `model.info()` reports `engine` and supported `capabilities`.
Only Drama 3 is implemented today; Qwen TTS support is not yet included.
Codec dimensions, sample rate, and available features depend on the engine.
Engine-specific knobs are passed through `engine_options={…}` on both
`Model(...)` and `model.synthesize(...)`; unknown keys raise `ValueError`.

Save a WAV with the CLI:

```bash
python -m faster_tts \
  --text "Hello, how are you?" --output hello.wav
```

Hear three short [Drama 3 samples](sample/README.md): expressive delivery,
two speakers, and multilingual speech.

## Model loading

Model loading accepts a short name, a compatible converted Hub repository ID,
or a local directory:

```python
model = Model("drama-3")  # Alias for the default checkpoint.
model = Model("jordimas/drama-3-ct2", revision="main", cache_dir="./models")
model = Model("/path/to/converted-model")
model = Model(local_files_only=True)  # Use only cached assets.
```

The CLI accepts the same model argument and cache options; Hub IDs may use an
`hf://` prefix. Existing local directories take precedence. Use a `Path` object
or an explicit path (`./`, `../`, `~/`, or absolute) to avoid interpreting missing
local paths as Hub IDs. Custom checkpoints must match the resolved engine's
layout; `cache.py` delegates checkpoint validation to the registered engine.

## Device and compute type

Select the device and precision:

```python
model = Model("drama-3", device="cuda", device_index=0, compute_type="float16")
model = Model("drama-3", device="cpu", compute_type="int8")
model = Model("drama-3", device="auto", compute_type="default")
```

The default device is `"cpu"`; `"auto"` selects GPU when available. CUDA and ROCm
both use `"cuda"` with a matching CT2 build. `device_index` selects a single GPU
(default: `0`). Waveform decoding stays on CPU. Compute types depend on the
device and build; `"default"` lets CT2 choose. Inspect settings with `model.info()`.

```bash
python -m faster_tts --device cuda --device-index 0 --compute-type float16 \
  --text "Hello!" --output hello.wav
```

## Expert memory on GPU and CPU (Drama 3)

Drama 3 accepts `gpu_expert_cache_gb` via `engine_options`. Other engines
ignore the `CT2_DRAMA3_*` environment variables and reject
`gpu_expert_cache_gb` as an unknown option.

On GPU, `"auto"` sizes the expert cache from free memory, reserving at least
2 GB and 30% for other allocations. Override it with a total budget in decimal GB:

```python
model = Model(device="cuda", engine_options={"gpu_expert_cache_gb": "auto"})
model = Model(
    device="cuda", num_workers=2, engine_options={"gpu_expert_cache_gb": 12}
)
```

Budgets are divided across workers (6 GB each above); `model.info()` reports them.
Omitting the option (or passing `None`) uses `CT2_DRAMA3_GPU_EXPERT_CACHE_GB`
per worker; unset or zero means uncapped. CLI:
`--gpu-expert-cache-gb auto` or a numeric budget.

When `device="auto"` selects CPU, expert memory mapping is enabled if `model.bin`
exceeds available RAM after reserving at least 2 GB and 30% headroom. The OS
manages cached pages. Set `CT2_DRAMA3_CPU_MMAP=0` or `1` before loading to override.
Requires Linux/macOS and `compute_type="default"` or `"float32"`; unavailable RAM
detection leaves CT2's default (off). Explicit `device="cpu"` keeps that default.
This uses checkpoint size as an estimate; keep the checkpoint unchanged while loaded.

Both features require a supporting CT2 build and do not limit total VRAM or RAM.

## Threads and concurrent requests

Set `cpu_threads` (default: 8) per worker and `num_workers` (default: 1)
for concurrent calls from Python threads:

```python
from concurrent.futures import ThreadPoolExecutor

model = Model("drama-3", cpu_threads=4, num_workers=2)
with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(model.synthesize, ["Hello!", "How are you?"]))
```

The CLI and benchmark use `--threads`. `cpu_threads` overrides the legacy
`intra_threads` argument. Workers share CPU weights but have separate GPU expert
caches and runtime state. A single request still runs sequentially.

## Results

`Model.synthesize()` returns a `SynthesisResult` with `audio`, `sample_rate`,
`codes`, `finish_reason`, `timings`, and `attempts`.

`finish_reason="length"` means generation exhausted its frame budget and may be
truncated; `"eos"` does not guarantee complete speech. The CLI saves the WAV and
exits with status 1 on `length`.

See the [Python API reference](docs/api.md),
[documentation build instructions](docs/README.md), or `python -m faster_tts --help`
for options, and the [CT2 Drama 3 guide](https://github.com/jordimas/CTranslate2/blob/feat/drama3-cpu/docs/guides/drama3.md)
for conversion and native runtime details.

## Benchmark

Drama 3 only. See [benchmark instructions](benchmark/README.md) to run your own measurements.

<!-- current-benchmarks -->
### Benchmark results (2026-10-24)

#### CPU — AMD EPYC 7B13 (62 GiB RAM)

Test text: `Hello! Welcome to our application.`

RTF is compute time divided by generated audio duration; lower is better, and 1 means real-time synthesis. Loading and warmup are excluded.

| Implementation | Precision | Mean time (s) | Audio duration (s) | RTF | Peak RAM (GiB RSS) | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| HF / PyTorch | bfloat16 | 300.868 | 2.833 | 106.207 | 44.507 | success |
| CT2 | bfloat16 | — | — | — | — | unsupported (no matching precision) |
| HF / PyTorch | float16 | 351.489 | 3.204 | 109.691 | 44.671 | success |
| CT2 | float16 | — | — | — | — | unsupported (no matching precision) |
| HF / PyTorch | int8 | — | — | — | — | unsupported (no matching precision) |
| CT2 | int8 | 63.644 | 2.601 | 24.473 | 48.619 | success |
| HF / PyTorch | int8_float32 | — | — | — | — | unsupported (no matching precision) |
| CT2 | int8_float32 | 62.379 | 2.601 | 23.986 | 44.711 | success |
| HF / PyTorch | float32 | 155.203 | 2.972 | 52.219 | 51.252 | success |
| CT2 | float32 | 37.660 | 2.508 | 15.017 | 45.470 | success |
