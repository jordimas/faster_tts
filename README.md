# faster_tts

`faster_tts` is a Python text-to-speech library built on CTranslate2. Its goal is
to provide a common interface for multiple TTS engines supported by CTranslate2:
load a model, synthesize text, and use the returned audio.

The library handles model downloading and caching, tokenization, prompt
preparation, text chunking, and audio output. CTranslate2 runs inference and
waveform decoding. Models are loaded once and reused across requests, with
advanced generation and runtime controls available when needed.

Requires CTranslate2 **4.8.3 or higher** and a compatible converted checkpoint.
The current implementation supports **Drama 3 with CPU inference and optional GPU acceleration**; support for additional
TTS engines is planned. A matching CTranslate2 build exposing `Drama3` is
currently required. Portable installation remains unverified.

## Quick start

Load the default model and generate speech:

```python
from faster_tts import Model

model = Model()
result = model.synthesize("Hello! Welcome to our application.")

audio = result.audio
sample_rate = result.sample_rate
```

`Model()` downloads or reuses the cached
[`jordimas/drama-3-ct2`](https://huggingface.co/jordimas/drama-3-ct2) checkpoint.
Reuse the same model for subsequent `synthesize()` calls. `result.audio` is a
NumPy array; use `result.sample_rate` when playing or saving it.

To save speech directly to a WAV file, use the CLI:

```bash
python -m faster_tts \
  --text "Hello, how are you?" --output hello.wav
```

## Model loading

Model loading accepts a short name, a compatible converted Hub repository ID,
or a local directory:

```python
model = Model("drama-3")  # Alias for the default checkpoint.
model = Model("jordimas/drama-3-ct2", revision="main", cache_dir="./models")
model = Model("/path/to/converted-model")
model = Model(local_files_only=True)  # Use only cached assets.
```

Like faster-whisper, `faster_tts` maps recognized short names to predefined
Hugging Face repositories, then downloads or reuses the cached checkpoint:

| Short name | Hugging Face repository |
| --- | --- |
| `drama-3` | `jordimas/drama-3-ct2` |

`Model()` uses this same default repository. An explicit Hub repository ID or
local directory lets you load a compatible custom checkpoint without adding an
internal alias. A custom checkpoint must use an engine and converted layout
supported by the installed library and CTranslate2 build.

As additional TTS engines are added, the same `Model(...).synthesize(...)` API
will remain the entry point. Aliases will select checkpoints, and the
checkpoint's `config.json` `model_type` field will select the engine adapter.
Currently, only Drama 3 (`model_type="drama3"`) is implemented; automatic
dispatch between multiple engines is planned.

The CLI accepts the same optional model argument, plus `--revision`, `--cache-dir`,
and `--local-files-only`. The `hf://` prefix remains supported. Existing local
directories take precedence; use a `Path` object or `./`, `../`, `~/`, or an
absolute path to keep missing local paths from being interpreted as Hub IDs.
Models must already be converted to CTranslate2 format.

## Device and compute type

Select the device and precision using faster-whisper parameter names:

```python
model = Model("drama-3", device="cuda", device_index=0, compute_type="float16")
model = Model("drama-3", device="cpu", compute_type="int8")
model = Model("drama-3", device="auto", compute_type="default")
```

`device` accepts `"cpu"` (default), `"cuda"`, or `"auto"` (GPU when available,
otherwise CPU). `device_index` selects one GPU and defaults to `0`; lists of
indices are unsupported by the native Drama 3 binding. CUDA and ROCm both use
`"cuda"` and require a matching CTranslate2 build. GPU acceleration covers
backbone and tower projections; waveform decoding stays on CPU. Supported
compute types depend on the device and build; `"default"` lets CTranslate2
choose. `model.info()` reports the selected device and requested compute type.

```bash
python -m faster_tts --device cuda --device-index 0 --compute-type float16 \
  --text "Hello!" --output hello.wav
```

## Threads and concurrent requests

Set the CPU thread count with `cpu_threads` (default: 8):

```python
model = Model("drama-3", cpu_threads=4, num_workers=2)
```

The CLI and benchmark accept `--threads 4`. The existing `intra_threads`
argument remains supported; `cpu_threads` takes precedence when both are supplied.

Like faster-whisper, `num_workers` (default: 1) enables parallel inference when
you call `synthesize()` from multiple Python threads:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(model.synthesize, ["Hello!", "How are you?"]))
```

Each worker uses the configured CPU thread count. Drama 3 currently lacks a
native worker pool, so the library pools separate native model instances, each
with its own weights and runtime memory. Calls wait when all workers are busy.
Workers improve concurrent throughput; a single request still runs sequentially.
`model.info()` reports both settings. The standalone benchmark measures sequential
latency with one worker.

## Results

`Model.synthesize()` returns a `SynthesisResult` with `audio`, `sample_rate`,
`codes`, `finish_reason`, `timings`, and `attempts`. Audio format, codec codes,
languages, and generation options depend on the supported engine.

`finish_reason="length"` means generation exhausted its frame budget and may be
truncated; `"eos"` does not guarantee complete speech. The CLI saves the WAV and
exits with status 1 on `length`.

## Performance benchmarking

See [benchmark instructions](benchmark/README.md).

## Engine documentation

For Drama 3 conversion, native runtime options, precision, and memory requirements,
see the [CTranslate2 Drama 3 guide](https://github.com/jordimas/CTranslate2/blob/feat/drama3-cpu/docs/guides/drama3.md).

For this library's synthesis options, see the
[`Model.synthesize()` API documentation](faster_tts/model.py) or run
`python -m faster_tts --help` for CLI options.

## Benchmark results (2026-10-19)

AMD Ryzen AI MAX+ PRO 395 (16 cores / 32 threads), Radeon 8060S
(`gfx1151`, ROCm, 61.41 GiB GPU-addressable shared memory), 122.8 GiB system
RAM; Linux 7.0.0-38-generic. Eight CPU threads, text
`Hello! Welcome to our application.`, one full warmup and three timed runs
per configuration. HF ran first, then CT2; configurations run sequentially.
GPU expert caches are uncapped; capped profiles were skipped.

Time is mean synthesis latency, excluding model loading and warmup. Peak
memory is process RSS including loading and warmup; it excludes GPU
allocations, which also consume shared system RAM on this integrated GPU.
Audio duration varies by backend and precision, so compare RTF alongside time.
No quality checks were enabled.

CTranslate2 4.8.3, Python 3.14; PyTorch/torchaudio 2.8.0+rocm6.4,
Python 3.13. CT2 checkpoint: `jordimas/drama-3-ct2` revision
`9aba67191f7f44b915c63ae3bfa08dde5eafdd84`. HF checkpoint:
`/home/jordi/sc/audio/models/drama-3`; Fish Speech source revision
`541566932abbaebcb0a40e562a1f846738383f38`. Both decode audio on CPU float32.

| Backend | Device | Compute type | Mean time (s) | Std (s) | Audio (s) | RTF | Peak RSS (GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| HF / PyTorch | CPU | bfloat16 | 109.206 | 0.128 | 3.158 | 34.582 | 45.564 |
| HF / PyTorch | CPU | float16 | 187.052 | 5.279 | 3.158 | 59.233 | 45.566 |
| HF / PyTorch | CPU | float32 | 105.888 | 0.474 | 2.508 | 42.224 | 52.434 |
| CT2 | CPU | float32 | 135.560 | 0.676 | 2.508 | 54.056 | 45.263 |
| CT2 | CPU | int8 | 27.462 | 0.145 | 2.554 | 10.752 | 44.752 |
| CT2 | CPU | int8_float32 | 27.862 | 0.354 | 2.554 | 10.908 | 44.751 |
| CT2 | ROCm GPU (cuda:0) | bfloat16 | 26.362 | 0.059 | 2.508 | 10.512 | 46.764 |
| CT2 | ROCm GPU (cuda:0) | float16 | 29.794 | 0.457 | 3.111 | 9.576 | 47.038 |
| CT2 | ROCm GPU (cuda:0) | float32 | 43.638 | 0.949 | 2.508 | 17.401 | 46.968 |
| CT2 | ROCm GPU (cuda:0) | int8 | 12.312 | 0.653 | 3.158 | 3.899 | 45.705 |
| CT2 | ROCm GPU (cuda:0) | int8_bfloat16 | 29.212 | 0.474 | 3.158 | 9.250 | 48.735 |

HF GPU bfloat16, float16, and float32 all failed during model allocation
with HIP out-of-memory errors, before warmup or timing. PyTorch reported
60.96 GiB allocated for bfloat16/float16 and 60.39 GiB for float32 against a
61.41 GiB GPU capacity; these are allocation-failure diagnostics, not measured
peak RSS. The official ROCm path expands FP8 experts to the requested floating
precision. No GPU timing is available for HF.

Commands from `benchmark/`:

```bash
/tmp/faster-tts-hf-rocm/bin/python -u benchmark_hf.py /home/jordi/sc/audio/models/drama-3 --runs 3
HF_HUB_OFFLINE=1 ../.venv/bin/python -u benchmark.py --memory-profile default --runs 3
```

Results above include all completed configurations (8/10 CT2); the remaining CT2 GPU configurations are still running.
