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

## Benchmark

Time and memory required to synthesize `Hello! Welcome to our application.`
with Drama 3, measured on 2026-10-19. Each result is the mean of three runs
after one full warmup; model loading and warmup are excluded from time.

### Drama 3 on GPU

| Implementation | Precision (compute_type) | Time | RAM usage (peak RSS) | RTF |
| --- | --- | ---: | ---: | ---: |
| Fish Speech (PyTorch) | bfloat16 | Out of memory | — | — |
| Fish Speech (PyTorch) | float16 | Out of memory | — | — |
| Fish Speech (PyTorch) | float32 | Out of memory | — | — |
| faster_tts (CT2) | bfloat16 | 26.362s | 46.764 GiB | 10.512 |
| faster_tts (CT2) | float16 | 29.794s | 47.038 GiB | 9.576 |
| faster_tts (CT2) | float32 | 43.638s | 46.968 GiB | 17.401 |
| faster_tts (CT2) | int8 | **12.312s** | 45.705 GiB | 3.899 |
| faster_tts (CT2) | int8_bfloat16 | 29.212s | 48.735 GiB | 9.250 |
| faster_tts (CT2) | int8_float16 | 43.077s | 49.663 GiB | 8.209 |
| faster_tts (CT2) | int8_float32 | 12.705s | 45.626 GiB | 4.023 |

Executed on AMD Radeon 8060S (`gfx1151`, ROCm, device `cuda:0`), with
61.41 GiB GPU-addressable memory shared with system RAM. All GPU runs are
uncapped. **RAM usage measures process RSS, excluding GPU allocations; it is
not VRAM usage or total shared memory consumption.**

Fish Speech expands FP8 experts to the selected floating precision on ROCm.
All three GPU precisions ran out of memory during model allocation, before
warmup or timing. PyTorch reported 60.96 GiB allocated for bfloat16/float16
and 60.39 GiB for float32 against the 61.41 GiB capacity.

### Drama 3 on CPU

| Implementation | Precision (compute_type) | Time | RAM usage (peak RSS) | RTF |
| --- | --- | ---: | ---: | ---: |
| Fish Speech (PyTorch) | bfloat16 | 109.206s | 45.564 GiB | 34.582 |
| Fish Speech (PyTorch) | float16 | 187.052s | 45.566 GiB | 59.233 |
| Fish Speech (PyTorch) | float32 | 105.888s | 52.434 GiB | 42.224 |
| faster_tts (CT2) | float32 | 135.560s | 45.263 GiB | 54.056 |
| faster_tts (CT2) | int8 | **27.462s** | 44.752 GiB | 10.752 |
| faster_tts (CT2) | int8_float32 | 27.862s | 44.751 GiB | 10.908 |

Executed with 8 threads on AMD Ryzen AI MAX+ PRO 395 (16 cores / 32 threads),
122.8 GiB system RAM, Linux 7.0.0-38-generic.

Lower time and RTF are better; RTF below 1 means faster than real time.
Sampling can produce different audio lengths across implementations and
precisions. These measurements do not establish equivalent audio quality;
no ASR quality checks were run. Both implementations decode audio on CPU float32.
Peak RSS includes model loading and warmup.

<details>
<summary>Run variation, audio duration, and reproduction</summary>

| Device | Implementation | Precision | Time std (s) | Mean audio (s) |
| --- | --- | --- | ---: | ---: |
| CPU | Fish Speech (PyTorch) | bfloat16 | 0.128 | 3.158 |
| CPU | Fish Speech (PyTorch) | float16 | 5.279 | 3.158 |
| CPU | Fish Speech (PyTorch) | float32 | 0.474 | 2.508 |
| CPU | faster_tts (CT2) | float32 | 0.676 | 2.508 |
| CPU | faster_tts (CT2) | int8 | 0.145 | 2.554 |
| CPU | faster_tts (CT2) | int8_float32 | 0.354 | 2.554 |
| GPU | faster_tts (CT2) | bfloat16 | 0.059 | 2.508 |
| GPU | faster_tts (CT2) | float16 | 0.457 | 3.111 |
| GPU | faster_tts (CT2) | float32 | 0.949 | 2.508 |
| GPU | faster_tts (CT2) | int8 | 0.653 | 3.158 |
| GPU | faster_tts (CT2) | int8_bfloat16 | 0.474 | 3.158 |
| GPU | faster_tts (CT2) | int8_float16 | 1.901 | 5.248 |
| GPU | faster_tts (CT2) | int8_float32 | 0.106 | 3.158 |

Software and checkpoints:

- CTranslate2 4.8.3 (local ROCm build), Python 3.14;
  `jordimas/drama-3-ct2` revision `9aba67191f7f44b915c63ae3bfa08dde5eafdd84`.
- PyTorch/torchaudio 2.8.0+rocm6.4, Python 3.13;
  official checkpoint `/home/jordi/sc/audio/models/drama-3`, Fish Speech source
  revision `541566932abbaebcb0a40e562a1f846738383f38`.

HF ran first, followed by CT2, with all configurations sequential.
Capped profiles were skipped. Commands from `benchmark/`:

```bash
/tmp/faster-tts-hf-rocm/bin/python -u benchmark_hf.py /home/jordi/sc/audio/models/drama-3 --runs 3
HF_HUB_OFFLINE=1 ../.venv/bin/python -u benchmark.py --memory-profile default --runs 3
```

</details>
