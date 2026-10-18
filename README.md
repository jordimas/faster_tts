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
