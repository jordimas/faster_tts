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

Requires CTranslate2 **4.8.4 or higher** and a compatible
converted checkpoint. Portable installation remains unverified.

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
local paths as Hub IDs. Custom checkpoints must use the supported Drama 3 layout.

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

## GPU expert-cache budget

Limit cached GPU expert weights with an explicit budget in decimal GB or automatic sizing:

```python
model = Model(device="cuda", gpu_expert_cache_gb="auto")
model = Model(device="cuda", gpu_expert_cache_gb=12, num_workers=2)
```

Budgets are shared across workers (6 GB each in the second example). `"auto"`
reserves at least 2 GB and 30% of free GPU memory for other allocations.
This caps expert weights, not total VRAM or RAM, so out-of-memory errors remain possible.
`model.info()` reports the selected budgets.

```bash
python -m faster_tts --device cuda --gpu-expert-cache-gb auto --text "Hello!" --output hello.wav
```

The default (`None`) preserves `CT2_DRAMA3_GPU_EXPERT_CACHE_GB`, which applies
per worker; unset or zero means uncapped. Requires GPU execution and a CT2 build
supporting this setting. If automatic memory detection fails, use an explicit budget.

## Threads and concurrent requests

Set the CPU thread count with `cpu_threads` (default: 8):

```python
model = Model("drama-3", cpu_threads=4, num_workers=2)
```

The CLI and benchmark use `--threads`. `cpu_threads` overrides the legacy
`intra_threads` argument.

`num_workers` (default: 1) enables concurrent calls from Python threads:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(model.synthesize, ["Hello!", "How are you?"]))
```

`num_workers` maps to CT2's `inter_threads`; `cpu_threads` maps to `intra_threads`.
Native replicas share CPU checkpoint weights, while GPU expert caches and runtime
state are per worker. A single blocking request still runs sequentially.

## Results

`Model.synthesize()` returns a `SynthesisResult` with `audio`, `sample_rate`,
`codes`, `finish_reason`, `timings`, and `attempts`.

`finish_reason="length"` means generation exhausted its frame budget and may be
truncated; `"eos"` does not guarantee complete speech. The CLI saves the WAV and
exits with status 1 on `length`.

See [`Model.synthesize()`](faster_tts/model.py) or `python -m faster_tts --help`
for options, and the [CT2 Drama 3 guide](https://github.com/jordimas/CTranslate2/blob/feat/drama3-cpu/docs/guides/drama3.md)
for conversion and native runtime details.

## Benchmark

See [benchmark instructions](benchmark/README.md) to run your own measurements.
