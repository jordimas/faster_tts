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

Fish Speech expands FP8 experts to the selected floating precision on ROCm;
all three precisions ran out of memory during model allocation, before timing.

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
