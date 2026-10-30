[![PyPI version](https://img.shields.io/pypi/v/faster-tts.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/faster-tts/)
[![PyPI downloads](https://img.shields.io/pypi/dm/faster-tts.svg)](https://pypistats.org/packages/faster-tts)

# faster_tts

The goal is to provide a Python library for accessing all text-to-speech models
supported by CTranslate2.

Text-to-speech with **Drama 3 on CPU and GPU**, powered by CTranslate2.
Provides a Python library API and the `faster-tts` command-line interface (CLI).
Includes model downloading and caching, sentence splitting, speaker tags, and WAV output.
Requires CTranslate2 **4.8.5+**.

Drama 3 supports [83 languages](https://fish.audio/blog/fish-audio-drama-3/).
With this library and CTranslate2, you can use:

- Natural-language directions for emotion, tone, and pacing, such as `[whispering]`.
- Inline vocal sounds such as laughter and breathing, and delivery tags such as
  `<emphasis>…</emphasis>` for individual words or phrases.
- Speaker-tagged dialogue using `<|speaker:0|>`, `<|speaker:1|>`, and so on.
- Offline synthesis with mono audio output at 44.1 kHz on CPU or GPU.

## Quick start

Python API:

```python
from faster_tts import Model

model = Model()
result = model.synthesize(
    "Hello! Welcome to our application.",
    output="hello.wav",  # Save the audio as a WAV file.
)
```

`Model()` downloads and caches
[`jordimas/drama-3-ct2`](https://huggingface.co/jordimas/drama-3-ct2).
Reuse it across calls.

CLI (also available as `python -m faster_tts`):

```bash
faster-tts --text "Hello!" --output hello.wav
```

Generate emotion, dialogue, and multilingual audio with the
[sample scripts](https://github.com/jordimas/faster_tts/blob/main/samples/README.md), or narrate a UTF-8 article:

```bash
python samples/article_voice.py article.txt --output article.wav
```

The article sample selects the device automatically. Use `--device cpu` or
`--device cuda` to override it. Its default audio budget is 120 seconds
(`--max-duration-s`); reaching it saves partial narration.

## Models and devices

```python
model = Model("drama-3")  # Default checkpoint
model = Model("jordimas/drama-3-ct2", revision="main", cache_dir="./models")
model = Model("/path/to/converted-model")
model = Model(local_files_only=True)  # Cached assets only

model = Model(device="cuda", device_index=0, compute_type="float16")
model = Model(device="cpu", compute_type="int8")
```

The default `device="auto"` selects GPU when available, otherwise CPU.
CUDA and ROCm both use `"cuda"` with a matching CT2 build. Compute types depend
on the build; `"default"` lets CT2 choose. Inspect settings with `model.info()`.
Set `CT2_DRAMA_GPU_WAVEFORM=0` to decode waveforms on CPU.

The CLI accepts the same model and cache options. For local models, use explicit
paths such as `./model` to avoid interpreting missing directories as Hub IDs.
Custom checkpoints must match the engine's layout.

## Memory

GPU expert caches size automatically. Set a total budget in decimal GB:

```python
model = Model(
    device="cuda", num_workers=2,
    engine_options={"gpu_expert_cache_gb": 12},
)
```

This gives each worker a 6 GB cache. Use `"auto"` or `"uncapped"` instead of a
number, or CLI `--gpu-expert-cache-gb`. Without an explicit budget,
`CT2_DRAMA3_GPU_EXPERT_CACHE_GB` sets the per-worker limit (`0` means uncapped).
Automatic sizing reserves at least 2 GB and 30% of free memory and fails if
memory cannot be queried or is insufficient. Cache budgets do not cap total VRAM.

When `device="auto"` selects CPU, large checkpoints can use expert memory mapping
on Linux/macOS with `compute_type="default"` or `"float32"`. Override with
`CT2_DRAMA3_CPU_MMAP=0` or `1`. Explicit `device="cpu"` leaves it off by default.
Keep the checkpoint unchanged while loaded; mapping does not cap RAM usage.
Both memory features require a supporting CT2 build.

## Synthesis options and results

Pass engine options through `engine_options`. Experimental `directed_spans`
splits explicit breathing events from speech:

```python
result = model.synthesize(
    "Set the bag down. [breathing] We can continue now.",
    engine_options={"directed_spans": True, "max_frames": 534},
)
```

Speech spans are capped at 256 frames and breathing events at 22, within the
overall budget. Events use empty history and do not condition later speech.
Listen to check voice, prosody, and event quality. CLI: `--directed-spans`.

[`SynthesisResult`](https://github.com/jordimas/faster_tts/blob/main/docs/api.md#synthesisresult) includes audio, sample rate, and
`finish_reason`. A `"length"` result may be truncated; the CLI saves it and exits
with status 1. `"eos"` does not guarantee complete speech.
Enable INFO logging for `faster_tts.engines.drama3` for device and segment details.

See the [API reference](https://github.com/jordimas/faster_tts/blob/main/docs/api.md), `python -m faster_tts --help`, and
[CT2 Drama 3 guide](https://github.com/OpenNMT/CTranslate2/blob/master/docs/guides/drama3.md)
for options and conversion instructions.

## Benchmark

Drama 3 only. See [benchmark instructions](https://github.com/jordimas/faster_tts/blob/main/benchmark/README.md) to run your own measurements.

<!-- current-benchmarks -->
### Benchmark results (2026-10-24)

#### CPU — AMD EPYC 7B13 (62 GiB RAM)

Test text: `Hello! Welcome to our application.`

Times in the CPU table are means. RTF is compute time divided by generated audio duration; lower is better, and 1 means real-time synthesis. Loading and warmup are excluded.

| Implementation | Precision | Time (s) | Audio duration (s) | RTF | Peak RAM (GiB RSS) | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| HF / PyTorch | bfloat16 | 300.9 | 2.8 | 106.2 | 44.5 | success |
| CT2 | bfloat16 | — | — | — | — | unsupported (no matching precision) |
| HF / PyTorch | float16 | 351.5 | 3.2 | 109.7 | 44.7 | success |
| CT2 | float16 | — | — | — | — | unsupported (no matching precision) |
| HF / PyTorch | int8 | — | — | — | — | unsupported (no matching precision) |
| CT2 | int8 | 18.4 | 2.4 | 7.6 | 44.7 | success |
| HF / PyTorch | int8_float32 | — | — | — | — | unsupported (no matching precision) |
| CT2 | int8_float32 | 17.9 | 2.4 | 7.4 | 44.7 | success |
| HF / PyTorch | float32 | 155.2 | 3.0 | 52.2 | 51.3 | success |
| CT2 | float32 | 37.7 | 2.5 | 15.0 | 45.5 | success |

<!-- gpu-benchmarks-ci40090377176 -->
#### GPU — NVIDIA Tesla T4 (15 GiB VRAM)

CT2 4.8.4, 8 CPU threads, one warmup and three timed runs per successful configuration.
Same test text as above. Times are means; peak RAM is process RSS and excludes VRAM.
Expert cache: auto (10.874 GB), except the successful float16 run (8 GB).

| Implementation | Precision | Time (s) | Audio duration (s) | RTF | Peak RAM (GiB RSS) | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CT2 | float16 | — | — | — | — | CUDA OOM during warmup |
| CT2 | float32 | 34.5 | 2.5 | 13.8 | 34.5 | success |
| CT2 | int8 | 27.3 | 4.3 | 6.3 | 36.6 | success |
| CT2 | int8_float16 | 26.9 | 3.0 | 9.0 | 35.9 | success |
| CT2 | int8_float32 | 27.2 | 4.3 | 6.3 | 36.6 | success |
| CT2 | float16 | 37.9 | 3.7 | 10.3 | 36.0 | success |
<!-- /gpu-benchmarks-ci40090377176 -->

<!-- gpu-benchmarks-l4-ci40090377176 -->
#### GPU — NVIDIA L4

CT2 4.8.4, 8 CPU threads, a 12 GB expert cache, one warmup and one timed run per configuration.
Same test text as above. Loading and warmup are excluded; peak RAM is process RSS and excludes VRAM.
Single runs do not measure timing variability.

| Implementation | Precision | Time (s) | Audio duration (s) | RTF | Peak RAM (GiB RSS) | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CT2 | bfloat16 | 26.5 | 2.8 | 9.5 | 35.5 | success |
| CT2 | float16 | 27.6 | 3.1 | 8.9 | 35.6 | success |
| CT2 | float32 | 29.4 | 2.5 | 11.7 | 34.6 | success |
| CT2 | int8 | 22.1 | 4.3 | 5.1 | 36.6 | success |
| CT2 | int8_bfloat16 | 21.4 | 2.8 | 7.6 | 35.1 | success |
| CT2 | int8_float16 | 22.8 | 3.0 | 7.7 | 35.9 | success |
| CT2 | int8_float32 | 22.1 | 4.3 | 5.1 | 36.7 | success |
<!-- /gpu-benchmarks-l4-ci40090377176 -->

Contact: Jordi Mas <jmas@softcatala.org>
