# faster_tts

`faster_tts` is a Python text-to-speech library built on CTranslate2. Its goal is
to provide a common interface for multiple TTS engines supported by CTranslate2:
load a model, synthesize text, and use the returned audio.

The library handles model downloading and caching, tokenization, prompt
preparation, text chunking, and audio output. CTranslate2 runs inference and
waveform decoding. Models are loaded once and reused across requests, with
advanced generation and runtime controls available when needed.

Requires CTranslate2 **4.8.3 or higher** and a compatible converted checkpoint.
The current implementation supports **Drama 3 on CPU**; support for additional
TTS engines is planned. A matching CTranslate2 build exposing `Drama3` is
currently required. Portable installation and broad speech reliability remain
unverified.

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

The CLI accepts the same optional model argument, plus `--revision`, `--cache-dir`,
and `--local-files-only`. The `hf://` prefix remains supported. Existing local
directories take precedence; use a `Path` object or `./`, `../`, `~/`, or an
absolute path to keep missing local paths from being interpreted as Hub IDs.
Models must already be converted to CTranslate2 format.

## Results

`Model.synthesize()` returns a `SynthesisResult` with `audio`, `sample_rate`,
`codes`, `finish_reason`, `timings`, and `attempts`. Audio format, codec codes,
languages, and generation options depend on the supported engine.

`finish_reason="length"` means generation exhausted its frame budget and may be
truncated; `"eos"` does not guarantee complete speech. The CLI saves the WAV and
exits with status 1 on `length`.

## Engine documentation

For Drama 3 conversion, native runtime options, precision, and memory requirements,
see the [CTranslate2 Drama 3 guide](https://github.com/jordimas/CTranslate2/blob/feat/drama3-cpu/docs/guides/drama3.md).

For this library's synthesis options, see the
[`Model.synthesize()` API documentation](faster_tts/model.py) or run
`python -m faster_tts --help` for CLI options.
