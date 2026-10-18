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

## Current engine: Drama 3

Drama 3 currently generates mono FLOAT32 audio at 44,100 Hz and INT32 codec
`codes` shaped `[frames, 10]`. Text input accepts the languages supported by the
model without a language selector. Speech quality across those languages has
not been validated in this runtime.

The following generation and precision details apply to the current Drama 3
implementation.

### Generation

Text splits using Unicode sentence punctuation, including CJK, Arabic, Indic
and other script-specific marks. Long sentences split at clause punctuation with
a soft target of 200 characters. Known abbreviations, Unicode initials and
decimals stay together. Splitting is heuristic: unfamiliar abbreviations may
split, and text without clear boundaries stays intact. A leading speaker tag repeats
on each chunk. Sentences with their own leading bracketed cues can split;
ambiguous cue scope, multiple speaker tags and other markup stay intact.
Use `split_sentences=False` or `--no-split-sentences` for a single pass.

Later chunks use completed text/audio from the first and latest chunks as
history. Audio is concatenated without silence trimming or padding.
`max_frames=512` is the shared output budget; `eos` requires every chunk to
finish. Splitting does not guarantee content completeness or speaker continuity.

Sampling defaults are temperature 0.7, top-k 30 and top-p 0.7, applied with the
same seed to each chunk. Native CTranslate2 defaults to temperature/top-p 0.8.
`greedy=True` enables diagnostic greedy decoding.

Optional `max_retries=1` retries each sampled, frame-limited chunk once with the
next seed. It can double computation and returns partial audio if the retry also
hits the limit. EOS, greedy decoding and errors do not retry. `attempts` retains
attempt costs and chunk metadata; `timings` includes discarded attempts.

For explicit conditioning, pass completed turns as
`history=[("Previous text.", previous_result.codes)]`. History counts towards the
context limit and is excluded from returned audio. Check each turn's
`finish_reason` before reusing it.

### Precision and memory

`compute_type="default"` follows the converted package precision. Default FP8
packages retain original expert bytes and block scales with FLOAT32 sensitive
weights and codec. FLOAT32 execution decodes selected experts on demand, without
expanding the entire bank. The audited FP8 tensor payload is about 44.43 GiB;
runtime allocations require additional memory.

CTranslate2 also supports standard quantized packages and runtime conversion,
including `compute_type="int8_float32"`. Converting FP8 weights at load time
requires additional temporary memory. Available compute types depend on the
selected CTranslate2 backend and hardware.
