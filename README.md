# Experimental Drama 3 CPU inference

Requires a matching experimental CTranslate2 build exposing `Drama3` and a
converted checkpoint. This wrapper currently uses CPU only. Portable installation
and broad speech reliability remain unverified.

Text input accepts all languages supported by
[Drama 3](https://huggingface.co/fishaudio/drama-3), without a language selector.
Speech quality across those languages has not been validated in this runtime.

Raw CTranslate2 accepts prepared token/audio rows. This library adds a text API
with tokenizer and prompt handling, automatic chunking with audio history,
optional retries, result validation, and a WAV-writing CLI. It also resolves
explicit `hf://` model identifiers with the optional `hub` dependency. Inference
and waveform decoding still run in CTranslate2; this wrapper adds no separate
model or inference engine.

```python
from faster_tts import Model

model = Model("/path/to/converted-drama3")
result = model.synthesize("Hello, how are you?")
print(result.sample_rate, result.finish_reason)
```

```bash
python -m faster_tts /path/to/converted-drama3 \
  --text "Hello, how are you?" --output hello.wav
```

Results contain mono FLOAT32 `audio` at 44,100 Hz, INT32 `codes` shaped
`[frames, 10]`, `finish_reason`, `timings`, and `attempts` metadata.
`finish_reason="length"` means generation exhausted its frame budget and may be
truncated; `"eos"` does not guarantee complete speech. The CLI saves the WAV and
exits with status 1 on `length`.

## Generation

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

## Precision and memory

`compute_type="default"` follows the converted package precision. Default FP8
packages retain original expert bytes and block scales with FLOAT32 sensitive
weights and codec. FLOAT32 execution decodes selected experts on demand, without
expanding the entire bank. The audited FP8 tensor payload is about 44.43 GiB;
runtime allocations require additional memory.

CTranslate2 also supports standard quantized packages and runtime conversion,
including `compute_type="int8_float32"`. Converting FP8 weights at load time
requires additional temporary memory. Available compute types follow the CPU
backend; the synthesis graph and codec use FLOAT32 activations.
