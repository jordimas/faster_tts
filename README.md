# Experimental Drama 3 CPU inference

This package requires the matching experimental CTranslate2 wheel exposing
`Drama3` and a converted checkpoint. Ordinary upstream CT2 wheels do not guarantee
that binding. This is a local functional prototype; held-out generation reliability
and portable installation remain open.

```python
from faster_tts import Model

model = Model("/path/to/converted-drama3", compute_type="int8_float32")
result = model.synthesize("Hello, how are you?")
print(result.sample_rate, result.finish_reason)
```

A bounded retry is optional and disabled by default:

```python
result = model.synthesize("Hello, how are you?", seed=0, max_retries=1)
print(result.attempts)  # Includes capped attempts and their cost.
```

Only sampled generation ending at the frame limit retries once, on the same
loaded model with fresh native state and the next seed. EOS, greedy decoding,
and errors never retry. This can double computation; a second capped attempt
returns its partial audio. Attempt metadata is retained, discarded audio/codes
are not, and `result.timings` sums native phase timings across all attempts.
Retrying does not detect incomplete speech that ends with EOS or establish
speech quality.

## Multi-sentence text

Plain text is generated sentence by sentence, conditioning later sentences on
completed text/audio from the first and latest chunks. Returned audio and codes
contain every generated chunk in order, without silence trimming or padding.
`max_frames` is the total returned-frame budget, shared across chunks. The result
reports `eos` only if every chunk reaches EOS; otherwise it reports `length`.
An empty generation raises an error. Check the final status before using audio.

`result.attempts` records each chunk's text/index, seed, stop reason and cost.
The same sampling settings/seed apply to each chunk. A retry is still opt-in,
per chunk, and its cost is retained. This is a mitigation for observed sentence
omissions, not a guarantee of complete content or consistent speaker identity.

The conservative English/Spanish splitter preserves common abbreviations,
initials and decimals. A single leading speaker tag is repeated on every chunk.
Dialogue with multiple tags, other markup and bracketed cues stays intact; use
explicit history for those inputs. For single-pass comparison,
use `split_sentences=False` or CLI `--no-split-sentences`.

## Compute type and memory

Use `compute_type="int8_float32"` (the default) with a checkpoint converted using
the intended quantization policy. With `experts_int8_sensitive_float32`, routed
experts remain INT8 while the other 519 matrix weights remain FLOAT32, including
embeddings, routers, shared experts and output heads. The independently audited
mixed checkpoint has 47,825,164,164 bytes of tensor payload (44.54 GiB); runtime
memory additionally includes caches, scratch buffers and process overhead. This
payload size is not a measured inference peak or a quality acceptance result.

`compute_type="float32"` dequantizes the routed expert bank during native loading;
it does not merely select FLOAT32 activations. Those 30,720 expert matrices alone
expand from 30 GiB of INT8 values to **120 GiB** of FLOAT32 values. The complete
FLOAT32 tensor payload is approximately **134.42 GiB** (144,336,099,204 bytes),
before runtime allocations or conversion temporaries. It exceeds both this
machine's approximately 122 GiB RAM and the intended 60 GiB memory budget. This
estimate follows the audited tensor inventory and loader conversion behavior;
no full FLOAT32 load was attempted. FLOAT32 remains available for machines with
sufficient memory.

The mixed checkpoint's on-disk audit covers all 519 FLOAT32 matrices. Native tiny
fixtures verify representative weight preservation and FLOAT32 Dense execution;
they do not enumerate all 519 variables in a freshly loaded full model. Structural
audits do not establish speech quality.

The CLI keeps the WAV for inspection but exits with status 1 if generation hits
the frame limit. An EOS result exits successfully; EOS alone does not establish
complete content or speech quality. The Python API retains the explicit
`finish_reason` field for callers to check.

To generate explicit sentence-sized chunks with earlier audio as conditioning:

```python
first = model.synthesize("The train followed the river through the valley.")
if first.finish_reason != "eos":
    raise RuntimeError("First chunk is incomplete")
second = model.synthesize(
    "From her seat, Maya watched the fields slowly disappear behind the trees.",
    history=[("The train followed the river through the valley.", first.codes)],
)
```

`history` contains completed `(text, codes)` turns; codes have shape `[frames,10]`.
History counts towards the context limit and is not included in returned audio.
Check every chunk's `finish_reason`. Explicit history also supports caller-chosen
boundaries; it does not guarantee complete content or establish speaker similarity.
