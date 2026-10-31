"""Reference audio loading and resampling for voice cloning."""

import io
import math
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_references(references):
    """Return speaker-tagged transcripts and mono float32 PCM at 44100 Hz."""
    if isinstance(references, (Mapping, str, bytes)):
        raise TypeError("references must be an iterable of reference mappings")
    entries = []
    for reference in references:
        if not isinstance(reference, Mapping):
            raise TypeError("each reference must be a mapping")
        if set(reference) - {"audio", "text", "speaker"}:
            raise ValueError(
                "reference fields must be audio, text, and optional speaker"
            )
        text = reference.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("reference text must be a nonempty transcript")
        speaker = reference.get("speaker")
        if speaker is not None:
            if (
                isinstance(speaker, bool)
                or not isinstance(speaker, Integral)
                or not 0 <= speaker <= 15
            ):
                raise ValueError("reference speaker must be an integer in 0..15")
            if not text.lstrip().startswith("<|speaker:"):
                text = f"<|speaker:{speaker}|> {text}"
        audio = reference.get("audio")
        if not isinstance(audio, (str, bytes, Path)):
            raise TypeError(
                "reference audio must be a file path or encoded audio bytes"
            )
        entries.append((text, audio))
    if not entries:
        return []
    loaded = []
    for text, audio in entries:
        waveform, rate = sf.read(
            io.BytesIO(audio) if isinstance(audio, bytes) else audio,
            dtype="float32",
            always_2d=True,
        )
        if not waveform.size or not np.all(np.isfinite(waveform)):
            raise ValueError("reference audio must be nonempty and finite")
        waveform = waveform.mean(axis=1)
        if rate != 44100:
            divisor = math.gcd(rate, 44100)
            waveform = resample_poly(waveform, 44100 // divisor, rate // divisor)
        loaded.append((text, np.ascontiguousarray(waveform, dtype=np.float32)))
    return loaded
