"""Public loading and synthesis API for registered CTranslate2 TTS engines."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from .base import _positive_int, get_engine_class
from .cache import DEFAULT_MODEL, resolve_model_path


@dataclass(frozen=True)
class SynthesisResult:
    """Generated speech and diagnostics returned by `Model.synthesize`.

    Attributes:
        audio: Contiguous mono float32 array `(samples,)`. Finite, unclipped.
        sample_rate: Audio sample rate in Hz. Engine-specific (44100 for Drama 3).
        codes: Contiguous int32 codec indices `(frames, codebooks)`, or `None`
            for engines without discrete codes. Drama 3 has 10 codebooks and
            2048 samples/frame.
        finish_reason: `"eos"` (stop token) or `"length"` (frame budget
            exhausted). `"eos"` does not guarantee complete speech.
        timings: Native phase names mapped to elapsed seconds, summed across
            chunks and attempts (including discarded retries). Keys depend on CT2.
        attempts: Tuple of per-attempt dicts with `seed`, `finish_reason`,
            `frames`, wall-clock `seconds`, and `timings`. Split input also
            records `chunk_index` and `text`. Discarded audio/codes not retained.
    """

    audio: np.ndarray
    sample_rate: int
    codes: np.ndarray | None
    finish_reason: str
    timings: dict = field(default_factory=dict)
    attempts: tuple = ()


def _check_engine_options(engine_class, options, phase):
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("engine_options must be a Mapping")
    allowed = getattr(engine_class, f"{phase}_options")
    unknown = set(options) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown engine_options for {engine_class.engine_model_type!r} "
            f"{phase}: {sorted(unknown)}; allowed: {sorted(allowed)}"
        )
    return dict(options)


class Model:
    """Load a converted checkpoint using its registered TTS engine.

    `config.json`'s `model_type` selects the engine. Required assets, native
    backends, GPU coverage, and compute types depend on the engine; `info()`
    reports the selection and its capabilities. Only Drama 3 is currently
    implemented. Engine-specific knobs (e.g. Drama 3's `prefill_chunk_size`
    and `gpu_expert_cache_gb`) are passed through `engine_options`; unknown
    keys raise `ValueError`.

    Args:
        model_path: Default checkpoint, `"drama-3"` alias, converted Hub ID
            (optionally `hf://`-prefixed), or local directory. Explicit paths
            never trigger downloads. Defaults to `jordimas/drama-3-ct2`.
        intra_threads: Legacy alias for `cpu_threads`. Defaults to 8.
        compute_type: CT2 precision (`"default"`, `"float16"`, `"int8"`, …);
            availability depends on the device and CT2 build.
        cpu_threads: CPU threads per worker; overrides `intra_threads`.
        num_workers: Native workers for concurrent calls. Replicas share CPU
            weights; a single call remains blocking. Defaults to 1.
        device: `"cpu"` (default), `"cuda"`, or `"auto"`. CUDA and ROCm both
            use `"cuda"`.
        device_index: GPU index. Defaults to 0.
        revision: Hub branch, tag, or commit.
        cache_dir: Hugging Face download cache directory.
        local_files_only: Use cached assets without downloading.
        engine_options: Mapping of engine-specific loader options. Drama 3
            accepts `prefill_chunk_size` (int, default 16) and
            `gpu_expert_cache_gb` (positive float, `"auto"`, or `None`;
            Drama 3 only, requires GPU execution). Unknown keys raise
            `ValueError`.

    Raises:
        ValueError: Invalid settings, incompatible checkpoint metadata, or
            unknown engine_options.
        TypeError: `engine_options` is not a mapping.
        FileNotFoundError: An explicit local checkpoint path does not exist.
        RuntimeError: Required native bindings are missing.
    """

    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        intra_threads=8,
        compute_type="default",
        *,
        cpu_threads=None,
        num_workers=1,
        device="cpu",
        device_index=0,
        revision=None,
        cache_dir=None,
        local_files_only=False,
        engine_options=None,
    ):
        if cpu_threads is not None:
            _positive_int(cpu_threads, "cpu_threads")
            intra_threads = cpu_threads
        _positive_int(intra_threads, "intra_threads")
        _positive_int(num_workers, "num_workers")
        if not isinstance(compute_type, str) or not compute_type:
            raise ValueError("compute_type must be a nonempty string")
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError('device must be "cpu", "cuda", or "auto"')
        if (
            not isinstance(device_index, int)
            or isinstance(device_index, bool)
            or not 0 <= device_index <= np.iinfo(np.int32).max
        ):
            raise ValueError("device_index must be a nonnegative int32 integer")
        directory = resolve_model_path(
            model_path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        metadata = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        engine_class = get_engine_class(metadata["model_type"])
        options = _check_engine_options(engine_class, engine_options, "init")
        self._engine_class = engine_class
        self._engine = engine_class(
            directory,
            metadata,
            intra_threads=intra_threads,
            compute_type=compute_type,
            num_workers=num_workers,
            device=device,
            device_index=device_index,
            **options,
        )

    def info(self):
        """Return resolved settings and supported engine capabilities.

        Returns:
            dict: Independent copy with `engine`, `capabilities` (tuple of
                feature names), and the resolved device/compute/worker/
                checkpoint settings. Drama 3 adds `backend`, `codec_device`,
                `mode`, `gpu_supported`, `cpu_threads`/`intra_threads`,
                `prefill_chunk_size`, `model_path`, expert-cache budgets
                (`gpu_expert_cache_gb` total and `*_per_worker`; `None`
                delegates to the environment), and checkpoint metadata
                (`source_revision`, `quantization_policy`, `matrix_dtype`,
                `activation_dtype`; any may be `None`).
        """
        return self._engine.info()

    def synthesize(
        self,
        text,
        max_duration_s=24.0,
        temperature=0.7,
        top_k=30,
        top_p=0.7,
        seed=0,
        greedy=False,
        *,
        max_retries=0,
        split_sentences=True,
        engine_options=None,
    ):
        """Synthesize speech using the selected engine.

        Codec dimensions, sample rate, and the interpretation of sampling,
        retries, sentence splitting, and speaker tags are engine-specific;
        engines reject unsupported nondefault options. Drama 3 generates
        sentence/clause-chunked audio preserving a leading speaker tag on
        each chunk. Sampling options and `seed` apply per chunk; retry
        seeds wrap uint64. Seeded sampling follows the model distribution
        without bitwise PyTorch RNG parity.

        Args:
            text: Nonempty text. For Drama 3, canonical speaker tags
                `<|speaker:0|>`–`<|speaker:15|>` are honored and other
                protocol tags are rejected; defaults to speaker 0.
            max_duration_s: Positive finite upper bound on returned audio
                duration in seconds. Engines map this to their native frame
                or token budget. Defaults to 24.0.
            temperature: Sampling temperature (positive, finite). Defaults to 0.7.
            top_k: Sampling candidate count. Defaults to 30.
            top_p: Nucleus threshold in `(0, 1]`. Defaults to 0.7.
            seed: Sampling seed in `[0, 2**64)`. Defaults to 0.
            greedy: Diagnostic greedy decoding instead of sampling.
            max_retries: 0 (default) or 1; retry a length-capped sampled
                result once with `seed+1`. Greedy never retries.
            split_sentences: Split and concatenate (default True); False
                forces one native generation.
            engine_options: Mapping of engine-specific synthesis options.
                Drama 3 accepts `history` (iterable of completed
                `(text, codes)` turns conditioning later chunks; codes are
                integer arrays `(frames, 10)` within the codec vocabulary;
                does not establish speaker continuity) and `max_frames`
                (positive int override of the frame budget derived from
                `max_duration_s`). Unknown keys raise `ValueError`.

        Returns:
            SynthesisResult: Audio, codec indices (`None` for engines
                without discrete codes), `finish_reason` (`"eos"` or
                `"length"`; `"length"` means the budget was exhausted and
                audio may end mid-utterance, and `"eos"` does not
                guarantee complete speech), `timings`, and `attempts`.

        Raises:
            TypeError: Text is not a string, or engine_options is not a mapping.
            ValueError: Invalid text, sampling settings, duration, context
                size, or unknown engine_options.
            RuntimeError: Native generation fails or returns invalid/empty audio.
        """
        if (
            not isinstance(max_duration_s, Real)
            or isinstance(max_duration_s, bool)
            or not math.isfinite(max_duration_s)
            or max_duration_s <= 0
        ):
            raise ValueError("max_duration_s must be positive and finite")
        options = _check_engine_options(
            self._engine_class, engine_options, "synthesize"
        )
        return self._engine.synthesize(
            text,
            max_duration_s=float(max_duration_s),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            greedy=greedy,
            max_retries=max_retries,
            split_sentences=split_sentences,
            **options,
        )
