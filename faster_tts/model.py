"""Thin inference API for converted Drama 3 checkpoints.

Requires a CTranslate2 build exposing _ext.Drama3. Ordinary upstream wheels may
not expose that experimental binding. Model names and Hub repository IDs resolve
converted Hub snapshots; explicit local paths never trigger downloads. No PyTorch runtime
or external TTS engine fallback is performed. Seeded sampling and diagnostic greedy decoding
do not themselves establish production speech quality.
"""

import json
import math
import time
from dataclasses import dataclass, field
from importlib import import_module
from numbers import Real

import numpy as np

from .base import BaseModel, _positive_int, register_engine
from .cache import DEFAULT_MODEL, resolve_model_path
from .chunking import split_text
from .memory import expert_cache_budget, validate_cache_budget
from .prompt import Drama3PromptBuilder


@dataclass(frozen=True)
class SynthesisResult:
    """Generated speech and diagnostics returned by `Model.synthesize`.

    Attributes:
        audio: Contiguous mono float32 NumPy array of shape `(samples,)`.
            Values are finite; no clipping or amplitude normalization is applied here.
        sample_rate: Audio sample rate in Hz (44100 for Drama 3).
        codes: Contiguous int32 codec indices of shape `(frames, 10)`.
            Each frame corresponds to 2048 audio samples.
        finish_reason: `"eos"` for a stop token, or `"length"` when generation
            exhausts its frame budget. EOS does not guarantee complete speech.
        timings: Native phase names mapped to elapsed seconds, summed across
            chunks and attempts, including discarded retries. Keys depend on CT2.
        attempts: Tuple of dictionaries with `seed`, `finish_reason`, `frames`,
            wall-clock `seconds`, and native `timings` for each attempt. Split
            input also records zero-based `chunk_index` and chunk `text`.
            Discarded retry audio and codes are not retained.
    """

    audio: np.ndarray
    sample_rate: int
    codes: np.ndarray
    finish_reason: str
    timings: dict = field(default_factory=dict)
    attempts: tuple = ()


def _native_class():
    try:
        extension = import_module("ctranslate2._ext")
        return extension.Drama3
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "This requires the experimental CTranslate2 Drama3 binding; "
            "install a matching build of this repository."
        ) from exc


@register_engine("drama3")
class Model(BaseModel):
    """Load a converted model once, then synthesize plain text.

    Generation is offline, without cloning or reference audio. Seeded categorical
    sampling follows the model distribution but is not bitwise PyTorch RNG parity.
    Conversion must provide model.bin, a Drama3 layout-version1 config.json with
    source_config, and frozen tokenizer.json.

    ``Model()`` loads jordimas/drama-3-ct2 from the Hugging Face cache or Hub.
    ``Model("drama-3")`` selects the same checkpoint. A converted Hub repository
    ID (with optional hf:// prefix) or local directory can also be supplied.
    ``device`` selects "cpu" (default), "cuda", or "auto"; ``device_index``
    selects one GPU. CUDA and ROCm builds both use "cuda". GPU acceleration
    covers backbone/tower projections; waveform decoding remains on CPU.
    ``compute_type`` is passed to the native loader (default: "default").
    ``revision`` can pin a commit; ``local_files_only=True`` prevents downloads.
    ``cpu_threads`` sets the CPU thread count (default: 8), overriding the
    legacy ``intra_threads`` argument when supplied.
    ``num_workers`` (default: 1) configures CT2's native ``inter_threads`` pool
    for concurrent calls. Replicas share CPU checkpoint weights and each uses
    ``cpu_threads`` threads and its own GPU expert cache.
    ``gpu_expert_cache_gb`` optionally caps GPU expert weights across all workers,
    in decimal GB. "auto" reserves at least 2 GB and 30% of currently free GPU
    memory for other allocations. None preserves CT2's environment configuration.
    This is not a total GPU or system memory limit.

    Args:
        model_path: Default checkpoint, `"drama-3"` alias, converted Hub ID
            (optionally prefixed with `hf://`), or local directory. Explicit
            paths never trigger downloads.
        intra_threads: Legacy CPU thread count. Defaults to 8.
        compute_type: CT2 precision, such as `"default"`, `"float16"`, or `"int8"`.
            Availability depends on the device and CT2 build.
        prefill_chunk_size: Positive number of prompt rows processed per prefill
            chunk. Defaults to 16.
        cpu_threads: Positive CPU thread count; overrides `intra_threads`.
        num_workers: Positive number of native workers for concurrent calls.
            Defaults to 1. A single call remains blocking and sequential.
        device: `"cpu"`, `"cuda"`, or `"auto"` (GPU when available).
        device_index: Nonnegative GPU index. Defaults to 0.
        revision: Optional Hub branch, tag, or commit revision.
        cache_dir: Optional Hugging Face download cache directory.
        local_files_only: Use cached/local assets without downloading.
        gpu_expert_cache_gb: Positive total expert-cache budget in decimal GB,
            `"auto"`, or `None` to preserve `CT2_DRAMA3_GPU_EXPERT_CACHE_GB`.
            Explicit and automatic budgets are divided across workers and
            require GPU execution and a supporting CT2 build.

    Raises:
        ValueError: Invalid settings or incompatible checkpoint metadata.
        FileNotFoundError: An explicit local checkpoint path does not exist.
        RuntimeError: Required native bindings or runtime capabilities are missing.

    Example:
        ```python
        from faster_tts import Model

        model = Model()
        result = model.synthesize("Hello!")
        print(result.audio.shape, result.sample_rate)
        ```
    """

    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        intra_threads=8,
        compute_type="default",
        prefill_chunk_size=16,
        *,
        cpu_threads=None,
        num_workers=1,
        device="cpu",
        device_index=0,
        revision=None,
        cache_dir=None,
        local_files_only=False,
        gpu_expert_cache_gb=None,
    ):
        validate_cache_budget(gpu_expert_cache_gb)
        if cpu_threads is not None:
            _positive_int(cpu_threads, "cpu_threads")
            intra_threads = cpu_threads
        _positive_int(intra_threads, "intra_threads")
        _positive_int(num_workers, "num_workers")
        _positive_int(prefill_chunk_size, "prefill_chunk_size")
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
        self._prompt = Drama3PromptBuilder(directory)
        self._prefill_chunk_size = prefill_chunk_size
        backend_class = _native_class()
        resolved_device = device
        if device == "auto":
            resolved_device = (
                "cuda"
                if import_module("ctranslate2").get_cuda_device_count() > 0
                else "cpu"
            )
        with expert_cache_budget(
            gpu_expert_cache_gb, resolved_device, device_index, num_workers
        ) as cache_gb:
            self._model = backend_class(
                str(directory),
                intra_threads=intra_threads,
                inter_threads=num_workers,
                compute_type=compute_type,
                device=resolved_device,
                device_index=device_index,
            )
        self._info = {
            "backend": resolved_device,
            "codec_device": "cpu",
            "mode": "offline",
            "gpu_supported": True,
            "device": resolved_device,
            "device_index": device_index,
            "compute_type": compute_type,
            "intra_threads": intra_threads,
            "cpu_threads": intra_threads,
            "num_workers": num_workers,
            "prefill_chunk_size": prefill_chunk_size,
            "model_path": str(directory),
            "gpu_expert_cache_gb": cache_gb * num_workers
            if cache_gb is not None
            else None,
            "gpu_expert_cache_gb_per_worker": cache_gb,
            "source_revision": metadata.get("source_revision"),
            "quantization_policy": metadata.get("quantization_policy"),
            "matrix_dtype": metadata.get("matrix_dtype"),
            "activation_dtype": metadata.get("activation_dtype"),
        }

    def synthesize(
        self,
        text,
        max_frames=512,
        temperature=0.7,
        top_k=30,
        top_p=0.7,
        seed=0,
        greedy=False,
        *,
        max_retries=0,
        history=(),
        split_sentences=True,
    ):
        """Return mono float32 audio and generated [frames,10] codec indices.

        Text is generated by sentence or long clause with completed text/audio
        history. A single leading speaker tag is preserved on every chunk.
        max_frames bounds the combined returned frames.
        split_sentences=False selects one native generation for comparison.
        Other marked-up text is left intact. Sampling seed/options apply to each chunk;
        attempts records chunk indices/text and all attempt costs for split input.

        history accepts completed (text, codes[frames,10]) turns for conditioning
        later chunks. It does not establish speaker continuity or speech quality.

        finish_reason='length' means the frame budget was exhausted and audio may
        end mid-utterance; 'eos' means the model emitted its stop token.
        Opt-in max_retries=1 retries a valid length-capped sampled result once,
        using seed+1 (wrapping uint64) and fresh native state on the loaded model.
        Greedy decoding never retries. This can double cost and cannot detect
        incomplete speech ending with EOS. attempts records all attempt metadata;
        discarded attempt audio/codes are not retained. timings sums native phase
        timings across attempts; exhausted retries return the last partial result.

        Args:
            text: Nonempty text, optionally containing canonical speaker tags
                `<|speaker:0|>` through `<|speaker:15|>`. Defaults to speaker 0
                when no leading tag is supplied. Reserved protocol tags are rejected.
            max_frames: Positive frame budget for the combined returned audio.
                Defaults to 512 (about 23.8 seconds at 44100 Hz).
            temperature: Positive finite sampling temperature. Defaults to 0.7.
            top_k: Positive sampling candidate count. Defaults to 30.
            top_p: Nucleus sampling threshold in `(0, 1]`. Defaults to 0.7.
            seed: Sampling seed in `[0, 2**64)`. Defaults to 0; each chunk starts
                with the same seed. Retry seeds increment with uint64 wrapping.
            greedy: Use diagnostic greedy decoding instead of sampling.
            max_retries: 0 (default) or 1; retry length-capped sampled generation.
                Greedy generation never retries.
            history: Iterable of completed `(text, codes)` turns. Codes must be
                nonempty integer arrays of shape `(frames, 10)` within the codec
                vocabulary. History consumes context but is not returned as audio.
            split_sentences: Split sentences/long clauses and concatenate audio.
                Defaults to True. Completed first and latest chunks condition
                subsequent chunks, together with supplied history.

        Returns:
            SynthesisResult: Speech, codec indices, finish status, and diagnostics.

        Raises:
            TypeError: Text is not a string.
            ValueError: Invalid text, sampling settings, history, or context size.
            RuntimeError: Native generation fails or returns invalid/empty audio.
        """
        _positive_int(max_frames, "max_frames")
        if (
            not isinstance(temperature, Real)
            or isinstance(temperature, bool)
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be positive and finite")
        _positive_int(top_k, "top_k")
        if (
            not isinstance(top_p, Real)
            or isinstance(top_p, bool)
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise ValueError("top_p must be finite and in (0,1]")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a nonnegative uint64 integer")
        if not isinstance(greedy, bool):
            raise ValueError("greedy must be a boolean")  # noqa: TRY004 - preserve API validation errors
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries not in (0, 1)
        ):
            raise ValueError("max_retries must be 0 or 1")
        if not isinstance(split_sentences, bool):
            raise ValueError("split_sentences must be a boolean")  # noqa: TRY004 - preserve API validation errors
        chunks = split_text(text) if split_sentences else [text]
        if len(chunks) > 1:
            # Validate all input before starting any partial synthesis.
            self._prompt.prepare(text)
            return self._synthesize_chunks(
                chunks,
                max_frames,
                temperature,
                top_k,
                top_p,
                seed,
                greedy,
                max_retries,
                history,
            )
        prepared = self._prompt.prepare(text, history=history)
        maximum = self._prompt.config.get("text_config", {}).get(
            "max_position_embeddings"
        )
        if isinstance(maximum, int) and len(prepared.rows) + max_frames > maximum:
            raise ValueError("prompt plus max_frames exceeds model context capacity")
        if np.any(prepared.rows < 0) or np.any(prepared.rows > np.iinfo(np.int32).max):
            raise ValueError("prompt token IDs do not fit the native int32 interface")
        rows = np.ascontiguousarray(prepared.rows, dtype=np.int32)
        attempts = []
        aggregate_timings = {}
        for attempt in range(1 + (max_retries if not greedy else 0)):
            attempt_seed = (seed + attempt) % (2**64)
            started = time.perf_counter()
            result = self._model.generate(
                rows,
                max_frames=max_frames,
                prefill_chunk_size=self._prefill_chunk_size,
                temperature=float(temperature),
                top_k=top_k,
                top_p=float(top_p),
                seed=attempt_seed,
                greedy=greedy,
            )
            if callable(getattr(result, "to_dict", None)):
                result = result.to_dict()
            if not isinstance(result, dict):
                raise RuntimeError("native Drama3 returned a malformed result")  # noqa: TRY004 - preserve API validation errors
            audio, codes = result.get("audio"), result.get("codes")
            rate, reason = result.get("sample_rate"), result.get("finish_reason")
            if (
                not isinstance(audio, np.ndarray)
                or audio.dtype != np.float32
                or audio.ndim != 1
                or not audio.flags.c_contiguous
                or not np.all(np.isfinite(audio))
                or not isinstance(codes, np.ndarray)
                or codes.dtype != np.int32
                or codes.ndim != 2
                or codes.shape[1] != 10
                or not codes.flags.c_contiguous
                or len(codes) > max_frames
                or np.any(codes < 0)
                or rate != 44100
                or reason not in ("eos", "length")
            ):
                raise RuntimeError("native Drama3 returned invalid audio/code metadata")
            if audio.size != codes.shape[0] * 2048:
                raise RuntimeError(
                    "native Drama3 audio length disagrees with codec frame count"
                )
            if not audio.size:
                raise RuntimeError("Incomplete synthesis: model returned no audio")
            timings = result.get("timings", {})
            if not isinstance(timings, dict) or any(
                not isinstance(k, str)
                or not isinstance(v, Real)
                or not math.isfinite(v)
                or v < 0
                for k, v in timings.items()
            ):
                raise RuntimeError("native Drama3 returned invalid timing metadata")
            attempts.append(
                {
                    "seed": attempt_seed,
                    "finish_reason": reason,
                    "frames": len(codes),
                    "seconds": time.perf_counter() - started,
                    "timings": dict(timings),
                }
            )
            for phase, seconds in timings.items():
                aggregate_timings[phase] = aggregate_timings.get(phase, 0.0) + seconds
            if reason != "length" or attempt == max_retries or greedy:
                break
            # Keep only compact metadata from discarded attempts.
            result = audio = codes = None
        return SynthesisResult(
            audio=audio,
            sample_rate=44100,
            codes=codes,
            finish_reason=reason,
            timings=aggregate_timings,
            attempts=tuple(attempts),
        )

    def _synthesize_chunks(
        self,
        chunks,
        max_frames,
        temperature,
        top_k,
        top_p,
        seed,
        greedy,
        max_retries,
        history,
    ):
        initial = list(history)
        completed, audio, codes, attempts, timings = [], [], [], [], {}
        remaining = max_frames
        for index, text in enumerate(chunks):
            if remaining == 0:
                break
            context = (
                initial + completed[:1] + (completed[-1:] if len(completed) > 1 else [])
            )
            result = self.synthesize(
                text,
                remaining,
                temperature,
                top_k,
                top_p,
                seed,
                greedy,
                max_retries=max_retries,
                history=context,
                split_sentences=False,
            )
            audio.append(result.audio)
            codes.append(result.codes)
            remaining -= len(result.codes)
            attempts.extend(
                dict(attempt, chunk_index=index, text=text)
                for attempt in result.attempts
            )
            for phase, seconds in result.timings.items():
                timings[phase] = timings.get(phase, 0.0) + seconds
            if result.finish_reason != "eos":
                break
            completed.append((text, result.codes))
        reason = "eos" if len(completed) == len(chunks) else "length"
        return SynthesisResult(
            audio=np.concatenate(audio),
            sample_rate=44100,
            codes=np.concatenate(codes),
            finish_reason=reason,
            timings=timings,
            attempts=tuple(attempts),
        )
