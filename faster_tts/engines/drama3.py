"""Drama 3 checkpoint validation, native loading, and synthesis."""

import json
import math
import time
from importlib import import_module
from numbers import Real

import numpy as np

from ..base import BaseModel, _positive_int, register_engine
from ..chunking import split_text
from ..memory import auto_cpu_mmap, expert_cache_budget, validate_cache_budget
from ..model import SynthesisResult
from ..prompt import Drama3PromptBuilder


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
class Drama3Model(BaseModel):
    """Internal Drama 3 implementation."""

    capabilities = (
        "synthesis",
        "history",
        "sentence_splitting",
        "retries",
        "gpu_expert_cache",
        "cpu_expert_mmap",
    )

    @staticmethod
    def validate_checkpoint(directory, config):
        """Validate Drama-specific configuration and tokenizer assets."""
        path = directory / "tokenizer.json"
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing or empty converted Drama 3 asset: {path}")
        source = config.get("source_config") if isinstance(config, dict) else None
        if (
            not isinstance(config, dict)
            or config.get("model_type") != "drama3"
            or type(config.get("ct2_layout_version")) is not int
            or config["ct2_layout_version"] != 1
            or not isinstance(source, dict)
            or not isinstance(source.get("text_config"), dict)
            or not source["text_config"]
            or not isinstance(source.get("audio_decoder_config"), dict)
            or not source["audio_decoder_config"]
        ):
            raise ValueError(
                f"Unsupported converted Drama 3 config in {directory}: expected "
                "model_type='drama3', ct2_layout_version=1 and source_config with "
                "text_config/audio_decoder_config. Convert the checkpoint with the matching converter."
            )
        try:
            tokenizer = json.loads(
                (directory / "tokenizer.json").read_text(encoding="utf-8")
            )
            if not isinstance(tokenizer, dict) or not isinstance(
                tokenizer.get("model"), dict
            ):
                raise ValueError("tokenizer model object missing")  # noqa: TRY004 - invalid JSON content
        except (ValueError, UnicodeError) as exc:
            raise ValueError(
                f"Invalid tokenizer JSON: {directory / 'tokenizer.json'}"
            ) from exc

    def __init__(
        self,
        directory,
        metadata,
        *,
        intra_threads,
        compute_type,
        prefill_chunk_size,
        num_workers,
        device,
        device_index,
        gpu_expert_cache_gb,
    ):
        validate_cache_budget(gpu_expert_cache_gb)
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
        with (
            expert_cache_budget(
                gpu_expert_cache_gb, resolved_device, device_index, num_workers
            ) as cache_gb,
            auto_cpu_mmap(
                directory, device == "auto" and resolved_device == "cpu", compute_type
            ),
        ):
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
        """Run Drama 3 generation, validation, retries, and sentence splitting."""
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
