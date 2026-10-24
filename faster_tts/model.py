"""Public loading and synthesis API for registered CTranslate2 TTS engines."""

import json
from dataclasses import dataclass, field

import numpy as np

from .base import _positive_int, get_engine_class
from .cache import DEFAULT_MODEL, resolve_model_path
from .memory import validate_cache_budget


@dataclass(frozen=True)
class SynthesisResult:
    """Generated speech and diagnostics returned by `Model.synthesize`.

    Attributes:
        audio: Contiguous mono float32 NumPy array of shape `(samples,)`.
            Values are finite; no clipping or amplitude normalization is applied here.
        sample_rate: Audio sample rate in Hz (44100 for Drama 3).
        codes: Contiguous int32 codec indices of shape `(frames, codebooks)`.
            Drama 3 has 10 codebooks and 2048 audio samples per frame.
            Codebook count and frame duration depend on the engine.
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


class Model:
    """Load a converted checkpoint using its registered TTS engine.

    The checkpoint's `config.json` selects the engine through `model_type`.
    Only Drama 3 is currently implemented; Qwen checkpoints are not yet supported.
    `info()` reports the selected engine and its supported capabilities.
    Checkpoint validation and synthesis are handled by the selected engine.

    Drama 3 generation is offline, without cloning or reference audio. Seeded
    sampling follows its model distribution, without bitwise PyTorch RNG parity.
    Drama checkpoints require model.bin, a layout-version1 config.json with
    source_config, and frozen tokenizer.json.

    ``Model()`` loads jordimas/drama-3-ct2 from the Hugging Face cache or Hub.
    ``Model("drama-3")`` selects the same checkpoint. A converted Hub repository
    ID (with optional hf:// prefix) or local directory can also be supplied.
    ``device`` selects "cpu" (default), "cuda", or "auto"; ``device_index``
    selects one GPU. CUDA and ROCm builds both use "cuda". For Drama 3, GPU
    acceleration covers backbone/tower projections; waveform decoding stays on CPU.
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
        device: `"cpu"`, `"cuda"`, or `"auto"` (GPU when available). For Drama 3,
            auto CPU execution enables expert memory mapping on Linux/macOS when
            `model.bin` exceeds available RAM after reserving at least 2 GB and
            30% headroom. Requires default/float32 compute and detectable RAM;
            explicit `CT2_DRAMA3_CPU_MMAP` settings take precedence. Device
            availability and memory policies depend on the engine.
        device_index: Nonnegative GPU index. Defaults to 0.
        revision: Optional Hub branch, tag, or commit revision.
        cache_dir: Optional Hugging Face download cache directory.
        local_files_only: Use cached/local assets without downloading.
        gpu_expert_cache_gb: Drama 3's positive total expert-cache budget in decimal GB,
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
        engine_class = get_engine_class(metadata["model_type"])
        self._engine = engine_class(
            directory,
            metadata,
            intra_threads=intra_threads,
            compute_type=compute_type,
            prefill_chunk_size=prefill_chunk_size,
            num_workers=num_workers,
            device=device,
            device_index=device_index,
            gpu_expert_cache_gb=gpu_expert_cache_gb,
        )

    def info(self):
        """Return resolved settings and supported engine capabilities.

        Returns:
            dict: An independent copy with `engine` (checkpoint model type),
                `capabilities` (tuple of supported feature names), and resolved
                device, compute, worker, and checkpoint settings. Additional
                settings depend on the engine. Drama 3 includes `backend`,
                `device`, `codec_device`, `mode`, `gpu_supported`, `device_index`,
                `compute_type`, `cpu_threads` (also `intra_threads`), `num_workers`,
                `prefill_chunk_size`, and resolved `model_path`. Its expert-cache
                budgets are `gpu_expert_cache_gb` (total) and
                `gpu_expert_cache_gb_per_worker`; None delegates to the environment.
                `source_revision`, `quantization_policy`, `matrix_dtype`, and
                `activation_dtype` may be None. Compute type is the requested setting.
                Capability names describe engine features, not runtime availability.
        """
        return self._engine.info()

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
        """Synthesize speech using the selected engine.

        Returns mono float32 audio and int32 codec indices. Sampling and frame
        budgets are interpreted by the engine. The following history, retry,
        sentence-splitting, and speaker-tag behavior describes Drama 3; engines
        without those features must reject unsupported nondefault options.

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
            max_frames: Positive codec-frame budget for the combined returned audio.
                Defaults to 512 (about 23.8 seconds for Drama 3).
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
        return self._engine.synthesize(
            text,
            max_frames=max_frames,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            greedy=greedy,
            max_retries=max_retries,
            history=history,
            split_sentences=split_sentences,
        )
