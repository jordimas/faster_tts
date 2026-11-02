"""Speech synthesis: python -m faster_tts [MODEL] --text TEXT --output out.wav.

MODEL defaults to jordimas/drama-3-ct2. Accepts drama-3, a converted Hub repository
ID (optionally prefixed with hf://), or a local converted directory.
"""

import argparse
import time
from pathlib import Path

from .cache import DEFAULT_MODEL
from .model import Model, _save_wav


def _cache_budget(value):
    if value in ("auto", "uncapped"):
        return value
    try:
        from .memory import validate_cache_budget

        result = float(value)
        validate_cache_budget(result)
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'expected "auto", "uncapped", or a finite positive GB budget'
        ) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help="model name, Hub repository ID, or local directory (default: %(default)s)",
    )
    parser.add_argument("--revision", help="Hub revision; prefer a pinned commit")
    parser.add_argument(
        "--cache-dir", type=Path, help="Hugging Face snapshot cache directory"
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="resolve Hub assets only from the local cache",
    )
    parser.add_argument(
        "--reference-audio",
        action="append",
        type=Path,
        help="reference audio file; repeat for multiple references",
    )
    parser.add_argument(
        "--reference-text",
        action="append",
        help="transcript for each reference audio file, in the same order",
    )
    parser.add_argument(
        "--reference-speaker",
        action="append",
        type=int,
        choices=range(16),
        help="optional speaker ID for each reference, in the same order",
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-duration-s", type=float, default=24.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--compute-type", default="default")
    parser.add_argument(
        "--gpu-expert-cache-gb",
        type=_cache_budget,
        help='GPU expert cache: decimal GB, "auto", or "uncapped"; defaults to environment or auto',
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-split-sentences",
        action="store_true",
        help="diagnostic single-pass generation",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="diagnostic greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--directed-spans",
        action="store_true",
        help="opt-in Drama 3 cue/turn splitting and bounded breathing events",
    )
    args = parser.parse_args()
    audios, texts = args.reference_audio or [], args.reference_text or []
    if len(audios) != len(texts):
        parser.error("each --reference-audio requires one --reference-text")
    if args.reference_speaker is not None and len(args.reference_speaker) != len(
        audios
    ):
        parser.error("provide one --reference-speaker per reference audio")
    references = [
        {"audio": audio, "text": text, "speaker": speaker}
        for audio, text, speaker in zip(
            audios, texts, args.reference_speaker or [None] * len(audios)
        )
    ]
    if args.directed_spans and args.no_split_sentences:
        parser.error("--directed-spans requires sentence splitting")
    engine_options = {}
    if args.gpu_expert_cache_gb is not None:
        engine_options["gpu_expert_cache_gb"] = args.gpu_expert_cache_gb
    model = Model(
        args.model,
        intra_threads=args.threads,
        compute_type=args.compute_type,
        device=args.device,
        device_index=args.device_index,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        engine_options=engine_options or None,
    )
    started = time.perf_counter()
    result = model.synthesize(
        args.text,
        max_duration_s=args.max_duration_s,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        greedy=args.greedy,
        split_sentences=not args.no_split_sentences,
        **({"references": references} if references else {}),
        **({"engine_options": {"directed_spans": True}} if args.directed_spans else {}),
    )
    inference_seconds = time.perf_counter() - started
    _save_wav(result, args.output)
    print(
        f"Saved {result.audio.size / result.sample_rate:.3f}s to {args.output}; "
        f"finish_reason={result.finish_reason}; decoding={'greedy' if args.greedy else 'sampled'}"
    )
    print(f"Inference time (excluding model loading): {inference_seconds:.3f}s")
    if result.finish_reason != "eos":
        raise SystemExit(
            "Incomplete synthesis: frame limit reached; the saved audio may be truncated."
        )


if __name__ == "__main__":
    main()
