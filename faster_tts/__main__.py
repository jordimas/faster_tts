"""Speech synthesis: python -m faster_tts [MODEL] --text TEXT --output out.wav.

MODEL defaults to jordimas/drama-3-ct2. Accepts drama-3, a converted Hub repository
ID (optionally prefixed with hf://), or a local converted directory.
"""

import argparse
from pathlib import Path
import time
import wave

import numpy as np

from .model import Model
from .cache import DEFAULT_MODEL


def _cache_budget(value):
    if value == "auto":
        return value
    try:
        from .memory import validate_cache_budget
        result = float(value)
        validate_cache_budget(result)
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected "auto" or a finite positive GB budget') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                        help="model name, Hub repository ID, or local directory (default: %(default)s)")
    parser.add_argument("--revision", help="Hub revision; prefer a pinned commit")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face snapshot cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="resolve Hub assets only from the local cache")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--gpu-expert-cache-gb", type=_cache_budget,
                        help='GPU expert cache budget across all workers in decimal GB, or "auto"; not total VRAM')
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-p", type=float, default=.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-split-sentences", action="store_true", help="diagnostic single-pass generation")
    parser.add_argument("--greedy", action="store_true", help="diagnostic greedy decoding instead of sampling")
    args = parser.parse_args()
    model = Model(args.model, intra_threads=args.threads, compute_type=args.compute_type,
                   device=args.device, device_index=args.device_index,
                   revision=args.revision, cache_dir=args.cache_dir,
                   local_files_only=args.local_files_only,
                   gpu_expert_cache_gb=args.gpu_expert_cache_gb)
    started = time.perf_counter()
    result = model.synthesize(
        args.text, max_frames=args.max_frames, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, seed=args.seed, greedy=args.greedy,
        split_sentences=not args.no_split_sentences)
    inference_seconds = time.perf_counter() - started
    pcm = np.rint(np.clip(result.audio, -1, 1) * 32767).astype("<i2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(result.sample_rate)
        output.writeframes(pcm.tobytes())
    print(f"Saved {result.audio.size / result.sample_rate:.3f}s to {args.output}; "
          f"finish_reason={result.finish_reason}; decoding={'greedy' if args.greedy else 'sampled'}")
    print(f"Inference time (excluding model loading): {inference_seconds:.3f}s")
    if result.finish_reason != "eos":
        raise SystemExit("Incomplete synthesis: frame limit reached; the saved audio may be truncated.")


if __name__ == "__main__":
    main()
