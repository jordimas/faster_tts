"""From the repository root: python samples/generate.py [--device cuda]."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faster_tts import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="sampling candidate limit; 4097 removes top-k filtering for Drama 3",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        choices=(0, 1),
        default=1,
        help="retry frame-limited generation once with seed + 1 (default: 1)",
    )
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    prompts = json.loads((directory / "prompts.json").read_text(encoding="utf-8"))
    model = Model(
        device=args.device,
        compute_type=args.compute_type,
        local_files_only=args.local_files_only,
    )
    for prompt in prompts:
        print(f"Generating {prompt['name']}...", flush=True)
        result = model.synthesize(
            prompt["text"],
            seed=args.seed,
            engine_options={"max_frames": args.max_frames},
            output=directory / f"{prompt['name']}.wav",
            max_retries=args.max_retries,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        if result.finish_reason != "eos":
            attempts = "; ".join(
                f"seed={attempt['seed']}, frames={attempt['frames']}, "
                f"finish={attempt['finish_reason']}"
                for attempt in result.attempts
            )
            raise RuntimeError(
                f"{prompt['name']}: generation reached the {args.max_frames}-frame limit "
                f"({result.audio.size / result.sample_rate:.1f}s of partial audio). "
                f"Attempts: {attempts}. Try a different --seed or simplify the prompt. "
                "The saved WAV contains partial audio."
            )
        print(
            f"Saved {prompt['name']}.wav ({result.audio.size / result.sample_rate:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
