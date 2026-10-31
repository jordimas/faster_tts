"""Read a UTF-8 article into a WAV: python samples/article_voice.py [article.txt]."""

import argparse
import logging
from pathlib import Path
from time import perf_counter

from faster_tts import Model
from faster_tts.chunking import split_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "text_file",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().with_suffix(".txt"),
        help="UTF-8 plain-text article (default: article_voice.txt beside this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output WAV (default: input path with .wav extension)",
    )
    parser.add_argument("--model", default="drama-3")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--reference-audio", type=Path, help="reference voice recording")
    parser.add_argument("--reference-text", help="transcript of the reference recording")
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=None,
        help="audio budget in seconds (default: 120 seconds per text chunk)",
    )
    args = parser.parse_args()
    if (args.reference_audio is None) != (args.reference_text is None):
        parser.error("provide both --reference-audio and --reference-text")
    logging.basicConfig(format="%(message)s")
    logging.getLogger("faster_tts.engines.drama3").setLevel(logging.INFO)
    if args.output is None:
        args.output = args.text_file.with_suffix(".wav")
    try:
        text = args.text_file.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        parser.error(str(error))
    if not text.strip():
        parser.error("the article is empty")
    if args.text_file.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    if args.max_duration_s is None:
        args.max_duration_s = 120.0 * len(split_text(text))
    print("Loading model…", flush=True)
    model = Model(args.model, device=args.device, compute_type=args.compute_type)
    # info() reports the requested compute type; the native model resolves auto.
    print(
        f"Device: {model.info()['device']}, "
        f"compute_type: {model._engine._model.compute_type}",
        flush=True,
    )
    print("Reading article…", flush=True)
    start = perf_counter()
    result = model.synthesize(
        text,
        max_duration_s=args.max_duration_s,
        output=args.output,
        **(
            {"references": [{"audio": args.reference_audio, "text": args.reference_text}]}
            if args.reference_audio is not None
            else {}
        ),
    )
    elapsed = perf_counter() - start
    audio_duration = result.audio.size / result.sample_rate
    print(f"Total inference time: {elapsed:.2f}s")
    print(f"Saved {audio_duration:.1f}s to {args.output}")
    print(f"RTF: {elapsed / audio_duration:.3f} (lower is faster)")
    if result.finish_reason != "eos":
        raise SystemExit(
            f"Duration limit reached. {args.output} contains partial narration; "
            "increase --max-duration-s."
        )


if __name__ == "__main__":
    main()
