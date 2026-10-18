"""Transcribe benchmark WAV files in a separate process, outside performance measurements."""

import argparse
import unicodedata


def normalized_words(text):
    """Ignore case and punctuation; preserve accents and internal apostrophes."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    text = "".join(
        char if char == "'" or not unicodedata.category(char).startswith("P") else " "
        for char in text
    )
    return [word.strip("'") for word in text.split() if word.strip("'")]


def word_errors(reference, hypothesis):
    """Word-level Levenshtein distance (substitutions + deletions + insertions)."""
    previous = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, predicted in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != predicted)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--label", help="benchmark device, compute type, and memory profile")
    parser.add_argument("audio", nargs="+")
    args = parser.parse_args()
    reference = normalized_words(args.text)
    if not reference:
        parser.error("WER requires at least one reference word")

    from huggingface_hub.utils import disable_progress_bars
    from faster_whisper import WhisperModel

    if not args.show_progress:
        disable_progress_bars()
    model = WhisperModel("medium", device="cpu", compute_type="int8",
                         cpu_threads=args.threads)
    errors = 0
    for path in args.audio:
        segments, _ = model.transcribe(path, beam_size=5, temperature=0,
                                       condition_on_previous_text=False)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        count = word_errors(reference, normalized_words(transcript))
        errors += count
    words = len(reference) * len(args.audio)
    label = f" [{args.label}]" if args.label else ""
    print(f"WER: {errors / words:.2%} ({errors}/{words}){label}", flush=True)


if __name__ == "__main__":
    main()
