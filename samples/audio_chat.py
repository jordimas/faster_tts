"""Type text and hear it spoken in an interactive terminal session."""

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from faster_tts import Model

TAG_HELP = """Drama 3 tags (place before the words they apply to):
  [whispering] Wait... did you hear that?
  [excited] We found it!
  <|speaker:0|> Hello! <|speaker:1|> Hi there!
Speaker IDs: 0–15; default: 0. Bracket cues guide delivery; results may vary.
Type /help to show these examples again, or /quit to exit."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="drama-3")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--max-duration-s", type=float, default=24.0)
    args = parser.parse_args()
    logging.basicConfig(format="%(message)s")
    logging.getLogger("faster_tts.engines.drama3").setLevel(logging.INFO)
    player = next(
        (
            path
            for name in ("aplay", "paplay", "afplay", "ffplay")
            if (path := shutil.which(name))
        ),
        None,
    )
    if player is None:
        parser.error("Playback needs aplay, paplay, afplay, or ffplay on PATH.")
    command = [player]
    if Path(player).name == "ffplay":
        command += ["-nodisp", "-autoexit", "-loglevel", "error"]
    print("Loading model…", flush=True)
    model = Model(args.model, device=args.device, compute_type=args.compute_type)
    print("Type text and press Enter to speak. Type /quit or press Ctrl+C to exit.")
    print(TAG_HELP)
    try:
        with tempfile.TemporaryDirectory(prefix="faster-tts-") as directory:
            audio_path = Path(directory) / "speech.wav"
            while True:
                text = input("> ").strip()
                if text == "/quit":
                    break
                if text == "/help":
                    print(TAG_HELP)
                    continue
                if not text:
                    continue
                print("Generating speech…", flush=True)
                try:
                    result = model.synthesize(
                        text, max_duration_s=args.max_duration_s, output=audio_path
                    )
                    if result.finish_reason == "length":
                        print(
                            "Duration limit reached; audio may be cut off. Try shorter text."
                        )
                    subprocess.run([*command, str(audio_path)], check=True)
                except (
                    ValueError,
                    RuntimeError,
                    OSError,
                    subprocess.CalledProcessError,
                ) as error:
                    print(f"Could not speak: {error}")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
