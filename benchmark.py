"""Run with: python benchmark.py [model] [--device cuda]."""

import argparse
from time import perf_counter

from faster_tts import Model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", nargs="?", default="drama-3")
parser.add_argument("--text", default="Hello! Welcome to our application.")
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--device", default="cpu")
parser.add_argument("--compute-type", default="default")
args = parser.parse_args()
if args.runs < 1:
    parser.error("--runs must be positive")

model = Model(args.model, cpu_threads=args.threads,
              device=args.device, compute_type=args.compute_type)
model.synthesize(args.text)  # Warm up; exclude loading and first-run costs.
total_time = total_audio = 0.0
for i in range(args.runs):
    start = perf_counter()
    result = model.synthesize(args.text)
    elapsed = perf_counter() - start
    duration = result.audio.size / result.sample_rate
    print(f"Run {i + 1}: {elapsed:.3f}s, audio={duration:.3f}s, "
          f"RTF={elapsed / duration:.3f}, finish_reason={result.finish_reason}")
    if result.finish_reason != "eos":
        raise SystemExit("Frame limit reached; shorten --text.")
    total_time += elapsed
    total_audio += duration
print(f"Mean: {total_time / args.runs:.3f}s; RTF: {total_time / total_audio:.3f}")
