"""Run from the repository root: uv run python benchmark/benchmark.py [model]."""

import argparse
import resource
import subprocess
import sys
from statistics import mean, stdev
from time import perf_counter

import ctranslate2
from faster_tts import Model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", nargs="?", default="drama-3")
parser.add_argument("--text", default="Hello! Welcome to our application.")
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
parser.add_argument("--compute-type", help="default: benchmark all supported types")
args = parser.parse_args()
if args.runs < 1:
    parser.error("--runs must be positive")

if args.device == "auto":
    args.device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
compute_types = [args.compute_type] if args.compute_type else sorted(
    ctranslate2.get_supported_compute_types(args.device))
if args.compute_type is None:
    failed = []
    for compute_type in compute_types:
        # Process exit releases native allocator caches as well as model weights.
        status = subprocess.run([sys.executable, "-u", *sys.argv,
                                 "--compute-type", compute_type]).returncode
        if status:
            print(f"{compute_type} failed (exit {status})", flush=True)
            failed.append(compute_type)
    raise SystemExit(1 if failed else 0)
for compute_type in compute_types:
    model = Model(args.model, cpu_threads=args.threads,
                  device=args.device, compute_type=compute_type)
    print(f"\nDevice: {args.device}; compute type: {compute_type}", flush=True)
    model.synthesize(args.text)  # Warm up; exclude loading and first-run costs.
    total_time = total_audio = 0.0
    times = []
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
        times.append(elapsed)
        total_audio += duration
        del result
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gib = peak_rss / (1024**3 if sys.platform == "darwin" else 1024**2)
    print(f"Mean: {mean(times):.3f}s; std: {stdev(times) if len(times) > 1 else 0:.3f}s; "
          f"RTF: {total_time / total_audio:.3f}; device: {args.device}; "
          f"peak RSS: {peak_gib:.3f} GiB")
    del model
