"""Run from the repository root: uv run python benchmark/benchmark.py [model]."""

import argparse
import math
import os
import resource
import subprocess
import sys
from statistics import mean, stdev
from time import perf_counter

import ctranslate2
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from faster_tts import Model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", nargs="?", default="drama-3")
parser.add_argument("--text", default="Hello! Welcome to our application.")
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--device", choices=("cpu", "cuda", "auto"),
                    help="default: benchmark CPU and every available GPU")
parser.add_argument("--device-index", type=int, default=0, help="GPU index for an explicit --device")
parser.add_argument("--compute-type", help="default: benchmark all supported types")
parser.add_argument("--compute-worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--show-progress", action="store_true", help="show model download progress bars")
parser.add_argument("--memory-profile", choices=("default", "limited"),
                    help="default: CPU defaults; GPU uncapped and expert-cache-limited configurations")
parser.add_argument("--gpu-expert-cache-gb", type=float, default=4,
                    help="limited profile: GPU expert cache in decimal GB (default: 4)")
args = parser.parse_args()
if args.runs < 1:
    parser.error("--runs must be positive")
if args.device_index < 0:
    parser.error("--device-index must be nonnegative")

if not math.isfinite(args.gpu_expert_cache_gb) or args.gpu_expert_cache_gb <= 0:
    parser.error("memory budgets must be finite and positive")

if args.show_progress:
    enable_progress_bars()
else:
    disable_progress_bars()

if args.device is None:
    devices = [("cpu", 0)] + [("cuda", i) for i in range(ctranslate2.get_cuda_device_count())]
    failed = []
    for device, device_index in devices:
        label = "cpu" if device == "cpu" else f"cuda:{device_index}"
        status = subprocess.run([sys.executable, "-u", *sys.argv,
                                 "--device", device,
                                 "--device-index", str(device_index)]).returncode
        if status:
            print(f"{label} failed (exit {status})", flush=True)
            failed.append(label)
    raise SystemExit(1 if failed else 0)

if args.device == "auto":
    args.device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
if args.memory_profile is None:
    failed = []
    profiles = ("default",) if args.device == "cpu" else ("default", "limited")
    for profile in profiles:
        status = subprocess.run([sys.executable, "-u", *sys.argv,
                                 "--memory-profile", profile]).returncode
        if status:
            print(f"{args.device}:{args.device_index} / {profile} failed (exit {status})", flush=True)
            failed.append(profile)
    raise SystemExit(1 if failed else 0)

# GPU cache comparisons override inherited expert-cache settings.
os.environ.pop("CT2_DRAMA3_GPU_EXPERT_CACHE_GB", None)
if args.memory_profile == "limited":
    if args.device == "cpu":
        print("SKIP limited CPU profile: CPU uses mapped weights and allocation controls, "
              "without a total RAM cap.", flush=True)
        raise SystemExit(0)
    os.environ["CT2_DRAMA3_GPU_EXPERT_CACHE_GB"] = f"{args.gpu_expert_cache_gb:g}"
memory_label = ("uncapped" if args.memory_profile == "default"
                else f"GPU expert cache={args.gpu_expert_cache_gb:g} GB")

if args.device == "cuda":
    # Half-precision expert weights persist across synthesis calls. The CUB
    # default rounds allocations to powers of four, inflating their footprint.
    # Use finer bins before the first allocation; keep explicit user settings.
    os.environ.setdefault("CT2_CUDA_CACHING_ALLOCATOR_CONFIG", "2,6,24,209715200")
compute_types = [args.compute_type] if args.compute_type else sorted(
    ctranslate2.get_supported_compute_types(args.device, args.device_index))
if not args.compute_worker:
    print(f"\nDevice: {args.device}:{args.device_index}; "
          f"Memory profile: {args.memory_profile}; {memory_label}", flush=True)
if args.compute_type is None:
    failed = []
    for compute_type in compute_types:
        # Process exit releases native allocator caches as well as model weights.
        status = subprocess.run([sys.executable, "-u", *sys.argv,
                                 "--compute-type", compute_type, "--compute-worker"]).returncode
        if status:
            print(f"{compute_type} failed (exit {status})", flush=True)
            failed.append(compute_type)
    raise SystemExit(1 if failed else 0)
for compute_type in compute_types:
    model = Model(args.model, cpu_threads=args.threads,
                  device=args.device, device_index=args.device_index, compute_type=compute_type)
    print(f"\nCompute type: {compute_type}", flush=True)
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
          f"RTF: {total_time / total_audio:.3f}; peak RSS: {peak_gib:.3f} GiB")
    del model
