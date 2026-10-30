"""Shared command-line options and execution for the benchmark runners."""

import argparse
import resource
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from time import perf_counter

TEXTS = {
    "short": "Hello! Welcome to our application.",
    "medium": (
        "The town library opened its doors early on Saturday morning. "
        "Outside, a gentle rain fell on the quiet streets, while inside, "
        "volunteers arranged books and prepared a reading corner for children. "
        "Maya arrived with her younger brother and chose a story about a journey "
        "across the mountains. They settled into a comfortable chair near the "
        "open window and began to read together. By the time the sun appeared, "
        "they had discovered a new favorite book and made plans to return."
    ),
}


def benchmark_parser(description):
    """Options shared by the CT2 and official PyTorch runners."""
    parser = argparse.ArgumentParser(description=description)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--text", help="custom input instead of the built-in workloads")
    inputs.add_argument("--workload", choices=("short", "medium", "all"), default="all")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        help="default: benchmark CPU and every available GPU",
    )
    parser.add_argument(
        "--device-index", type=int, default=0, help="GPU index for an explicit --device"
    )
    parser.add_argument(
        "--show-progress", action="store_true", help="show model download progress bars"
    )
    parser.add_argument(
        "--check-quality",
        action="store_true",
        help="transcribe every timed output with faster-whisper medium and report WER",
    )
    parser.add_argument(
        "--check", choices=("quality",), help="alias for --check-quality"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="JSON output (default: <hardware>_<backend>_<date>.json per device)",
    )
    for flag in ("--compute-worker", "--log-worker"):
        parser.add_argument(flag, action="store_true", help=argparse.SUPPRESS)
    return parser


def workloads(args):
    """Return fixed inputs and audio budgets, or a single custom input."""
    if args.text is not None:
        return [("custom", args.text, 24.0)]
    return [
        (name, text, 24.0 if name == "short" else 90.0)
        for name, text in TEXTS.items()
        if args.workload in (name, "all")
    ]


def run_workloads(results, synthesize, checker, synchronize=lambda: None):
    """Warm and measure each input in one process, then report shared peak RAM."""
    args = results.args
    selected = workloads(args)
    # Replace the model-loading failure placeholder with per-workload entries.
    results.document["results"] = []
    with tempfile.TemporaryDirectory() as directory:
        for name, text, budget in selected:
            args.workload, args.text, args.max_duration_s = name, text, budget
            print(f"\nWorkload: {name} ({len(text.split())} words)", flush=True)
            results.start()
            synthesize()
            synchronize()
            audio_dir = Path(directory) / name
            audio_dir.mkdir()
            measure(results, synthesize, audio_dir, synchronize, report_peak=False)
        peak = peak_rss_gib()
        for entry in results.document["results"]:
            if "summary" in entry:
                entry["summary"]["peak_rss_gib"] = peak
        results.save()
        print(f"Peak RSS across workloads: {peak:.3f} GiB", flush=True)
        if args.check_quality:
            for name, text, budget in selected:
                args.workload, args.text, args.max_duration_s = name, text, budget
                checker(
                    args,
                    Path(directory) / name,
                    f"{args.device}:{args.device_index} {args.compute_type} {name}",
                )


def peak_rss_gib():
    """Process lifetime high-water mark, including loading and warmup."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**3 if sys.platform == "darwin" else 1024**2)


def run_logged(args, backend):
    """Run the sweep once under a console/log tee; children inherit the marker."""
    if args.log_worker:
        return
    with (
        open(f"{backend}.log", "wb") as log,
        subprocess.Popen(
            [sys.executable, "-u", *sys.argv, "--log-worker"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as worker,
    ):
        for chunk in iter(worker.stdout.read1, b""):
            log.write(chunk)
            log.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        status = worker.wait()
    raise SystemExit(status)


def measure(results, synthesize, audio_dir, synchronize=lambda: None, report_peak=True):
    """Time warmed-up synthesis; record peak RSS before optional child ASR."""
    runs = []
    for i in range(results.args.runs):
        synchronize()
        start = perf_counter()
        audio, sample_rate, finish = synthesize()
        synchronize()
        elapsed = perf_counter() - start
        duration = audio.size / sample_rate
        if results.args.runs > 1:
            print(
                f"Run {i + 1}: {elapsed:.3f}s, audio={duration:.3f}s, "
                f"RTF={elapsed / duration:.3f}, finish_reason={finish}"
            )
        if finish != "eos":
            raise SystemExit("Frame limit reached; shorten the input.")
        runs.append(
            {
                "time_seconds": elapsed,
                "audio_seconds": duration,
                "rtf": elapsed / duration,
                "finish_reason": finish,
            }
        )
        if results.args.check_quality:
            import numpy as np

            with wave.open(str(Path(audio_dir) / f"run-{i + 1}.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        del audio
    peak_gib = peak_rss_gib()
    results.success(runs, peak_gib)
    summary = results.document["results"][-1]["summary"]
    print(
        f"Mean: {summary['mean_time_seconds']:.3f}s; "
        f"std: {summary['std_time_seconds']:.3f}s; "
        f"RTF: {summary['rtf']:.3f}"
        + (f"; peak RSS: {peak_gib:.3f} GiB" if report_peak else ""),
        flush=True,
    )


def sweep(results, configurations):
    failed = False
    for options in configurations:
        status = results.run(options)
        if status:
            print(f"{' '.join(options)} failed (exit {status})", flush=True)
            failed = True
    raise SystemExit(int(failed))
