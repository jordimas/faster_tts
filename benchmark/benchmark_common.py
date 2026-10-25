"""Shared command-line options and execution for the benchmark runners."""

import argparse
import resource
import subprocess
import sys
import wave
from pathlib import Path
from time import perf_counter


def benchmark_parser(description):
    """Options shared by the CT2 and official PyTorch runners."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--text", default="Hello! Welcome to our application.")
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


def measure(results, synthesize, audio_dir, synchronize=lambda: None):
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
            raise SystemExit("Frame limit reached; shorten --text.")
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
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gib = peak_rss / (1024**3 if sys.platform == "darwin" else 1024**2)
    results.success(runs, peak_gib)
    summary = results.document["results"][0]["summary"]
    print(
        f"Mean: {summary['mean_time_seconds']:.3f}s; "
        f"std: {summary['std_time_seconds']:.3f}s; "
        f"RTF: {summary['rtf']:.3f}; peak RSS: {peak_gib:.3f} GiB",
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
