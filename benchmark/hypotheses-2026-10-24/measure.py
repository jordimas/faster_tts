"""Measure loading, one warmup, one timed synthesis, and process peak RSS."""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import ctranslate2

from faster_tts import Model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def rss():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20


def signature(result):
    return {
        "audio_sha256": hashlib.sha256(result.audio.tobytes()).hexdigest(),
        "codes_sha256": hashlib.sha256(result.codes.tobytes()).hexdigest(),
        "audio_seconds": result.audio.size / result.sample_rate,
        "finish_reason": result.finish_reason,
    }


report = {
    "label": args.label,
    "model": args.model,
    "threads": 8,
    "warmup_runs": 1,
    "timed_runs": 1,
    "ct2_version": ctranslate2.__version__,
}
started = time.perf_counter()
model = Model(args.model, device="cpu", compute_type="int8", cpu_threads=8)
report["load_seconds"] = time.perf_counter() - started
report["load_peak_rss_gib"] = peak_rss()
report["loaded_rss_gib"] = rss()
print(json.dumps({"phase": "loaded", **report}), flush=True)
began = time.perf_counter()
warmup = model.synthesize("Hello! Welcome to our application.")
report["warmup_seconds"] = time.perf_counter() - began
report["warmup_output"] = signature(warmup)
del warmup
print(json.dumps({"phase": "warmup", **report}), flush=True)
began = time.perf_counter()
result = model.synthesize("Hello! Welcome to our application.")
report["synthesis_seconds"] = time.perf_counter() - began
report["total_seconds"] = time.perf_counter() - started
report["peak_rss_gib"] = peak_rss()
report["final_rss_gib"] = rss()
report["output"] = signature(result)
report["status"] = "success" if result.finish_reason == "eos" else "incomplete"
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
