"""Shared JSON reporting for the CT2 and PyTorch benchmark scripts."""

import copy
import ctypes
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev


def host_hardware():
    processor = platform.processor() or None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("model name", "Hardware")):
                processor = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        memory_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        memory_bytes = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "logical_cpu_count": os.cpu_count(),
        "system_ram_bytes": memory_bytes,
        "system_ram_gib": memory_bytes / 2**30 if memory_bytes is not None else None,
        "python_version": platform.python_version(),
    }


def ct2_gpu_hardware(index):
    """Query the runtime already used by CT2, preserving visible-device indices."""
    try:
        maps = Path("/proc/self/maps").read_text()
        hip_paths = [
            line.split()[-1] for line in maps.splitlines() if "libamdhip64.so" in line
        ]
        if hip_paths:
            runtime = "rocm"
            library = ctypes.CDLL(hip_paths[0])
            name_function = library.hipDeviceGetName
            memory_function = library.hipDeviceTotalMem
            device = index
        else:
            runtime = "cuda"
            library = ctypes.CDLL("libcuda.so.1")
            if library.cuInit(0) != 0:
                raise RuntimeError("CUDA driver initialization failed")
            handle = ctypes.c_int()
            if library.cuDeviceGet(ctypes.byref(handle), index) != 0:
                raise RuntimeError("CUDA device lookup failed")
            device = handle.value
            name_function = library.cuDeviceGetName
            memory_function = library.cuDeviceTotalMem_v2
        name = ctypes.create_string_buffer(256)
        memory = ctypes.c_size_t()
        if name_function(name, len(name), device) != 0:
            raise RuntimeError("GPU name query failed")
        if memory_function(ctypes.byref(memory), device) != 0:
            raise RuntimeError("GPU memory query failed")
        return {
            "name": name.value.decode(),
            "runtime": runtime,
            "total_memory_bytes": memory.value,
            "total_memory_gib": memory.value / 2**30,
        }
    except (OSError, AttributeError, RuntimeError) as error:
        return {
            "name": None,
            "runtime": None,
            "total_memory_bytes": None,
            "total_memory_gib": None,
            "query_error": str(error),
        }


class Results:
    def __init__(self, args, backend, version):
        self.args = args
        self.path = args.output_json
        self.gpu_hardware = None
        self.document = {
            "schema_version": 1,
            "backend": backend,
            "backend_version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "host": host_hardware(),
            "memory_metric": "peak_process_rss_including_load_and_warmup_excluding_gpu",
            "results": [],
        }
        self.save()

    def configuration(self, options=()):
        args = copy.copy(self.args)
        for flag in ("device", "device-index", "compute-type", "memory-profile"):
            option = "--" + flag
            if option in options:
                setattr(
                    args, flag.replace("-", "_"), options[options.index(option) + 1]
                )
        return {
            "model": str(args.model),
            "source": str(args.source) if hasattr(args, "source") else None,
            "device": args.device,
            "device_index": int(args.device_index),
            "hardware": {
                "cpu": {
                    key: self.document["host"][key]
                    for key in (
                        "processor",
                        "logical_cpu_count",
                        "system_ram_bytes",
                        "system_ram_gib",
                    )
                },
                "gpu": self.gpu_hardware if args.device == "cuda" else None,
            },
            "compute_type": args.compute_type,
            "memory_profile": getattr(args, "memory_profile", None) or "default",
            "gpu_expert_cache_gb": (
                args.gpu_expert_cache_gb
                if getattr(args, "memory_profile", None) == "limited"
                else None
            ),
            "threads": args.threads,
            "text": args.text,
            "workload": getattr(args, "workload", "custom"),
            "max_duration_s": getattr(args, "max_duration_s", 24.0),
            "requested_runs": args.runs,
            "warmup_runs": 1,
        }

    def save(self):
        if self.path is None:
            name = platform.node() or self.document["host"]["machine"]
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            date = self.document["created_at"][:10]
            stem = f"{slug}_{self.document['backend']}_{date}"
            self.path = Path(f"{stem}.json")
            suffix = 1
            while self.path.exists():
                self.path = Path(f"{stem}_{suffix:02d}.json")
                suffix += 1
            print(f"JSON report: {self.path}", flush=True)
        self.write(Path(self.path), self.document)

    @staticmethod
    def write(path, document):
        groups = {}
        for entry in document["results"]:
            key = tuple(
                entry[field]
                for field in (
                    "device",
                    "device_index",
                    "compute_type",
                    "memory_profile",
                    "model",
                    "threads",
                )
            )
            groups.setdefault(key, []).append(entry)
        summaries = []
        for entries in groups.values():
            first = entries[0]
            runs = [
                run
                for entry in entries
                if entry["status"] == "success"
                for run in entry.get("runs", [])
            ]
            elapsed = sum(run["time_seconds"] for run in runs)
            duration = sum(run["audio_seconds"] for run in runs)
            peaks = [
                entry["summary"]["peak_rss_gib"]
                for entry in entries
                if "summary" in entry
            ]
            summaries.append(
                {
                    **{
                        field: first[field]
                        for field in (
                            "device",
                            "device_index",
                            "compute_type",
                            "memory_profile",
                            "model",
                            "threads",
                        )
                    },
                    "workloads": [entry.get("workload", "custom") for entry in entries],
                    "status": "success"
                    if all(entry["status"] == "success" for entry in entries)
                    else "incomplete",
                    "total_time_seconds": elapsed,
                    "total_audio_seconds": duration,
                    "rtf": elapsed / duration if duration else None,
                    "peak_rss_gib": max(peaks) if peaks else None,
                }
            )
        document = {**document, "summary": summaries}
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace atomically so interrupted sweeps leave a readable report.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(document, stream, indent=2, allow_nan=False)
                stream.write("\n")
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def record(self, status, **fields):
        entry = {**self.configuration(), "status": status, **fields}
        self.document["results"] = [
            previous
            for previous in self.document["results"]
            if previous.get("workload") != entry["workload"]
        ]
        self.document["results"].append(entry)
        self.save()

    def start(self):
        # A crash/OOM still leaves an entry when this is an explicit single run.
        self.record(
            "failed", error="Configuration did not complete; see console output."
        )

    def skipped(self, reason):
        self.record("skipped", reason=reason)

    def success(self, runs, peak_rss_gib):
        times = [run["time_seconds"] for run in runs]
        durations = [run["audio_seconds"] for run in runs]
        self.record(
            "success",
            runs=runs,
            summary={
                "mean_time_seconds": mean(times),
                "std_time_seconds": stdev(times) if len(times) > 1 else 0,
                "mean_audio_seconds": mean(durations),
                "rtf": sum(times) / sum(durations),
                "peak_rss_gib": peak_rss_gib,
            },
        )

    def run(self, options):
        command = [sys.executable, "-u", *sys.argv, *options]
        # Each child owns a separate file; nested device/precision sweeps merge
        # results without clobbering earlier configurations or prior failures.
        with tempfile.TemporaryDirectory() as directory:
            child_path = Path(directory) / "results.json"
            status = subprocess.run(
                [*command, "--output-json", str(child_path)], check=False
            ).returncode
            if child_path.exists():
                entries = json.loads(child_path.read_text())["results"]
            else:
                entries = []
            if not entries:
                entries = [
                    {
                        **self.configuration(options),
                        "status": "failed",
                        "error": "Worker produced no results; see console output.",
                    }
                ]
            if status:
                for entry in entries:
                    if entry["status"] == "failed":
                        entry.setdefault("exit_code", status)
            self.document["results"].extend(entries)
            self.save()
            return status
