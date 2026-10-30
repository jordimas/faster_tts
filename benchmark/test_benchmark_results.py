"""Exercise report aggregation with real subprocesses, without loading models."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmark_common import TEXTS, benchmark_parser, measure, run_workloads, workloads
from benchmark_results import Results

WORKER = """
import argparse
from pathlib import Path
from benchmark_results import Results
p = argparse.ArgumentParser()
p.add_argument('--output-json', type=Path)
p.add_argument('--device')
p.add_argument('--compute-type')
p.add_argument('--backend', default='test')
a = p.parse_args()
a.device_index = 0
a.model = 'test-model'
a.threads = 8
a.text = 'Hello'
a.runs = 2
r = Results(a, a.backend, '1.0')
if a.device is None:
    statuses = [r.run(['--device', d]) for d in ['cpu', 'cuda']]
    raise SystemExit(int(any(statuses)))
if a.compute_type is None:
    statuses = [r.run(['--compute-type', t]) for t in ['float32', 'int8']]
    raise SystemExit(int(any(statuses)))
if a.device == 'cuda':
    r.gpu_hardware = {'name': 'Test GPU', 'runtime': 'rocm', 'total_memory_bytes': 1024}
r.start()
if a.device == 'cuda' and a.compute_type == 'float32':
    raise SystemExit(9)
r.success([
    {'time_seconds': 1.0, 'audio_seconds': 2.0, 'rtf': 0.5, 'finish_reason': 'eos'},
    {'time_seconds': 3.0, 'audio_seconds': 2.0, 'rtf': 1.5, 'finish_reason': 'eos'},
], 4.5)
"""


class ResultsTest(unittest.TestCase):
    def test_workload_selection(self):
        parser = benchmark_parser("test")
        self.assertEqual(parser.parse_args([]).runs, 2)
        self.assertEqual(
            [name for name, _, _ in workloads(parser.parse_args([]))],
            ["short", "medium"],
        )
        for name in TEXTS:
            selected = workloads(parser.parse_args(["--workload", name]))
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0][:2], (name, TEXTS[name]))
        self.assertEqual(
            workloads(parser.parse_args(["--text", "My custom input."])),
            [("custom", "My custom input.", 24.0)],
        )

    def test_workloads_keep_separate_timings_and_shared_peak(self):
        args = benchmark_parser("test").parse_args(["--runs", "2"])
        args.model, args.device, args.compute_type = "test-model", "cpu", "float32"
        observed = []

        def synthesize():
            observed.append((args.workload, args.text, args.max_duration_s))
            return SimpleNamespace(size=200), 100, "eos"

        with tempfile.TemporaryDirectory() as directory:
            args.output_json = Path(directory) / "results.json"
            results = Results(args, "ct2", "test")
            results.start()
            with (
                patch(
                    "benchmark_common.perf_counter",
                    side_effect=[0, 1, 2, 3, 4, 7, 8, 11],
                ),
                patch("benchmark_common.peak_rss_gib", side_effect=[4, 6, 6]),
            ):
                run_workloads(results, synthesize, Mock())
            report = json.loads(args.output_json.read_text())
            entries = report["results"]
        self.assertEqual(len(report["summary"]), 1)
        summary = report["summary"][0]
        self.assertEqual(summary["total_time_seconds"], 8)
        self.assertEqual(summary["total_audio_seconds"], 8)
        self.assertEqual(summary["rtf"], 1)
        self.assertEqual(summary["peak_rss_gib"], 6)
        self.assertEqual(summary["workloads"], ["short", "medium"])
        self.assertEqual([e["workload"] for e in entries], ["short", "medium"])
        self.assertEqual([e["summary"]["mean_time_seconds"] for e in entries], [1, 3])
        self.assertEqual([e["summary"]["peak_rss_gib"] for e in entries], [6, 6])
        self.assertEqual([e["max_duration_s"] for e in entries], [24, 90])
        self.assertEqual(
            [name for name, _, _ in observed], ["short"] * 3 + ["medium"] * 3
        )
        self.assertTrue(all(text == TEXTS[name] for name, text, _ in observed))

    def run_worker(self, *options):
        with tempfile.TemporaryDirectory() as directory:
            # Put the worker beside the reporting module via an explicit path.
            worker = Path(directory) / "worker.py"
            worker.write_text(
                "import sys\nsys.path.insert(0, "
                + repr(str(Path(__file__).resolve().parent))
                + ")\n"
                + WORKER
            )
            output = Path(directory) / "nested" / "report.json"
            status = subprocess.run(
                [sys.executable, str(worker), *options, "--output-json", str(output)],
                check=False,
            ).returncode
            return status, json.loads(output.read_text())

    def test_nested_sweep_preserves_successes_and_failure(self):
        status, report = self.run_worker()
        self.assertEqual(len(report["summary"]), 4)
        for summary in report["summary"]:
            failed = (
                summary["device"] == "cuda" and summary["compute_type"] == "float32"
            )
            self.assertEqual(summary["status"], "incomplete" if failed else "success")
            self.assertEqual(summary["total_time_seconds"], 0 if failed else 4)
            self.assertEqual(summary["rtf"], None if failed else 1)
        self.assertEqual(status, 1)
        self.assertEqual(report["schema_version"], 1)
        entries = report["results"]
        self.assertEqual(len(entries), 4)
        self.assertEqual(
            [(e["device"], e["compute_type"]) for e in entries],
            [
                ("cpu", "float32"),
                ("cpu", "int8"),
                ("cuda", "float32"),
                ("cuda", "int8"),
            ],
        )
        self.assertEqual(
            [e["status"] for e in entries], ["success", "success", "failed", "success"]
        )
        self.assertEqual(entries[2]["exit_code"], 9)
        self.assertIn("processor", report["host"])
        self.assertIn("system_ram_gib", report["host"])
        self.assertEqual(
            entries[0]["hardware"]["cpu"]["processor"], report["host"]["processor"]
        )
        self.assertIsNone(entries[0]["hardware"]["gpu"])
        self.assertEqual(entries[2]["hardware"]["gpu"]["name"], "Test GPU")
        self.assertEqual(entries[3]["hardware"]["gpu"]["runtime"], "rocm")
        summary = entries[0]["summary"]
        self.assertEqual(summary["mean_time_seconds"], 2)
        self.assertAlmostEqual(summary["std_time_seconds"], 2**0.5)
        self.assertEqual(summary["rtf"], 1)
        self.assertEqual(summary["peak_rss_gib"], 4.5)
        self.assertEqual(len(entries[0]["runs"]), 2)

    def test_default_reports_group_precisions_and_preserve_previous_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "worker.py"
            worker.write_text(
                "import sys\nsys.path.insert(0, "
                + repr(str(Path(__file__).resolve().parent))
                + ")\n"
                + WORKER
            )
            for _ in range(2):
                completed = subprocess.run(
                    [sys.executable, str(worker), "--backend", "hf"],
                    cwd=directory,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 1, completed.stderr)
            paths = sorted(Path(directory).glob("*.json"))
            self.assertEqual(len(paths), 4)
            self.assertEqual(len(list(Path(directory).glob("*_01.json"))), 2)
            self.assertTrue(any(p.name.startswith("test-gpu_hf_") for p in paths))
            for path in paths:
                report = json.loads(path.read_text())
                self.assertEqual(report["backend"], "hf")
                self.assertIn("_hf_", path.name)
                entries = report["results"]
                self.assertEqual(len(entries), 2)
                self.assertEqual(
                    [e["compute_type"] for e in entries], ["float32", "int8"]
                )
                self.assertEqual(len({e["device"] for e in entries}), 1)
                self.assertIn(report["created_at"][:10], path.name)

    def test_logging_overwrites_and_captures_worker_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "logged.py"
            worker.write_text(
                "import sys, argparse\n"
                + "sys.path.insert(0, "
                + repr(str(Path(__file__).resolve().parent))
                + ")\n"
                + "from benchmark_common import run_logged\n"
                + "p = argparse.ArgumentParser()\n"
                + 'p.add_argument("--log-worker", action="store_true")\n'
                + 'run_logged(p.parse_args(), "test")\n'
                + 'print("measurement", flush=True)\n'
                + 'print("worker error", file=sys.stderr, flush=True)\n'
                + "raise SystemExit(9)\n"
            )
            log = Path(directory) / "test.log"
            log.write_text("old log contents")
            completed = subprocess.run(
                [sys.executable, str(worker)],
                cwd=directory,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 9)
            self.assertEqual(log.read_bytes(), completed.stdout)
            self.assertIn(b"measurement", completed.stdout)
            self.assertIn(b"worker error", completed.stdout)
            self.assertNotIn(b"old log contents", completed.stdout)

    def test_measure_preserves_timings_sync_and_frame_limit_failure(self):
        args = benchmark_parser("test").parse_args([])
        args.model, args.device, args.compute_type = "test-model", "cpu", "float32"
        args.runs = 2
        with tempfile.TemporaryDirectory() as directory:
            args.output_json = Path(directory) / "results.json"
            results = Results(args, "hf", "1.0")
            for finish in ("eos", "length"):
                results.start()
                synthesize = Mock(return_value=(SimpleNamespace(size=200), 100, finish))
                synchronize = Mock()
                with patch("benchmark_common.perf_counter", side_effect=[0, 1, 2, 5]):
                    if finish == "length":
                        with self.assertRaisesRegex(SystemExit, "Frame limit"):
                            measure(results, synthesize, directory, synchronize)
                        self.assertEqual(
                            results.document["results"][0]["status"], "failed"
                        )
                        continue
                    measure(results, synthesize, directory, synchronize)
                self.assertEqual(synchronize.call_count, 4)
                summary = results.document["results"][0]["summary"]
                self.assertEqual(summary["mean_time_seconds"], 2)
                self.assertEqual(summary["mean_audio_seconds"], 2)
                self.assertEqual(summary["rtf"], 1)

    def test_direct_failed_configuration_leaves_readable_report(self):
        status, report = self.run_worker(
            "--device", "cuda", "--compute-type", "float32"
        )
        self.assertEqual(status, 9)
        self.assertEqual(report["results"][0]["status"], "failed")
        self.assertNotIn("summary", report["results"][0])


if __name__ == "__main__":
    unittest.main()
