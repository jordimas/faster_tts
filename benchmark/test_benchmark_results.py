"""Exercise report aggregation with real subprocesses, without loading models."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

WORKER = """
import argparse
from pathlib import Path
from benchmark_results import Results
p = argparse.ArgumentParser()
p.add_argument('--output-json', type=Path)
p.add_argument('--device')
p.add_argument('--compute-type')
a = p.parse_args()
a.device_index = 0
a.model = 'test-model'
a.threads = 8
a.text = 'Hello'
a.runs = 2
r = Results(a, 'test', '1.0')
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

    def test_logging_overwrites_and_captures_worker_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "logged.py"
            worker.write_text(
                "import sys, argparse\n"
                + "sys.path.insert(0, "
                + repr(str(Path(__file__).resolve().parent))
                + ")\n"
                + "from benchmark_results import run_logged\n"
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

    def test_direct_failed_configuration_leaves_readable_report(self):
        status, report = self.run_worker(
            "--device", "cuda", "--compute-type", "float32"
        )
        self.assertEqual(status, 9)
        self.assertEqual(report["results"][0]["status"], "failed")
        self.assertNotIn("summary", report["results"][0])


if __name__ == "__main__":
    unittest.main()
