"""Check public reporting keeps incomplete measurements visible."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from json_to_markdown import render


def report_fixture():
    result = {
        "model": "drama-3",
        "device": "cpu",
        "hardware": {"cpu": {"processor": "test-cpu", "system_ram_gib": 24}},
        "compute_type": "float32",
        "memory_profile": "default",
        "threads": 8,
        "text": "Hello!",
        "requested_runs": 1,
        "warmup_runs": 1,
        "status": "success",
        "runs": [{"finish_reason": "eos"}],
        "summary": {
            "mean_time_seconds": 284.7,
            "std_time_seconds": 0,
            "mean_audio_seconds": 2.5,
            "rtf": 113.5,
            "peak_rss_gib": 11.4,
        },
    }
    return {
        "schema_version": 1,
        "backend": "ct2",
        "backend_version": "test",
        "created_at": "2026-10-25T00:00:00+00:00",
        "results": [copy.deepcopy(result) for _ in range(3)],
    }


class MarkdownTest(unittest.TestCase):
    def test_workload_reports_keep_timings_and_shared_peak(self):
        document = report_fixture()
        short = document["results"][0]
        short["workload"] = "short"
        short["summary"]["peak_rss_gib"] = 12
        medium = copy.deepcopy(short)
        medium.update(workload="medium", text="A longer paragraph.")
        medium["summary"]["mean_time_seconds"] = 900
        document["results"] = [short, medium]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(document))
            markdown = render([path], Path(directory))
        self.assertIn("Workload: short.", markdown)
        self.assertIn("Workload: medium.", markdown)
        self.assertIn("900.0", markdown)
        self.assertEqual(markdown.count("| 12.0 |"), 2)

    def test_incomplete_runs_failures_and_markdown_cells(self):
        document = report_fixture()
        success, failed, skipped = document["results"]
        success["requested_runs"] = 3
        success["runs"][0]["finish_reason"] = "length"
        failed.update(status="failed", error="OOM | <device>\nfailed")
        skipped.update(status="skipped", reason="unsupported")
        for result in (failed, skipped):
            result.pop("summary")
            result.pop("runs")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "report one.json"
            path.write_text(json.dumps(document))
            markdown = render([path, path], root)
        self.assertEqual(markdown.count("### report one"), 2)
        self.assertIn("[Raw JSON](report%20one.json)", markdown)
        self.assertIn("| 1/3 | 284.7 | 0.0 | 2.5 | 113.5 | 11.4 |", markdown)
        self.assertIn("success (length)", markdown)
        self.assertIn("failed: OOM &#124; &lt;device&gt; failed", markdown)
        self.assertIn("skipped: unsupported", markdown)
        self.assertIn("| 0/1 | — | — | — | — | — |", markdown)


if __name__ == "__main__":
    unittest.main()
