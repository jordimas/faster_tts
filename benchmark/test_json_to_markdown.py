"""Check public reporting keeps incomplete measurements visible."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from json_to_markdown import render


class MarkdownTest(unittest.TestCase):
    def test_workload_reports_keep_timings_and_shared_peak(self):
        source = Path(__file__).parent / "arm_ct2_2026-10-25.json"
        document = json.loads(source.read_text())
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
        source = Path(__file__).parent / "arm_ct2_2026-10-25.json"
        document = json.loads(source.read_text())
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
