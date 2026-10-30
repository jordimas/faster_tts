"""Exercise HTML comparisons using self-contained reports."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from json_to_html import render


def report(time=10, device="cpu", name="Test CPU"):
    return {
        "schema_version": 1,
        "backend": "ct2",
        "results": [
            {
                "device": device,
                "device_index": 0,
                "hardware": {"cpu": {"processor": name}, "gpu": {"name": name}},
            }
        ],
        "summary": [
            {
                "device": device,
                "device_index": 0,
                "compute_type": "float32",
                "model": "drama-3",
                "memory_profile": "default",
                "threads": 8,
                "workloads": ["short", "medium"],
                "status": "success",
                "total_time_seconds": time,
                "rtf": time / 20,
                "peak_rss_gib": 4,
            }
        ],
    }


class HtmlTest(unittest.TestCase):
    def test_hf_and_ct2_share_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            ct2 = Path(directory) / "485-results.json"
            hf = Path(directory) / "hf-results.json"
            ct2.write_text(json.dumps(report()))
            document = report(time=20)
            document["backend"] = "hf"
            document["summary"][0]["model"] = "/models/drama-3"
            hf.write_text(json.dumps(document))
            html = render([ct2, hf])
        self.assertEqual(html.count("<tr><th>cpu:0"), 1)
        self.assertIn("485: ct2; hf: hf", html)

    def test_versions_hardware_missing_values_and_escaping(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                Path(directory) / name
                for name in ("v1-cpu.json", "v2-cpu.json", "v2-gpu.json")
            ]
            for path, document in zip(
                paths,
                (
                    report(name="CPU <test>"),
                    report(time=8, name="CPU <test>"),
                    report(device="cuda", name="GPU"),
                ),
            ):
                path.write_text(json.dumps(document))
            html = render(paths)
        self.assertEqual(html.count('<th colspan="3">v2</th>'), 1)
        self.assertIn('<th colspan="3">v1</th>', html)
        self.assertIn("cpu:0 — CPU &lt;test&gt;", html)
        self.assertIn("cuda:0 — GPU", html)
        self.assertEqual(html.count("drama-3; default; 8 threads; short, medium"), 1)
        self.assertIn(">8.000</td>", html)
        self.assertIn(">0.400</td>", html)
        self.assertIn('<td colspan="3">—</td>', html)

    def test_incomplete_and_duplicate_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1-report.json"
            document = report()
            document["summary"][0].update(status="incomplete", rtf=None)
            path.write_text(json.dumps(document))
            self.assertIn("— (incomplete)", render([path]))
            with self.assertRaisesRegex(ValueError, "duplicate v1 measurement"):
                render([path, path])

    def test_cli_selects_only_requested_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "selected-report.json"
            selected.write_text(json.dumps(report()))
            (root / "ignored-report.json").write_text("invalid JSON")
            output = root / "comparison.html"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("json_to_html.py")),
                    str(selected),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('<th colspan="3">selected</th>', output.read_text())
