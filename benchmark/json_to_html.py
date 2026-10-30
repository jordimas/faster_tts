"""Compare benchmark JSON summaries in a standalone HTML table."""

import argparse
import json
from html import escape
from pathlib import Path


def render(paths):
    versions, rows = [], {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1 or "summary" not in document:
            raise ValueError(f"{path}: expected a version 1 report with a summary")
        version = path.stem.split("-", 1)[0]
        if version not in versions:
            versions.append(version)
        for summary in document["summary"]:
            device = summary["device"]
            index = summary["device_index"]
            hardware = next(
                (
                    result["hardware"]
                    for result in document["results"]
                    if result["device"] == device and result["device_index"] == index
                ),
                {},
            )
            name = (hardware.get("gpu" if device == "cuda" else "cpu") or {}).get(
                "name" if device == "cuda" else "processor", "unknown hardware"
            )
            configuration = (
                document["backend"],
                summary["model"],
                summary["memory_profile"],
                summary["threads"],
                tuple(summary["workloads"]),
            )
            key = (f"{device}:{index} — {name}", summary["compute_type"], configuration)
            measurements = rows.setdefault(key, {})
            if version in measurements:
                raise ValueError(f"{path}: duplicate {version} measurement for {key}")
            measurements[version] = (summary, path.name)
    lines = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Benchmark comparison</title>",
        (
            "<style>body{font:16px system-ui;margin:2rem;color:#17202a}"
            "table{border-collapse:collapse}th,td{border:1px solid #ccd1d1;padding:.6rem}"
            "thead{background:#eef2f5}td{text-align:right}"
            "small{display:block;font-weight:normal;color:#555}</style>"
        ),
        "<h1>Benchmark comparison</h1>",
        (
            "<p>Totals cover timed runs only. RTF = synthesis time / audio duration; "
            "lower is better. Peak RAM includes loading and warmup, excluding GPU memory.</p>"
        ),
        '<table><thead><tr><th rowspan="2">Device</th><th rowspan="2">Compute type</th>',
        *[f'<th colspan="3">{escape(version)}</th>' for version in versions],
        "</tr><tr>",
        *[
            "<th>Total time (s)</th><th>RTF</th><th>Peak RAM (GiB)</th>"
            for _ in versions
        ],
        "</tr></thead><tbody>",
    ]
    for (device, compute, configuration), measurements in sorted(rows.items()):
        backend, model, profile, threads, workloads = configuration
        detail = (
            f"{backend}; {model}; {profile}; {threads} threads; {', '.join(workloads)}"
        )
        lines.append(
            f"<tr><th>{escape(device)}</th><th>{escape(compute)}"
            f"<small>{escape(detail)}</small></th>"
        )
        for version in versions:
            if version not in measurements:
                lines.append('<td colspan="3">—</td>')
                continue
            summary, filename = measurements[version]
            for field in ("total_time_seconds", "rtf", "peak_rss_gib"):
                value = summary.get(field)
                label = "—" if value is None else f"{value:.3f}"
                if summary["status"] != "success":
                    label += " (incomplete)"
                lines.append(f'<td title="{escape(filename, quote=True)}">{label}</td>')
        lines.append("</tr>")
    lines.append("</tbody></table></html>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports", nargs="*", type=Path, help="default: all benchmark/*.json"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("benchmark.html"),
        help="HTML output (default: benchmark.html)",
    )
    args = parser.parse_args()
    paths = args.reports or sorted(Path(__file__).parent.glob("*.json"))
    if not paths:
        parser.error("no JSON reports found; pass report paths explicitly")
    try:
        html = render(paths)
        args.output.write_text(html, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
