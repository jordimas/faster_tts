"""Render benchmark JSON reports as a Markdown section for the public README."""

import argparse
import json
import os
from html import escape
from pathlib import Path
from urllib.parse import quote


def cell(value):
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return (
        escape(str(value), quote=False)
        .replace("|", "&#124;")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def render(paths, relative_to):
    lines = [
        "## Public benchmark results",
        "",
        (
            "RTF is synthesis time divided by audio duration; lower is better and "
            "1 is real time. Loading and warmup are excluded from timings. Peak RAM "
            "is process RSS, including loading and warmup, and excludes GPU memory. "
            "These measurements describe speed, not speech quality; EOS does not "
            "guarantee complete or accurate speech. Single runs do not measure "
            "timing variability."
        ),
        "",
    ]
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1:
            raise ValueError(f"{path}: expected benchmark schema_version 1")
        link = quote(Path(os.path.relpath(path, relative_to)).as_posix())
        lines.extend(
            [
                f"### {cell(path.stem)}",
                "",
                (
                    f"{cell(document['backend'])} {cell(document['backend_version'])}; "
                    f"{cell(document['created_at'])}. [Raw JSON]({link})."
                ),
                "",
            ]
        )
        groups = {}
        for result in document["results"]:
            hardware = result["hardware"]
            cpu = hardware["cpu"]
            gpu = hardware.get("gpu") or {}
            context = (
                result["device"],
                gpu.get("name") if result["device"] == "cuda" else cpu["processor"],
                cpu.get("system_ram_gib"),
                gpu.get("total_memory_gib"),
                gpu.get("runtime"),
                result["model"],
                result["text"],
                result["threads"],
                result["warmup_runs"],
            )
            groups.setdefault(context, []).append(result)
        for context, results in groups.items():
            device, name, ram, vram, runtime, model, text, threads, warmup = context
            hardware = f"**{cell(device)} — {cell(name)}**; RAM: {cell(ram)} GiB"
            if device == "cuda":
                hardware += f"; GPU memory: {cell(vram)} GiB; runtime: {cell(runtime)}"
            lines.extend(
                [
                    hardware + ".",
                    "",
                    f"Model: {cell(model)}; CPU threads: {threads}; warmup runs: {warmup}.",
                    f"Test text: {cell(text)}",
                    "",
                    (
                        "| Precision | Memory profile | Expert cache (GB) | Runs | "
                        "Mean time (s) | Std dev (s) | Audio (s) | RTF | "
                        "Peak RAM (GiB) | Status |"
                    ),
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for result in results:
                summary = result.get("summary", {})
                runs = result.get("runs", [])
                status = result["status"]
                reasons = sorted({run["finish_reason"] for run in runs})
                if reasons:
                    status += " (" + ", ".join(reasons) + ")"
                detail = result.get("reason") or result.get("error")
                if detail:
                    status += ": " + detail
                values = [
                    result["compute_type"],
                    result["memory_profile"],
                    result.get("gpu_expert_cache_gb"),
                    f"{len(runs)}/{result['requested_runs']}",
                    *[
                        summary.get(key)
                        for key in (
                            "mean_time_seconds",
                            "std_time_seconds",
                            "mean_audio_seconds",
                            "rtf",
                            "peak_rss_gib",
                        )
                    ],
                    status,
                ]
                lines.append("| " + " | ".join(map(cell, values)) + " |")
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="Benchmark JSON files")
    parser.add_argument(
        "-o", "--output", type=Path, help="Write Markdown instead of stdout"
    )
    args = parser.parse_args()
    relative_to = args.output.resolve().parent if args.output else Path.cwd()
    markdown = render(args.reports, relative_to)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
