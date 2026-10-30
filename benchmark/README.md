# Benchmarks

Run from the repository root:

```bash
# CTranslate2
uv run python benchmark/benchmark.py drama-3 --device cpu
uv run python benchmark/benchmark.py drama-3 --device cuda --compute-type float16

# Official Hugging Face / PyTorch implementation
python benchmark/benchmark_hf.py --device cuda --compute-type float16
```

The Hugging Face benchmark requires the Fish Speech dependencies, checkpoint,
and source tree. Set `--source` and the model path as needed.

Both runners measure two fixed workloads by default: **short** (the original
greeting) and **medium** (an 80-word paragraph). Use `--workload short` or
`--workload medium` to select one, or `--text "Your text"` for a custom input.
`--text` and `--workload` are mutually exclusive. Use `--help` for all options.
Each workload runs twice by default; use `--runs` to override this.

Each device/precision worker loads the model once and warms up each workload
before timing it. Latency, audio duration, standard deviation, and real-time
factor are reported separately. One peak process RAM value covers model loading,
warmups, and all timed runs across both workloads; GPU memory is excluded.
JSON repeats this shared peak in each workload summary. Audio budgets are
24 seconds for short/custom input and 90 seconds for medium input.

The top-level JSON `summary` groups workloads by device/index and compute type
(and model, thread count, and memory profile). It includes `total_time_seconds`,
`total_audio_seconds`, overall `rtf` (total time divided by total audio), and
`peak_rss_gib` (the maximum observed peak). Totals include all timed runs of
successful workloads, excluding loading and warmup. Groups containing failed or
skipped workloads are marked `incomplete`; unavailable RTF or memory is `null`.

By default, each invocation writes one report in the current directory named
`<hostname>_<backend>_<YYYY-MM-DD>.json`, with all tested devices, workloads,
compute types, and memory profiles together. Hostnames use lowercase letters
and hyphens. Existing
reports are preserved using `_01`, `_02`, etc.; dates use UTC. Omit
`--compute-type` to test all supported types. Use `--output-json <path>` to choose
an explicit path (overwritten on each invocation).

Convert all saved reports to a public Markdown section (from the repository root):

```bash
python benchmark/json_to_markdown.py benchmark/*.json --output benchmark/public.md
```

Omit `--output` to print Markdown, with raw JSON links relative to the current
directory. To append the section to the main README:

```bash
python benchmark/json_to_markdown.py benchmark/*.json >> README.md
```

Each report remains a separate measurement, including repeated hardware runs.
Failed and skipped configurations are retained; successful rows show completed
versus requested runs and generation finish reasons.

Compare JSON summaries in a standalone HTML table:

```bash
# All JSON reports directly under benchmark/
python benchmark/json_to_html.py
# Selected reports (shell wildcards also work)
python benchmark/json_to_html.py benchmark/v1-*.json benchmark/v2-*.json -o benchmark/comparison.html
```

Column headers use the filename prefix before the first `-` (for example,
`v2-cpu.json` becomes `v2`). Rows identify hardware/device and compute type;
each version shows total synthesis seconds, RTF, and peak host RAM from the
JSON summary. Different backends, models, memory profiles, thread counts, or
workload selections stay in separate rows. Missing measurements show a dash;
incomplete measurements are labeled. Duplicate measurements for the same
version and row are rejected. Reports must contain the top-level `summary`.
Output defaults to `benchmark.html` in the current directory; use `--output` to
choose another path.
