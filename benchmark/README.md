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

By default, each hardware model gets a report in the current directory named
`<hardware>_<backend>_<YYYY-MM-DD>.json`, with all tested compute types and memory
profiles together. Hardware names use lowercase letters and hyphens. Existing
reports are preserved using `_01`, `_02`, etc.; dates use UTC. Omit
`--compute-type` to test all supported types. Use `--output-json <path>` to write
all devices into one file at an explicit path (overwritten on each invocation).

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
