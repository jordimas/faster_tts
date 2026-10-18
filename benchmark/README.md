## Performance benchmarking

Run the standalone script from the repository root:

```bash
uv run python benchmark/benchmark.py drama-3 --threads 8 --runs 3
uv run python benchmark/benchmark.py drama-3 --device cuda --compute-type float16
```

Use `--text` for your own input, or pass a local model directory instead of
`drama-3`. The script loads once, warms up once, then reports synthesis time,
audio duration, mean latency and sample standard deviation across 3 runs by
default, real-time factor (RTF), device, and peak process RAM (RSS, in GiB).
Peak RSS includes loading, warmup, and timed runs; it excludes filesystem cache
outside the process. The script uses Python's resource module (Linux/macOS).
RTF below 1 means faster than real
time. Loading and warmup are excluded. Keep text and model fixed when comparing
settings; check audio durations because different precisions can change output.
Without `--compute-type`, it benchmarks every compute type reported as supported
by CTranslate2 for the selected device. Specify a type to benchmark only that one.
Each compute type runs in a separate process so model memory and native allocator
caches are released before the next type starts. Failed types are reported and
the sweep continues, exiting with status 1 if any failed. A single type can still
exceed available memory; Drama 3 GPU inference also keeps CPU float32 weights.
