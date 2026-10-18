## Performance benchmarking

Run the standalone script from the repository root:

```bash
uv run python benchmark/benchmark.py drama-3 --threads 8 --runs 3
uv run python benchmark/benchmark.py drama-3 --device cuda --compute-type float16
```

Use `--text` for your own input, or pass a local model directory instead of
`drama-3`. The script loads once, warms up once, then reports synthesis time,
audio duration, latency, real-time factor (RTF), device, and peak process RAM
(RSS, in GiB). It performs one timed run by default; use `--runs` for more
runs and their mean latency and sample standard deviation.
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

For GPU runs, the script defaults `CT2_CUDA_CACHING_ALLOCATOR_CONFIG` to
`2,6,24,209715200`. When the CUB allocator is active (the ROCm default), this
rounds allocations to powers of two instead of powers of four, reducing padding
for cached float16/bfloat16 expert weights. Explicit environment settings take
precedence; the setting has no effect on the asynchronous CUDA allocator.
It limits unused cached blocks to 200 MiB, not total GPU memory. Live expert
weights can still exceed available memory. On integrated GPUs, GPU allocations
also consume system RAM and are not included in the reported process RSS.
