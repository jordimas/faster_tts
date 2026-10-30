"""Run from the repository root: uv run python benchmark/benchmark.py [model]."""

import importlib.util
import os
import tempfile

import ctranslate2
from benchmark_common import benchmark_parser, measure, run_logged, sweep
from benchmark_results import Results, ct2_gpu_hardware
from check_quality import run_checker
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

from faster_tts import Model

parser = benchmark_parser(__doc__)
parser.add_argument("model", nargs="?", default="drama-3")
parser.add_argument("--compute-type", help="default: benchmark all supported types")
args = parser.parse_args()
args.check_quality = args.check_quality or args.check == "quality"
if args.check_quality and importlib.util.find_spec("faster_whisper") is None:
    parser.error(
        "quality checking requires faster-whisper; run with uv run --with faster-whisper"
    )
if args.check_quality:
    from check_quality import normalized_words

    if not normalized_words(args.text):
        parser.error("quality checking requires at least one reference word")
if args.runs < 1:
    parser.error("--runs must be positive")
if args.device_index < 0:
    parser.error("--device-index must be nonnegative")

run_logged(args, "ct2")

if args.show_progress:
    enable_progress_bars()
else:
    disable_progress_bars()

results = Results(args, "ct2", ctranslate2.__version__)

if args.device is None:
    devices = [("cpu", 0)] + [
        ("cuda", i) for i in range(ctranslate2.get_cuda_device_count())
    ]
    sweep(
        results,
        [
            ["--device", device, "--device-index", str(index)]
            for device, index in devices
        ],
    )

if args.device == "auto":
    args.device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
if args.device == "cuda":
    # HIP asynchronous memory-pool teardown can segfault after synthesis.
    # Keep this workaround local to the benchmark and honor explicit overrides.
    os.environ.setdefault("CT2_CUDA_ALLOCATOR", "cub_caching")
    # Half-precision expert weights persist across synthesis calls. The CUB
    # default rounds allocations to powers of four, inflating their footprint.
    # Use finer bins before the first allocation; keep explicit user settings.
    os.environ.setdefault("CT2_CUDA_CACHING_ALLOCATOR_CONFIG", "2,6,24,209715200")
if not args.compute_worker:
    print(f"\nDevice: {args.device}:{args.device_index}", flush=True)
if args.compute_type is None:
    # Process exit releases native allocator caches as well as model weights.
    sweep(
        results,
        [
            ["--compute-type", dtype, "--compute-worker"]
            for dtype in sorted(
                ctranslate2.get_supported_compute_types(args.device, args.device_index)
            )
        ],
    )
if args.device == "cuda":
    results.gpu_hardware = ct2_gpu_hardware(args.device_index)
results.start()
model = Model(
    args.model,
    cpu_threads=args.threads,
    device=args.device,
    device_index=args.device_index,
    compute_type=args.compute_type,
)
print(f"\nCompute type: {args.compute_type}", flush=True)
model.synthesize(args.text)  # Exclude loading and first-run costs.


def synthesize():
    result = model.synthesize(args.text)
    return result.audio, result.sample_rate, result.finish_reason


with tempfile.TemporaryDirectory() as audio_dir:
    measure(results, synthesize, audio_dir)
    model = None
    if args.check_quality:
        run_checker(
            args, audio_dir, f"{args.device}:{args.device_index} {args.compute_type}"
        )
