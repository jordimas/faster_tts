"""Check CPU sweep memory safeguards without loading model weights."""

import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from faster_tts.memory import CPU_MMAP_ENV


class BenchmarkMemoryTest(unittest.TestCase):
    def test_explicit_cpu_sweep_maps_large_float32_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            with (model_path / "model.bin").open("wb") as weights:
                weights.truncate(3_000_000_000)
            for compute, override, expected in (
                ("float32", None, "1"),
                ("int8", None, None),
                ("float32", "0", "0"),
            ):
                observed = []

                def load(*args, observed=observed, **kwargs):
                    observed.append(os.environ.get(CPU_MMAP_ENV))
                    return Mock()

                with (
                    self.subTest(compute=compute, override=override),
                    patch.dict(os.environ),
                    patch(
                        "sys.argv",
                        ["benchmark.py", "--device", "cpu", "--compute-type", compute],
                    ),
                    patch("benchmark_common.run_logged"),
                    patch("benchmark_common.measure"),
                    patch("benchmark_results.Results"),
                    patch("faster_tts.Model", side_effect=load),
                    patch(
                        "faster_tts.cache.resolve_model_path", return_value=model_path
                    ),
                    patch("faster_tts.memory.sys.platform", "linux"),
                    patch("faster_tts.memory.available_cpu_memory", return_value=4e9),
                ):
                    os.environ.pop(CPU_MMAP_ENV, None)
                    if override is not None:
                        os.environ[CPU_MMAP_ENV] = override
                    runpy.run_path(str(Path(__file__).with_name("benchmark.py")))
                    self.assertEqual(observed, [expected])
                    self.assertEqual(os.environ.get(CPU_MMAP_ENV), override)
