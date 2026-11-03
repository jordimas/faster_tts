"""GPU runtime queries are tested without requiring a GPU."""

import argparse
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch

from faster_tts.__main__ import _cache_budget
from faster_tts.memory import (
    CACHE_ENV,
    CPU_MMAP_ENV,
    auto_cpu_mmap,
    available_cpu_memory,
    expert_cache_budget,
    free_gpu_memory,
)


class MemoryQueryTest(unittest.TestCase):
    def test_available_ram_uses_reclaimable_memory(self):
        with (
            patch("faster_tts.memory.sys.platform", "linux"),
            patch.object(Path, "read_text", return_value="MemAvailable: 12345 kB\n"),
        ):
            self.assertEqual(available_cpu_memory(), 12345 * 1024)

    def test_unavailable_ram_query_is_optional(self):
        with (
            patch.object(Path, "read_text", side_effect=OSError),
            patch("faster_tts.memory.os.sysconf", side_effect=ValueError),
        ):
            self.assertIsNone(available_cpu_memory())

    def test_cpu_mapping_threshold_overrides_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with (directory / "model.bin").open("wb") as weights:
                weights.truncate(3_000_000_000)
            cases = [
                (4e9, True, "default", "linux", None, "1"),
                (10e9, True, "default", "linux", None, None),
                (None, True, "default", "linux", None, None),
                (4e9, False, "default", "linux", None, None),
                (4e9, True, "int8", "linux", None, None),
                (4e9, True, "default", "win32", None, None),
                (4e9, True, "float32", "darwin", None, "1"),
                (4e9, True, "default", "linux", "0", "0"),
                (10e9, True, "default", "linux", "1", "1"),
                (5e9, True, "default", "linux", None, None),
            ]
            for ram, enabled, compute, platform, explicit, expected in cases:
                with (
                    self.subTest(case=(ram, enabled, compute, platform, explicit)),
                    patch.dict(os.environ),
                    patch("faster_tts.memory.sys.platform", platform),
                    patch("faster_tts.memory.available_cpu_memory", return_value=ram),
                ):
                    os.environ.pop(CPU_MMAP_ENV, None)
                    if explicit is not None:
                        os.environ[CPU_MMAP_ENV] = explicit
                    with self.assertRaisesRegex(RuntimeError, "load failed"):
                        with auto_cpu_mmap(directory, enabled, compute):
                            self.assertEqual(os.environ.get(CPU_MMAP_ENV), expected)
                            raise RuntimeError("load failed")
                    self.assertEqual(os.environ.get(CPU_MMAP_ENV), explicit)

    def test_cli_budget_parser(self):
        self.assertEqual(_cache_budget("auto"), "auto")
        self.assertEqual(_cache_budget("uncapped"), "uncapped")
        self.assertEqual(_cache_budget("12.5"), 12.5)
        for value in ("0", "-2", "nan", "inf", "bad"):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                _cache_budget(value)

    def test_concurrent_loads_do_not_mix_environment_budgets(self):
        barrier = Barrier(2)

        def load(budget):
            barrier.wait(timeout=5)
            with expert_cache_budget(budget, "cuda", 0, 1):
                before = os.environ[CACHE_ENV]
                time.sleep(0.02)
                return before, os.environ[CACHE_ENV]

        with patch.dict(os.environ, {CACHE_ENV: "7"}):
            with ThreadPoolExecutor(max_workers=2) as executor:
                observed = list(executor.map(load, [4, 8]))
            self.assertEqual(os.environ[CACHE_ENV], "7")
        self.assertEqual(observed, [("4", "4"), ("8", "8")])

    def runtime(self, prefix, status=0):
        def get_device(pointer):
            pointer._obj.value = 3
            return 0

        def mem_info(free, total):
            free._obj.value = 8_000_000_000
            total._obj.value = 16_000_000_000
            return status

        return SimpleNamespace(
            **{
                prefix + "GetDevice": Mock(side_effect=get_device),
                prefix + "SetDevice": Mock(return_value=0),
                prefix + "MemGetInfo": Mock(side_effect=mem_info),
            }
        )

    def test_cuda_and_rocm_select_device_and_restore_previous_device(self):
        for prefix in ("hip", "cuda"):
            runtime = self.runtime(prefix)
            libraries = [runtime] if prefix == "hip" else [OSError("no HIP"), runtime]
            with (
                self.subTest(prefix=prefix),
                patch("faster_tts.memory.find_library", return_value=None),
                patch("faster_tts.memory.ctypes.CDLL", side_effect=libraries),
            ):
                self.assertEqual(free_gpu_memory(2), 8_000_000_000)
            self.assertEqual(
                getattr(runtime, prefix + "SetDevice").call_args_list,
                [((2,),), ((3,),)],
            )

    def test_failed_query_restores_device_and_requests_explicit_budget(self):
        runtime = self.runtime("hip", status=1)
        with (
            patch("faster_tts.memory.find_library", return_value=None),
            patch(
                "faster_tts.memory.ctypes.CDLL",
                side_effect=[runtime, OSError("no CUDA")],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "explicit GB budget"):
                free_gpu_memory(2)
        runtime.hipSetDevice.assert_called_with(3)


if __name__ == "__main__":
    unittest.main()
