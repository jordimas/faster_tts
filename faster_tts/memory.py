"""GPU expert-cache configuration for the environment-based Drama3 loader."""

from contextlib import contextmanager
import ctypes
from ctypes.util import find_library
import math
from numbers import Real
import os
from threading import RLock


CACHE_ENV = "CT2_DRAMA3_GPU_EXPERT_CACHE_GB"
LOADER_LOCK = RLock()


def validate_cache_budget(value):
    if value is None or isinstance(value, str) and value == "auto":
        return
    if (not isinstance(value, Real) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0):
        raise ValueError('gpu_expert_cache_gb must be None, "auto", or a finite positive number')


def free_gpu_memory(device_index):
    """Query CUDA/ROCm free bytes without importing PyTorch or creating tensors."""
    for name, prefix in (("amdhip64", "hip"), ("cudart", "cuda")):
        try:
            runtime = ctypes.CDLL(find_library(name) or f"lib{name}.so")
            get_device = getattr(runtime, prefix + "GetDevice")
            set_device = getattr(runtime, prefix + "SetDevice")
            mem_info = getattr(runtime, prefix + "MemGetInfo")
            get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
            set_device.argtypes = [ctypes.c_int]
            mem_info.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            for function in (get_device, set_device, mem_info):
                function.restype = ctypes.c_int
            previous = ctypes.c_int()
            if get_device(ctypes.byref(previous)) != 0:
                continue
            if set_device(device_index) != 0:
                continue
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            try:
                status = mem_info(ctypes.byref(free), ctypes.byref(total))
            finally:
                set_device(previous.value)
            if status == 0 and 0 < free.value <= total.value:
                return free.value
        except (OSError, AttributeError):
            continue
    raise RuntimeError("Cannot query free GPU memory via CUDA or ROCm; "
                       "set gpu_expert_cache_gb to an explicit GB budget")


@contextmanager
def expert_cache_budget(value, device, device_index, num_workers):
    """Serialize wrapper loads and restore the caller's environment even on failure.

    Explicit/automatic budgets cover all workers. Inherited CT2 budgets retain
    CT2's per-worker semantics. CT2 reads the environment during construction.
    """
    with LOADER_LOCK:
        previous = os.environ.get(CACHE_ENV)
        if value is not None:
            if device != "cuda":
                raise ValueError("gpu_expert_cache_gb requires GPU execution")
            if value == "auto":
                free = free_gpu_memory(device_index)
                # Reserve at least 2 decimal GB and 30% for non-expert allocations.
                value = min(free * .7, free - 2e9) / 1e9
                if value <= 0:
                    raise RuntimeError("Insufficient free GPU memory for automatic expert caching; "
                                       "at least 2 GB of headroom is required")
            per_worker = float(value) / num_workers
        else:
            per_worker = float(previous) if previous else None
        try:
            if value is not None:
                os.environ[CACHE_ENV] = format(per_worker, ".17g")
            yield per_worker
        finally:
            if value is not None:
                if previous is None:
                    os.environ.pop(CACHE_ENV, None)
                else:
                    os.environ[CACHE_ENV] = previous
