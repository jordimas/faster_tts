"""Engine-agnostic scaffolding for CT2 TTS engines.

Each engine lives under ``faster_tts/engines/<name>/`` and provides a ``Model``
subclass of :class:`BaseModel`, registered with :func:`register_engine` keyed by
the converted checkpoint's ``model_type``. Drama 3 is the first engine; other
CT2 TTS engines are expected to plug in through the same seam.

This module intentionally holds only cross-engine helpers. Drama 3-specific
constants (sample rate, codebook count, codec hop, native class name,
expert-cache budget) belong on the engine subclass, not here.
"""

from copy import deepcopy
from importlib import import_module

import numpy as np

_ENGINE_REGISTRY: dict[str, type] = {}


def register_engine(model_type):
    """Decorator that registers a :class:`BaseModel` subclass by ``model_type``.

    ``model_type`` is matched against the ``model_type`` field of the converted
    checkpoint's ``config.json``. Re-registering an existing key replaces the
    previous binding so engines can be reloaded during tests.
    """
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("model_type must be a nonempty string")

    def decorator(cls):
        _ENGINE_REGISTRY[model_type] = cls
        cls.engine_model_type = model_type
        return cls

    return decorator


def get_engine_class(model_type):
    """Look up the engine class previously registered for ``model_type``."""
    import_module(".engines.drama3", __package__)
    if not isinstance(model_type, str):
        raise ValueError("Converted checkpoint must specify a string model_type")  # noqa: TRY004 - invalid checkpoint data
    try:
        return _ENGINE_REGISTRY[model_type]
    except KeyError:
        raise ValueError(
            f"No CT2 TTS engine registered for model_type={model_type!r}; "
            f"known engines: {sorted(_ENGINE_REGISTRY)}"
        ) from None


def _positive_int(value, name):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= np.iinfo(np.int32).max
    ):
        raise ValueError(f"{name} must be a positive int32 integer")


class BaseModel:
    """Shared surface across CT2 TTS engines.

    Subclasses set ``self._info`` in their constructor and implement
    ``validate_checkpoint(directory, config)`` and ``synthesize``. Constructors
    receive a resolved directory, checkpoint metadata, and the public loader's
    settings. Engines must reject unsupported nondefault synthesis options and
    return a SynthesisResult with engine-specific codec dimensions. Declare
    supported features in ``capabilities``. The base class adds engine identity
    and capabilities to an independent copy of ``_info``.
    """

    engine_model_type: str
    capabilities = ("synthesis",)

    def info(self):
        """Return device/offline execution settings and recorded checkpoint metadata.

        The returned dict is a deep copy of the engine's internal view, safe to
        mutate without affecting the model. Engine-specific fields may be
        present in addition to the shared keys.

        Returns:
            dict: For Drama 3, includes `backend` and `device` (resolved device),
                `codec_device`, `mode`, `gpu_supported`, `device_index`,
                `compute_type`, `cpu_threads` (also `intra_threads`), `num_workers`,
                `prefill_chunk_size`, and resolved `model_path`. Expert-cache
                budgets appear as `gpu_expert_cache_gb` (total) and
                `gpu_expert_cache_gb_per_worker`; these are None when delegated
                to the environment. Checkpoint metadata includes `source_revision`,
                `quantization_policy`, `matrix_dtype`, and `activation_dtype`,
                which may be None. Compute type is the requested loader setting.
        """
        return dict(
            deepcopy(self._info),
            engine=self.engine_model_type,
            capabilities=self.capabilities,
        )
