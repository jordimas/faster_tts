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
    ``synthesize``. The base class provides a stable :meth:`info` contract so
    callers can introspect any engine the same way.
    """

    engine_model_type: str

    def info(self):
        """Return device/offline execution settings and recorded checkpoint metadata.

        The returned dict is a deep copy of the engine's internal view, safe to
        mutate without affecting the model. Engine-specific fields may be
        present in addition to the shared keys.
        """
        return deepcopy(self._info)
