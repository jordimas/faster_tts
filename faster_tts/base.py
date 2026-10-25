"""Engine registration and shared helpers; engine-specific code lives in engines/."""

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
    settings. Engine-specific knobs are declared in ``init_options`` and
    ``synthesize_options`` as frozensets of accepted keys; the dispatcher
    validates ``engine_options`` against these and raises ``ValueError`` on
    unknown keys. Return a SynthesisResult with engine-specific codec
    dimensions. Declare supported features in ``capabilities``. The base class
    adds engine identity and capabilities to an independent copy of ``_info``.
    """

    engine_model_type: str
    capabilities = ("synthesis",)
    init_options: frozenset = frozenset()
    synthesize_options: frozenset = frozenset()

    def info(self):
        """Copy engine metadata and add its registered identity and capabilities.

        Returns:
            dict: Independent deep copy of `_info`, plus `engine` and
                `capabilities`. Public fields are documented in `Model.info`.
        """
        return dict(
            deepcopy(self._info),
            engine=self.engine_model_type,
            capabilities=self.capabilities,
        )
