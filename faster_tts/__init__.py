"""CTranslate2 text-to-speech through the public Model and SynthesisResult API.

See README.md for setup and examples. Engine-specific helpers are internal.
Importing the package does not download checkpoints or initialize a model.
"""

from .model import Model, SynthesisResult

__all__ = ["Model", "SynthesisResult"]
