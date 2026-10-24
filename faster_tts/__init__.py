"""Text-to-speech with CPU and GPU acceleration through CTranslate2.

Create a `Model` once and reuse it across synthesis calls:

```python
from faster_tts import Model

model = Model()
result = model.synthesize("Hello! Welcome to our application.")
audio, sample_rate = result.audio, result.sample_rate
```

`Model` selects an engine from the converted checkpoint's `model_type`; only
Drama 3 is currently supported and remains the default. `SynthesisResult`
contains audio, codec indices, and generation diagnostics. For lower-level prompt
inspection without loading model weights, use `Drama3PromptBuilder` and
`Drama3Prompt`.

This reference covers the four public classes exported from `faster_tts`.
Importing the package or building these pages does not download a checkpoint
or initialize a model. Constructing `Model` requires a matching CT2 engine build.
"""

from .model import Model, SynthesisResult
from .prompt import Drama3Prompt, Drama3PromptBuilder

__all__ = ["Drama3Prompt", "Drama3PromptBuilder", "Model", "SynthesisResult"]
