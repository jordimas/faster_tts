"""Reference generation must follow the public API without loading models."""

import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import faster_tts

generator = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "generate_docs.py")
)


class DocsTest(unittest.TestCase):
    def test_reference_covers_exports_and_inherited_methods_without_loading(self):
        with patch(
            "faster_tts.engines.drama3._native_class", side_effect=AssertionError
        ):
            markdown = generator["generate"]()
        for name in faster_tts.__all__:
            self.assertIn(f"## {name}\n", markdown)
            self.assertIn(f"[{name}](#{name.lower()})", markdown)
        self.assertIn("### Model.info\n", markdown)
        self.assertIn("### Model.synthesize\n", markdown)
        self.assertIn("- **audio**:", markdown)
        self.assertNotIn("### Model.__init__", markdown)
        self.assertNotIn("0x", markdown)

    def test_new_methods_and_docstrings_are_picked_up_automatically(self):
        class Example:
            """A new exported API."""

            def method(self, text, *, limit=7):
                """Do something.

                Args:
                    text: Input text.
                    limit: Maximum count.

                Returns:
                    str: Result text.

                Example:
                    ```python
                    example.method("hi")
                    ```
                """

        with (
            patch.object(faster_tts, "__all__", ["Example"]),
            patch.object(faster_tts, "Example", Example, create=True),
        ):
            markdown = generator["generate"]()
        self.assertIn("Example.method(text, *, limit=7)", markdown)
        self.assertIn("- **text**: Input text.", markdown)
        self.assertIn("- **str**: Result text.", markdown)
        self.assertIn('```python\nexample.method("hi")\n```', markdown)
