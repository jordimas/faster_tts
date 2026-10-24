"""Engine dispatch and public compatibility, without native bindings or weights."""

import inspect
import json
import tempfile
import types
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import faster_tts
from faster_tts import Model, SynthesisResult
from faster_tts.base import _ENGINE_REGISTRY, BaseModel, register_engine


class TestEngine(BaseModel):
    """Synthetic engine deliberately requiring no Drama assets or codec layout."""

    capabilities = ("synthesis",)
    init_options = frozenset({"custom_init"})
    synthesize_options = frozenset({"custom_syn"})

    @staticmethod
    def validate_checkpoint(directory, metadata):
        if metadata.get("test_layout") != 1:
            raise ValueError("test engine requires test_layout=1")

    def __init__(self, directory, metadata, custom_init=None, **settings):
        self.settings = dict(settings, custom_init=custom_init)
        self._info = {"model_path": str(directory), "settings": self.settings}
        self.result = SynthesisResult(
            audio=np.zeros(400, np.float32),
            sample_rate=24000,
            codes=np.zeros((2, 4), np.int32),
            finish_reason="eos",
        )

    def synthesize(self, text, custom_syn=None, **settings):
        self.request = (text, dict(settings, custom_syn=custom_syn))
        return self.result


class ModelDispatchTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "model.bin").write_bytes(b"not native weights")
        self.metadata = {"model_type": "test_tts", "test_layout": 1}
        self.save_config()
        registry = patch.dict(_ENGINE_REGISTRY)
        registry.start()
        self.addCleanup(registry.stop)
        register_engine("test_tts")(TestEngine)

    def save_config(self):
        (self.root / "config.json").write_text(json.dumps(self.metadata))

    def test_selects_engine_and_preserves_positional_and_keyword_settings(self):
        with patch(
            "faster_tts.engines.drama3._native_class", side_effect=AssertionError
        ):
            model = Model(
                self.root,
                3,
                "float32",
                cpu_threads=5,
                num_workers=2,
                engine_options={"custom_init": 7},
            )
            result = model.synthesize(
                "Hello",
                11.0,
                0.8,
                20,
                0.9,
                42,
                True,
                engine_options={"custom_syn": "ok"},
            )
        self.assertIsInstance(model, Model)
        self.assertEqual(model._engine.settings["intra_threads"], 5)
        self.assertEqual(model._engine.settings["compute_type"], "float32")
        self.assertEqual(model._engine.settings["custom_init"], 7)
        self.assertEqual(model._engine.settings["num_workers"], 2)
        self.assertEqual(
            model._engine.request,
            (
                "Hello",
                {
                    "max_duration_s": 11.0,
                    "temperature": 0.8,
                    "top_k": 20,
                    "top_p": 0.9,
                    "seed": 42,
                    "greedy": True,
                    "max_retries": 0,
                    "split_sentences": True,
                    "custom_syn": "ok",
                },
            ),
        )
        self.assertIs(result, model._engine.result)
        self.assertEqual(result.sample_rate, 24000)
        self.assertEqual(result.codes.shape, (2, 4))
        self.assertEqual(result.codes.dtype, np.int32)

    def test_rejects_unknown_engine_options_in_init_and_synthesize(self):
        with patch(
            "faster_tts.engines.drama3._native_class", side_effect=AssertionError
        ):
            with self.assertRaisesRegex(ValueError, "Unknown engine_options"):
                Model(self.root, engine_options={"not_a_real_option": 1})
            model = Model(self.root)
            with self.assertRaisesRegex(ValueError, "Unknown engine_options"):
                model.synthesize("Hello", engine_options={"not_a_real_option": 1})
            with self.assertRaises(TypeError):
                Model(self.root, engine_options=42)
            with self.assertRaises(TypeError):
                model.synthesize("Hello", engine_options=42)

    def test_info_and_engine_errors_pass_through(self):
        model = Model(self.root)
        info = model.info()
        self.assertEqual(info["engine"], "test_tts")
        self.assertEqual(info["capabilities"], ("synthesis",))
        info["settings"]["device"] = "changed"
        self.assertEqual(model.info()["settings"]["device"], "cpu")
        with self.assertRaisesRegex(ValueError, "Unknown engine_options"):
            model.synthesize(
                "Hello",
                engine_options={"history": [("Hi", np.zeros((1, 4), np.int32))]},
            )

    def test_engine_validates_its_own_checkpoint(self):
        self.metadata["test_layout"] = 2
        self.save_config()
        with self.assertRaisesRegex(ValueError, "test_layout"):
            Model(self.root)

    def test_hub_checkpoint_uses_the_same_engine_dispatch(self):
        download = Mock(return_value=str(self.root))
        with patch(
            "faster_tts.cache.import_module",
            return_value=types.SimpleNamespace(snapshot_download=download),
        ):
            model = Model(
                "hf://organization/test-tts", revision="pinned", local_files_only=True
            )
        self.assertEqual(model.info()["engine"], "test_tts")
        download.assert_called_once()
        self.assertEqual(download.call_args.kwargs["repo_id"], "organization/test-tts")
        self.assertEqual(download.call_args.kwargs["revision"], "pinned")
        self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_unknown_or_invalid_engine_never_loads_drama(self):
        with patch(
            "faster_tts.engines.drama3._native_class", side_effect=AssertionError
        ):
            for model_type in ("qwen_not_implemented", None, {}, []):
                with self.subTest(model_type=model_type):
                    self.metadata["model_type"] = model_type
                    self.save_config()
                    with self.assertRaisesRegex(ValueError, "model_type|known engines"):
                        Model(self.root)

    def test_public_signatures_defaults_and_result_fields_remain_compatible(self):
        self.assertEqual(faster_tts.__all__, ["Model", "SynthesisResult"])
        for obj, expected, keyword_only in (
            (
                Model,
                {
                    "model_path": "jordimas/drama-3-ct2",
                    "intra_threads": 8,
                    "compute_type": "default",
                    "cpu_threads": None,
                    "num_workers": 1,
                    "device": "cpu",
                    "device_index": 0,
                    "revision": None,
                    "cache_dir": None,
                    "local_files_only": False,
                    "engine_options": None,
                },
                "cpu_threads",
            ),
            (
                Model.synthesize,
                {
                    "max_duration_s": 24.0,
                    "temperature": 0.7,
                    "top_k": 30,
                    "top_p": 0.7,
                    "seed": 0,
                    "greedy": False,
                    "max_retries": 0,
                    "split_sentences": True,
                    "engine_options": None,
                },
                "max_retries",
            ),
        ):
            parameters = list(inspect.signature(obj).parameters.values())
            if obj is Model.synthesize:
                self.assertEqual([p.name for p in parameters[:2]], ["self", "text"])
                self.assertIs(parameters[1].default, inspect.Parameter.empty)
                parameters = parameters[2:]
            self.assertEqual([p.name for p in parameters], list(expected))
            kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters:
                if parameter.name == keyword_only:
                    kind = inspect.Parameter.KEYWORD_ONLY
                self.assertEqual(parameter.kind, kind)
                self.assertEqual(parameter.default, expected[parameter.name])
        self.assertEqual(
            [field.name for field in fields(SynthesisResult)],
            ["audio", "sample_rate", "codes", "finish_reason", "timings", "attempts"],
        )
