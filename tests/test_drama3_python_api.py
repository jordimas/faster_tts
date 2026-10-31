"""Python API boundary tests with a fake native engine; no model inference claims."""

import json
import os
import sys
import tempfile
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock, patch

import numpy as np
from drama3_fixtures import make_prompt_assets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from faster_tts import Model, SynthesisResult
from faster_tts.__main__ import main


class Drama3PythonAPITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _, config = make_prompt_assets(self.root, {"[UNK]": 0, "Hello": 1})
        config["text_config"] = {"max_position_embeddings": 8192}
        (self.root / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "drama3",
                    "ct2_layout_version": 1,
                    "source_config": config,
                    "source_revision": "source-commit",
                    "quantization_policy": "all_matrices_int8",
                    "matrix_dtype": "int8",
                    "activation_dtype": "float32",
                }
            )
        )
        (self.root / "model.bin").write_bytes(
            b"synthetic assets; native engine is mocked"
        )
        self.result = {
            "audio": np.zeros(2048, np.float32),
            "codes": np.zeros((1, 10), np.int32),
            "sample_rate": 44100,
            "finish_reason": "eos",
        }
        self.backend = Mock()
        self.backend.generate.return_value = self.result
        self.constructor = Mock(return_value=self.backend)
        self.native_patch = patch(
            "faster_tts.engines.drama3._native_class", return_value=self.constructor
        )
        self.native_patch.start()
        self.addCleanup(self.native_patch.stop)
        memory_patch = patch("faster_tts.memory.free_gpu_memory", return_value=20e9)
        memory_patch.start()
        self.addCleanup(memory_patch.stop)
        device_patch = patch(
            "faster_tts.engines.drama3.import_module",
            return_value=Mock(get_cuda_device_count=Mock(return_value=0)),
        )
        device_patch.start()
        self.addCleanup(device_patch.stop)

    def test_references_encode_once_and_stay_in_system_across_chunks(self):
        self.backend.prefix_cache_stats = {"entries": 0}
        self.backend.encode_reference.return_value = np.ones((3, 10), np.int32)
        waveform = np.zeros(4410, np.float32)
        model = Model(self.root)
        with patch(
            "faster_tts.references.load_references",
            return_value=[("<|speaker:2|> Reference", waveform)],
        ):
            result = model.synthesize(
                "<|speaker:2|> Hello. Goodbye.",
                references=[{"audio": "voice.wav", "text": "Reference", "speaker": 2}],
            )
        self.backend.encode_reference.assert_called_once_with(waveform)
        self.assertEqual(len(result.codes), 2)
        builder = model._engine._prompt
        codes = self.backend.encode_reference.return_value
        for call in self.backend.generate.call_args_list:
            rows = call.args[0]
            expected = builder.prepare(
                "<|speaker:2|> Hello.", references=[("<|speaker:2|> Reference", codes)]
            )
            system_rows = sum(
                len(builder.tokenizer.encode(segment).ids)
                if isinstance(segment, str)
                else len(segment)
                for segment in expected.segments[:5]
            )
            np.testing.assert_array_equal(
                rows[:system_rows], expected.rows[:system_rows]
            )
            self.assertGreaterEqual(call.kwargs["static_prompt_rows"], system_rows)

    def test_references_are_not_reencoded_on_retry(self):
        self.backend.encode_reference.return_value = np.ones((3, 10), np.int32)
        self.backend.generate.side_effect = [
            dict(self.result, finish_reason="length"),
            self.result,
        ]
        with patch(
            "faster_tts.references.load_references",
            return_value=[("Reference", np.zeros(10, np.float32))],
        ):
            Model(self.root).synthesize(
                "Hello",
                max_retries=1,
                references=[{"audio": "voice.wav", "text": "Reference"}],
            )
        self.assertEqual(self.backend.encode_reference.call_count, 1)
        self.assertEqual(self.backend.generate.call_count, 2)
        first, second = self.backend.generate.call_args_list
        np.testing.assert_array_equal(first.args[0], second.args[0])

    def test_cli_reference_files_and_speakers(self):
        model = Mock()
        model.synthesize.return_value = SynthesisResult(**self.result)
        with (
            patch(
                "sys.argv",
                [
                    "faster-tts",
                    "--text",
                    "Hello",
                    "--output",
                    str(self.root / "out.wav"),
                    "--reference-audio",
                    "voice.wav",
                    "--reference-text",
                    "Reference",
                    "--reference-speaker",
                    "2",
                ],
            ),
            patch("faster_tts.__main__.Model", return_value=model),
        ):
            main()
        self.assertEqual(
            model.synthesize.call_args.kwargs["references"],
            [{"audio": Path("voice.wav"), "text": "Reference", "speaker": 2}],
        )

    def test_reference_runtime_without_encoder_is_actionable(self):
        del self.backend.encode_reference
        with patch(
            "faster_tts.references.load_references",
            return_value=[("Reference", np.zeros(10, np.float32))],
        ):
            with self.assertRaisesRegex(RuntimeError, "encode_reference support"):
                Model(self.root).synthesize(
                    "Hello", references=[{"audio": "voice.wav", "text": "Reference"}]
                )
        self.backend.generate.assert_not_called()

    def test_empty_references_preserve_generation(self):
        model = Model(self.root)
        model.synthesize("Hello", references=[])
        self.backend.encode_reference.assert_not_called()

    def test_prefix_cache_uses_system_turn_without_history_and_across_retries(self):
        self.backend.prefix_cache_stats = {"entries": 0, "hits": 0, "misses": 0}
        model = Model(self.root)
        self.backend.generate.side_effect = [
            dict(self.result, finish_reason="length"),
            self.result,
        ]
        model.synthesize("Hello", max_retries=1)
        calls = self.backend.generate.call_args_list
        builder = model._engine._prompt
        expected = sum(
            len(builder.tokenizer.encode(segment).ids)
            for segment in builder.prepare("Hello").segments[:3]
        )
        for call in calls:
            self.assertEqual(call.kwargs["static_prompt_rows"], expected)
            self.assertTrue(call.kwargs["cache_static_prompt"])
            self.assertLess(expected, len(call.args[0]))
            np.testing.assert_array_equal(
                call.args[0][:expected], calls[0].args[0][:expected]
            )

    def test_prefix_cache_pins_first_history_turn_including_audio_and_end_marker(self):
        self.backend.prefix_cache_stats = {"entries": 0, "hits": 0, "misses": 0}
        model = Model(self.root)
        builder = model._engine._prompt
        first = ("Earlier", np.full((3, 10), 1, np.int32))
        plain = builder.prepare("Hello")
        system_rows = sum(
            len(builder.tokenizer.encode(segment).ids) for segment in plain.segments[:3]
        )
        expected = (
            system_rows
            + len(builder.prepare("Hello", history=[first]).rows)
            - len(plain.rows)
        )
        for history in (
            [first],
            [first, ("Recent", self.result["codes"])],
            [first, ("Changed", np.full((2, 10), 2, np.int32))],
        ):
            model.synthesize("Hello", engine_options={"history": iter(history)})
        calls = self.backend.generate.call_args_list
        for call in calls:
            self.assertEqual(call.kwargs["static_prompt_rows"], expected)
            self.assertTrue(call.kwargs["cache_static_prompt"])
            self.assertLess(expected, len(call.args[0]))
            np.testing.assert_array_equal(
                call.args[0][:expected], calls[0].args[0][:expected]
            )
        self.assertEqual(np.count_nonzero(calls[0].args[0][:expected, 1]), 3)

    def test_chunk_prefix_cache_reuses_first_completed_segment(self):
        self.backend.prefix_cache_stats = {"entries": 0, "hits": 0, "misses": 0}
        Model(self.root).synthesize("First. Second. Third. Fourth.")
        calls = self.backend.generate.call_args_list
        sizes = [call.kwargs["static_prompt_rows"] for call in calls]
        self.assertLess(sizes[0], sizes[1])
        self.assertEqual(sizes[1:], [sizes[1]] * 3)
        for call in calls[2:]:
            np.testing.assert_array_equal(
                call.args[0][: sizes[1]], calls[1].args[0][: sizes[1]]
            )

    def test_older_bindings_generate_without_prefix_cache_keywords(self):
        del self.backend.prefix_cache_stats
        Model(self.root).synthesize("Hello. Goodbye.")
        for call in self.backend.generate.call_args_list:
            self.assertNotIn("static_prompt_rows", call.kwargs)
            self.assertNotIn("cache_static_prompt", call.kwargs)

    def test_loading_logs_resolved_device_and_compute_type(self):
        self.backend.compute_type = "int8_float32"
        for requested in ("default", "auto", "int8"):
            with self.subTest(compute_type=requested):
                with self.assertLogs("faster_tts.engines.drama3", level="INFO") as logs:
                    Model(self.root, device="auto", compute_type=requested)
                self.assertIn(
                    "device=cpu (device_index=0), compute_type=int8_float32 "
                    f"(requested={requested})",
                    logs.output[0],
                )

        with (
            patch(
                "faster_tts.engines.drama3.import_module",
                return_value=Mock(get_cuda_device_count=Mock(return_value=1)),
            ),
            self.assertLogs("faster_tts.engines.drama3", level="INFO") as logs,
        ):
            self.backend.compute_type = "float16"
            Model(self.root, device="auto", device_index=2)
        self.assertIn(
            "device=cuda (device_index=2), compute_type=float16 (requested=default)",
            logs.output[0],
        )

    def test_loading_logs_unavailable_compute_type_for_older_bindings(self):
        del self.backend.compute_type
        with self.assertLogs("faster_tts.engines.drama3", level="INFO") as logs:
            Model(self.root)
        self.assertIn("compute_type=unavailable (requested=default)", logs.output[0])

    def test_segment_logging_reports_each_chunk_and_includes_retry_time(self):
        model = Model(self.root)
        with (
            patch(
                "faster_tts.engines.drama3.time.perf_counter",
                side_effect=[100, 102, 1000, 1003],
            ),
            self.assertLogs("faster_tts.engines.drama3", level="INFO") as logs,
        ):
            model.synthesize("Hello. Goodbye.")
        self.assertEqual(len(logs.output), 2)
        for message, text in zip(logs.output, ("Hello.", "Goodbye.")):
            self.assertRegex(message, r"Synthesized segment in \d+\.\d{3} s \(eos\)")
            self.assertIn(f": {text}\nHistory:", message)
        self.assertTrue(logs.output[0].endswith("History: []"))
        self.assertTrue(logs.output[1].endswith("History: ['Hello.']"))
        self.assertIn("2.000 s (eos): Hello.", logs.output[0])
        self.assertIn("3.000 s (eos): Goodbye.", logs.output[1])

        self.backend.generate.side_effect = [
            dict(self.result, finish_reason="length"),
            self.result,
        ]
        with (
            patch(
                "faster_tts.engines.drama3.time.perf_counter",
                side_effect=[100, 102, 200, 203],
            ),
            self.assertLogs("faster_tts.engines.drama3", level="INFO") as logs,
        ):
            model.synthesize("Hello", max_retries=1)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("5.000 s (eos): Hello", logs.output[0])

    def test_segment_logging_shows_supplied_and_bounded_chunk_history(self):
        model = Model(self.root)
        history = iter([("Earlier turn.", self.result["codes"])])
        with self.assertLogs("faster_tts.engines.drama3", level="INFO") as logs:
            model.synthesize(
                "First. Second. Third. Fourth.", engine_options={"history": history}
            )
        expected = [
            ["Earlier turn."],
            ["Earlier turn.", "First."],
            ["Earlier turn.", "First.", "Second."],
            ["Earlier turn.", "First.", "Third."],
        ]
        self.assertEqual(len(logs.output), len(expected))
        for message, texts in zip(logs.output, expected):
            self.assertTrue(message.endswith(f"History: {texts!r}"))

    def test_reuses_loaded_backend_and_passes_contiguous_int32_prompt(self):
        model = Model(
            self.root, intra_threads=3, engine_options={"prefill_chunk_size": 2}
        )
        self.constructor.assert_called_once_with(
            str(self.root.resolve()),
            intra_threads=3,
            inter_threads=1,
            compute_type="default",
            device="cpu",
            device_index=0,
        )
        first = model.synthesize("Hello", engine_options={"max_frames": 1})
        second = model.synthesize("Hello", engine_options={"max_frames": 2})
        self.assertIsInstance(first, SynthesisResult)
        self.assertEqual(second.finish_reason, "eos")
        self.assertEqual(second.sample_rate, 44100)
        self.assertEqual(second.audio.shape, (2048,))
        self.assertEqual(self.constructor.call_count, 1)
        self.assertEqual(self.backend.generate.call_count, 2)
        rows = self.backend.generate.call_args.args[0]
        self.assertEqual(rows.dtype, np.int32)
        self.assertTrue(rows.flags.c_contiguous)
        self.assertEqual(rows.shape[1], 11)
        self.assertFalse(np.any(rows[:, 1:]))
        self.assertEqual(
            self.backend.generate.call_args.kwargs,
            {
                "max_frames": 2,
                "prefill_chunk_size": 2,
                "temperature": 0.7,
                "top_k": 30,
                "top_p": 0.7,
                "seed": 0,
                "greedy": False,
            },
        )

    def test_cpu_threads_controls_native_thread_count(self):
        for options in ({"cpu_threads": 4}, {"cpu_threads": 4, "intra_threads": 2}):
            model = Model(self.root, **options)
            self.assertEqual(self.constructor.call_args.kwargs["intra_threads"], 4)
            self.assertEqual(model.info()["intra_threads"], 4)

    def test_auto_cpu_mapping_only_applies_during_cpu_auto_load(self):
        key = "CT2_DRAMA3_CPU_MMAP"
        for device, gpu_count, expected in (
            ("auto", 0, "1"),
            ("auto", 1, None),
            ("cpu", 0, None),
        ):
            observed = []
            self.constructor.side_effect = lambda *a, observed=observed, **kw: (
                observed.append(os.environ.get(key)) or self.backend
            )
            with (
                self.subTest(device=device, gpu_count=gpu_count),
                patch.dict(os.environ),
                patch("faster_tts.memory.sys.platform", "linux"),
                patch("faster_tts.memory.available_cpu_memory", return_value=1e9),
                patch(
                    "faster_tts.engines.drama3.import_module",
                    return_value=Mock(
                        get_cuda_device_count=Mock(return_value=gpu_count)
                    ),
                ),
            ):
                os.environ.pop(key, None)
                Model(self.root, device=device)
                self.assertNotIn(key, os.environ)
            self.assertEqual(observed, [expected])

    def test_cache_budget_is_shared_by_workers_and_environment_restored(self):
        key = "CT2_DRAMA3_GPU_EXPERT_CACHE_GB"
        observed = []
        self.constructor.side_effect = lambda *a, **kw: (
            observed.append(os.environ[key]) or self.backend
        )
        with patch.dict(os.environ, {key: "3"}):
            model = Model(
                self.root,
                device="cuda",
                num_workers=2,
                engine_options={"gpu_expert_cache_gb": 12},
            )
            self.assertEqual(os.environ[key], "3")
        self.assertEqual(observed, ["6"])
        self.assertEqual(self.constructor.call_args.kwargs["inter_threads"], 2)
        self.assertEqual(model.info()["gpu_expert_cache_gb"], 12)
        self.assertEqual(model.info()["gpu_expert_cache_gb_per_worker"], 6)

    def test_auto_cache_budget_reserves_headroom(self):
        for free, expected in ((20e9, 14), (5e9, 3)):
            with (
                self.subTest(free=free),
                patch("faster_tts.memory.free_gpu_memory", return_value=free) as query,
            ):
                model = Model(
                    self.root,
                    device="cuda",
                    device_index=2,
                    num_workers=2,
                    engine_options={"gpu_expert_cache_gb": "auto"},
                )
                query.assert_called_once_with(2)
                self.assertEqual(model.info()["gpu_expert_cache_gb"], expected)
                self.assertEqual(
                    model.info()["gpu_expert_cache_gb_per_worker"], expected / 2
                )

    def test_cache_budget_validation_and_cpu_rejection(self):
        for value in (0, -1, True, "12", "bad", float("nan"), float("inf")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "gpu_expert_cache_gb"),
            ):
                Model(self.root, engine_options={"gpu_expert_cache_gb": value})
        with self.assertRaisesRegex(ValueError, "requires GPU"):
            Model(self.root, engine_options={"gpu_expert_cache_gb": "auto"})
        self.constructor.assert_not_called()

    def test_auto_cache_budget_fails_before_loading_if_memory_unavailable(self):
        with patch.dict(os.environ):
            os.environ.pop("CT2_DRAMA3_GPU_EXPERT_CACHE_GB", None)
            for options in ({}, {"gpu_expert_cache_gb": "auto"}):
                with patch("faster_tts.memory.free_gpu_memory", return_value=1e9):
                    with self.assertRaisesRegex(RuntimeError, "Insufficient"):
                        Model(self.root, device="cuda", engine_options=options)
                with patch(
                    "faster_tts.memory.free_gpu_memory",
                    side_effect=RuntimeError("query failed"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "query failed"):
                        Model(self.root, device="cuda", engine_options=options)
        self.constructor.assert_not_called()

    def test_cache_environment_restored_on_loading_failure(self):
        key = "CT2_DRAMA3_GPU_EXPERT_CACHE_GB"
        self.constructor.side_effect = RuntimeError("loading failed")
        with patch.dict(os.environ):
            os.environ.pop(key, None)
            with self.assertRaisesRegex(RuntimeError, "loading failed"):
                Model(
                    self.root,
                    device="cuda",
                    engine_options={"gpu_expert_cache_gb": 12},
                )
            self.assertNotIn(key, os.environ)

    def test_default_gpu_cache_is_automatic_for_explicit_and_auto_devices(self):
        key = "CT2_DRAMA3_GPU_EXPERT_CACHE_GB"
        for device in ("cuda", "auto"):
            with (
                self.subTest(device=device),
                patch.dict(os.environ),
                patch(
                    "faster_tts.engines.drama3.import_module",
                    return_value=Mock(get_cuda_device_count=Mock(return_value=1)),
                ),
            ):
                os.environ.pop(key, None)
                for options in ({}, {"gpu_expert_cache_gb": None}):
                    model = Model(
                        self.root, device=device, num_workers=2, engine_options=options
                    )
                    self.assertEqual(model.info()["gpu_expert_cache_gb"], 14)
                    self.assertEqual(model.info()["gpu_expert_cache_gb_per_worker"], 7)
                    self.assertNotIn(key, os.environ)

    def test_uncapped_gpu_cache_overrides_environment_without_querying_memory(self):
        key = "CT2_DRAMA3_GPU_EXPERT_CACHE_GB"
        observed = []
        self.constructor.side_effect = lambda *a, **kw: (
            observed.append(os.environ[key]) or self.backend
        )
        with (
            patch.dict(os.environ, {key: "3"}),
            patch("faster_tts.memory.free_gpu_memory", side_effect=AssertionError),
        ):
            model = Model(
                self.root,
                device="cuda",
                engine_options={"gpu_expert_cache_gb": "uncapped"},
            )
            self.assertEqual(model.info()["gpu_expert_cache_gb"], 0)
            self.assertEqual(observed, ["0"])
            self.assertEqual(os.environ[key], "3")

    def test_cache_environment_inherited_without_override(self):
        key = "CT2_DRAMA3_GPU_EXPERT_CACHE_GB"
        for value in ("3", "0"):
            with (
                patch.dict(os.environ, {key: value}),
                patch("faster_tts.memory.free_gpu_memory", side_effect=AssertionError),
            ):
                model = Model(self.root, device="cuda", num_workers=2)
                self.assertEqual(os.environ[key], value)
            self.assertEqual(model.info()["gpu_expert_cache_gb"], float(value) * 2)

    def test_cpu_threads_validation(self):
        for value in (0, -1, True, 1.5, 2**31):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "cpu_threads"),
            ):
                Model(self.root, cpu_threads=value)
        self.constructor.assert_not_called()

    def test_concurrent_calls_use_one_native_pool(self):
        barrier = Barrier(2)

        def generate(*args, **kwargs):
            barrier.wait(timeout=5)
            return self.result

        self.backend.generate.side_effect = generate
        model = Model(self.root, cpu_threads=2, num_workers=2)
        self.constructor.assert_called_once_with(
            str(self.root.resolve()),
            intra_threads=2,
            inter_threads=2,
            compute_type="default",
            device="cpu",
            device_index=0,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: model.synthesize("Hello"), range(4)))
        self.assertTrue(all(result.finish_reason == "eos" for result in results))
        self.assertEqual(model.info()["num_workers"], 2)
        self.assertEqual(self.backend.generate.call_count, 4)

    def test_native_pool_remains_usable_after_failure(self):
        model = Model(self.root)
        self.backend.generate.side_effect = [
            RuntimeError("native failure"),
            self.result,
        ]
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            model.synthesize("Hello")
        self.assertEqual(model.synthesize("Hello").finish_reason, "eos")
        self.assertEqual(self.constructor.call_count, 1)

    def test_typed_native_result_is_converted_and_validated(self):
        native_result = Mock()
        native_result.to_dict.return_value = dict(
            self.result, timings={"codec_seconds": 0.2}
        )
        self.backend.generate.return_value = native_result
        result = Model(self.root).synthesize("Hello")
        native_result.to_dict.assert_called_once_with()
        self.assertIs(result.audio, self.result["audio"])
        self.assertIs(result.codes, self.result["codes"])
        self.assertEqual(result.timings, {"codec_seconds": 0.2})
        native_result.to_dict.return_value = dict(
            self.result, audio=np.zeros(1, np.float32)
        )
        with self.assertRaisesRegex(RuntimeError, "audio length"):
            Model(self.root).synthesize("Hello")

    def test_num_workers_validation_before_loading(self):
        for value in (0, -1, True, 1.5, 2**31):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "num_workers"),
            ):
                Model(self.root, num_workers=value)
        self.constructor.assert_not_called()

    def test_device_selection_and_compute_type(self):
        model = Model(self.root, device="cuda", device_index=2, compute_type="float16")
        self.constructor.assert_called_once_with(
            str(self.root.resolve()),
            intra_threads=8,
            inter_threads=1,
            compute_type="float16",
            device="cuda",
            device_index=2,
        )
        self.assertEqual(model.info()["device"], "cuda")
        self.assertEqual(model.info()["device_index"], 2)
        self.assertEqual(model.info()["codec_device"], "cuda")

    def test_codec_device_honors_native_waveform_setting(self):
        for device in ("cpu", "cuda"):
            for setting in (None, "1", "true", "TRUE", "0", "false", ""):
                with (
                    self.subTest(device=device, setting=setting),
                    patch.dict(os.environ),
                ):
                    if setting is None:
                        os.environ.pop("CT2_DRAMA_GPU_WAVEFORM", None)
                    else:
                        os.environ["CT2_DRAMA_GPU_WAVEFORM"] = setting
                    model = Model(self.root, device=device)
                    expected = (
                        device if setting in (None, "1", "true", "TRUE") else "cpu"
                    )
                    self.assertEqual(model.info()["codec_device"], expected)
                    # Report the load-time setting even if the environment changes.
                    os.environ["CT2_DRAMA_GPU_WAVEFORM"] = "0"
                    self.assertEqual(model.info()["codec_device"], expected)

    def test_default_and_auto_select_gpu_when_available_and_cpu_otherwise(self):
        for count, expected in ((0, "cpu"), (2, "cuda")):
            with (
                self.subTest(count=count),
                patch(
                    "faster_tts.engines.drama3.import_module",
                    return_value=Mock(get_cuda_device_count=Mock(return_value=count)),
                ),
            ):
                for options in ({}, {"device": "auto"}):
                    self.constructor.reset_mock()
                    model = Model(self.root, **options)
                    self.assertEqual(
                        self.constructor.call_args.kwargs["device"], expected
                    )
                    self.assertEqual(model.info()["backend"], expected)
                    self.assertEqual(model.info()["codec_device"], expected)
                model = Model(self.root, device="cpu")
                self.assertEqual(self.constructor.call_args.kwargs["device"], "cpu")
                self.assertEqual(model.info()["backend"], "cpu")

    def test_invalid_devices_are_rejected_before_loading(self):
        for args in (
            {"device": "invalid"},
            {"device": None},
            {"device_index": -1},
            {"device_index": True},
            {"device_index": [0, 1]},
            {"device_index": 1.5},
            {"device_index": 2**31},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Model(self.root, **args)
        self.constructor.assert_not_called()

    def test_cli_device_and_compute_type(self):
        fake = Mock()
        fake.return_value.synthesize.return_value = SynthesisResult(
            np.zeros(2048, np.float32), 44100, np.zeros((1, 10), np.int32), "eos"
        )
        arguments = [
            "faster-tts",
            str(self.root),
            "--device",
            "cuda",
            "--device-index",
            "2",
            "--compute-type",
            "float16",
            "--gpu-expert-cache-gb",
            "auto",
            "--text",
            "Hello",
            "--output",
            str(self.root / "gpu.wav"),
        ]
        with (
            patch("faster_tts.__main__.Model", fake),
            patch.object(sys, "argv", arguments),
            patch("builtins.print"),
        ):
            main()
        self.assertEqual(fake.call_args.kwargs["device"], "cuda")
        self.assertEqual(fake.call_args.kwargs["device_index"], 2)
        self.assertEqual(fake.call_args.kwargs["compute_type"], "float16")
        self.assertEqual(
            fake.call_args.kwargs["engine_options"], {"gpu_expert_cache_gb": "auto"}
        )

    def test_multi_sentence_joins_outputs_and_conditions_later_sentence(self):
        model = Model(self.root)
        first = dict(
            self.result, audio=np.full(2048, 0.1, np.float32), timings={"native": 2.0}
        )
        second = dict(
            self.result, audio=np.full(2048, 0.2, np.float32), timings={"native": 3.0}
        )
        self.backend.generate.side_effect = [first, second]
        result = model.synthesize("Hello. Hello?", engine_options={"max_frames": 3})
        self.assertEqual(result.finish_reason, "eos")
        np.testing.assert_array_equal(
            result.audio, np.concatenate([first["audio"], second["audio"]])
        )
        self.assertEqual(result.codes.shape, (2, 10))
        self.assertEqual(result.timings, {"native": 5.0})
        calls = self.backend.generate.call_args_list
        self.assertEqual([call.kwargs["max_frames"] for call in calls], [3, 2])
        begin = model._engine._prompt.config["semantic_start_token_id"]
        self.assertEqual(np.count_nonzero(calls[1].args[0][:, 0] == begin), 1)
        self.assertEqual([a["chunk_index"] for a in result.attempts], [0, 1])
        self.assertEqual([a["text"] for a in result.attempts], ["Hello.", "Hello?"])

    def test_large_article_budget_respects_native_and_context_limits(self):
        for maximum in (8192, 32768):
            with self.subTest(maximum=maximum):
                model = Model(self.root)
                model._engine._prompt.config["text_config"][
                    "max_position_embeddings"
                ] = maximum
                self.backend.generate.reset_mock()
                result = model.synthesize("Hello. Hello?", max_duration_s=600)
                self.assertEqual(result.finish_reason, "eos")
                calls = self.backend.generate.call_args_list
                self.assertEqual(len(calls), 2)
                for call in calls:
                    self.assertEqual(
                        call.kwargs["max_frames"],
                        min(8192, maximum - len(call.args[0])),
                    )

    def test_chunk_budget_exhaustion_never_reports_full_success_or_skips_ahead(self):
        model = Model(self.root)
        result = model.synthesize(
            "Hello. Hello! Hello?", engine_options={"max_frames": 2}
        )
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.codes.shape[0], 2)
        self.assertEqual(self.backend.generate.call_count, 2)
        self.backend.generate.reset_mock()
        self.backend.generate.side_effect = [
            self.result,
            dict(self.result, finish_reason="length"),
        ]
        result = model.synthesize(
            "Hello. Hello! Hello?", engine_options={"max_frames": 4}
        )
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(self.backend.generate.call_count, 2)

    def test_chunk_history_is_bounded_and_not_retained_between_requests(self):
        model = Model(self.root)
        model.synthesize("Hello. Hello! Hello? Hello.")
        begin = model._engine._prompt.config["semantic_start_token_id"]
        self.assertEqual(
            [
                np.count_nonzero(c.args[0][:, 0] == begin)
                for c in self.backend.generate.call_args_list
            ],
            [0, 1, 2, 2],
        )
        self.backend.generate.reset_mock()
        model.synthesize("Hello. Hello!")
        self.assertEqual(
            np.count_nonzero(
                self.backend.generate.call_args_list[0].args[0][:, 0] == begin
            ),
            0,
        )

    def test_single_speaker_tag_survives_chunking_and_history(self):
        model = Model(self.root)
        result = model.synthesize("<|speaker:2|> Hello. Hello?")
        self.assertEqual(result.finish_reason, "eos")
        self.assertEqual(
            [a["text"] for a in result.attempts],
            ["<|speaker:2|> Hello.", "<|speaker:2|> Hello?"],
        )
        speaker = model._engine._prompt.tokenizer.token_to_id("<|speaker:2|>")
        default = model._engine._prompt.tokenizer.token_to_id("<|speaker:0|>")
        calls = self.backend.generate.call_args_list
        self.assertEqual(
            [np.count_nonzero(c.args[0][:, 0] == speaker) for c in calls], [1, 2]
        )
        self.assertTrue(all(not np.any(c.args[0][:, 0] == default) for c in calls))

    def test_split_opt_out_and_invalid_input_before_partial_generation(self):
        model = Model(self.root)
        model.synthesize("Hello. Hello!", split_sentences=False)
        self.assertEqual(self.backend.generate.call_count, 1)
        self.backend.generate.reset_mock()
        for kwargs in (
            {"text": "Hello. <|im_end|>"},
            {"text": "Hello.", "split_sentences": 1},
        ):
            with self.assertRaises(ValueError):
                model.synthesize(**kwargs)
        self.backend.generate.assert_not_called()

    def test_history_is_forwarded_as_audio_rows_without_changing_output_contract(self):
        model = Model(self.root)
        codes = np.zeros((2, 10), np.int32)
        result = model.synthesize(
            "Hello", engine_options={"max_frames": 1, "history": [("Hello", codes)]}
        )
        rows = self.backend.generate.call_args.args[0]
        begin = model._engine._prompt.config["semantic_start_token_id"]
        self.assertEqual(np.count_nonzero(rows[:, 0] == begin), 2)
        self.assertEqual(rows.dtype, np.int32)
        self.assertTrue(rows.flags.c_contiguous)
        self.assertEqual(result.finish_reason, "eos")
        self.backend.generate.reset_mock()
        with self.assertRaises(ValueError):
            model.synthesize(
                "Hello",
                engine_options={"history": [("Hello", np.zeros((1, 9), np.int32))]},
            )
        self.backend.generate.assert_not_called()

    def test_sampling_parameters_are_validated_and_forwarded_in_greedy_mode_too(self):
        model = Model(self.root)
        invalid = (
            {"temperature": 0},
            {"temperature": float("nan")},
            {"temperature": True},
            {"top_k": 0},
            {"top_k": 1.5},
            {"top_p": 0},
            {"top_p": 1.1},
            {"top_p": float("inf")},
            {"seed": -1},
            {"seed": 2**64},
            {"seed": True},
            {"greedy": 1},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model.synthesize("Hello", **kwargs)
        self.backend.generate.assert_not_called()
        model.synthesize(
            "Hello",
            engine_options={"max_frames": 1},
            temperature=0.6,
            top_k=7,
            top_p=0.95,
            seed=2**64 - 1,
            greedy=True,
        )
        self.assertEqual(
            self.backend.generate.call_args.kwargs,
            {
                "max_frames": 1,
                "prefill_chunk_size": 128,
                "temperature": 0.6,
                "top_k": 7,
                "top_p": 0.95,
                "seed": 2**64 - 1,
                "greedy": True,
            },
        )

    def test_model_explicit_remote_resolves_before_native_load(self):
        snapshot = Mock(return_value=str(self.root))
        hub = Mock(snapshot_download=snapshot)
        cache_dir = self.root / "hub-cache"
        with patch("faster_tts.cache.import_module", return_value=hub):
            model = Model(
                "hf://organization/converted-drama3",
                revision="pinned-commit",
                cache_dir=cache_dir,
                local_files_only=True,
            )
        self.assertEqual(
            snapshot.call_args.kwargs["repo_id"], "organization/converted-drama3"
        )
        self.assertEqual(snapshot.call_args.kwargs["revision"], "pinned-commit")
        self.assertEqual(snapshot.call_args.kwargs["cache_dir"], str(cache_dir))
        self.assertTrue(snapshot.call_args.kwargs["local_files_only"])
        self.constructor.assert_called_once_with(
            str(self.root.resolve()),
            intra_threads=8,
            inter_threads=1,
            compute_type="default",
            device="cpu",
            device_index=0,
        )
        self.backend.generate.assert_not_called()
        self.assertIsNotNone(model)

    def test_default_model_resolves_and_loads(self):
        snapshot = Mock(return_value=str(self.root))
        with patch(
            "faster_tts.cache.import_module",
            return_value=Mock(snapshot_download=snapshot),
        ):
            Model()
        self.assertEqual(snapshot.call_args.kwargs["repo_id"], "jordimas/drama-3-ct2")
        self.constructor.assert_called_once_with(
            str(self.root.resolve()),
            intra_threads=8,
            inter_threads=1,
            compute_type="default",
            device="cpu",
            device_index=0,
        )

    def test_cli_default_model(self):
        fake = Mock()
        fake.return_value.synthesize.return_value = SynthesisResult(
            np.zeros(2048, np.float32), 44100, np.zeros((1, 10), np.int32), "eos"
        )
        arguments = [
            "faster-tts",
            "--text",
            "Hello",
            "--output",
            str(self.root / "default.wav"),
        ]
        with (
            patch("faster_tts.__main__.Model", fake),
            patch.object(sys, "argv", arguments),
            patch("builtins.print"),
        ):
            main()
        self.assertEqual(fake.call_args.args, ("jordimas/drama-3-ct2",))

    def test_info_reports_actual_configuration_without_inference(self):
        model = Model(
            self.root,
            intra_threads=3,
            compute_type="float32",
            engine_options={"prefill_chunk_size": 2},
        )
        expected = {
            "engine": "drama3",
            "capabilities": (
                "synthesis",
                "history",
                "sentence_splitting",
                "retries",
                "gpu_expert_cache",
                "cpu_expert_mmap",
            ),
            "backend": "cpu",
            "codec_device": "cpu",
            "mode": "offline",
            "gpu_supported": True,
            "device": "cpu",
            "device_index": 0,
            "compute_type": "float32",
            "intra_threads": 3,
            "cpu_threads": 3,
            "num_workers": 1,
            "prefill_chunk_size": 2,
            "gpu_expert_cache_gb": None,
            "gpu_expert_cache_gb_per_worker": None,
            "model_path": str(self.root.resolve()),
            "source_revision": "source-commit",
            "quantization_policy": "all_matrices_int8",
            "matrix_dtype": "int8",
            "activation_dtype": "float32",
        }
        self.assertEqual(model.info(), expected)
        changed = model.info()
        changed["backend"] = "cuda"
        self.assertEqual(model.info()["backend"], "cpu")
        self.backend.generate.assert_not_called()

    def test_info_does_not_invent_unrecorded_quantization_policy(self):
        path = self.root / "config.json"
        metadata = json.loads(path.read_text())
        metadata.pop("quantization_policy")
        metadata.pop("source_revision")
        path.write_text(json.dumps(metadata))
        info = Model(self.root).info()
        self.assertIsNone(info["quantization_policy"])
        self.assertIsNone(info["source_revision"])
        self.backend.generate.assert_not_called()

    def test_validation_before_native_call(self):
        for args in (
            {"intra_threads": 0},
            {"intra_threads": True},
            {"engine_options": {"prefill_chunk_size": -1}},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Model(self.root, **args)
        with self.assertRaisesRegex(ValueError, "Unknown engine_options"):
            Model(self.root, engine_options={"not_a_real_option": 1})
        model = Model(self.root)
        for duration in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                model.synthesize("Hello", max_duration_s=duration)
        for frames in (0, -1, True, 1.5, 2**31):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                model.synthesize("Hello", engine_options={"max_frames": frames})
        with self.assertRaises(ValueError):
            model.synthesize("Hello", engine_options={"max_frames": 8192})
        with self.assertRaisesRegex(ValueError, "Unknown engine_options"):
            model.synthesize("Hello", engine_options={"not_a_real_option": 1})
        with self.assertRaises(ValueError):
            model.synthesize("<|im_end|>")
        self.backend.generate.assert_not_called()

    def test_missing_assets_and_missing_native_binding_are_actionable(self):
        with self.assertRaises(FileNotFoundError):
            Model(self.root / "absent")
        self.native_patch.stop()
        with patch(
            "faster_tts.engines.drama3.import_module",
            side_effect=ImportError("no extension"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, r"requires ctranslate2>=4\.8\.5 with Drama 3 support"
            ):
                Model(self.root)
        self.native_patch.start()

    def test_duration_budget_rounds_down_and_applies_across_chunks(self):
        model = Model(self.root)
        frame_seconds = 2048 / 44100
        for text in ("Hello", "Hello. Goodbye."):
            with self.subTest(text=text):
                self.backend.generate.reset_mock()
                result = model.synthesize(text, max_duration_s=2.5 * frame_seconds)
                self.assertLessEqual(
                    len(result.audio) / result.sample_rate, 2.5 * frame_seconds
                )
                calls = self.backend.generate.call_args_list
                self.assertEqual(calls[0].kwargs["max_frames"], 2)
                if len(calls) > 1:
                    self.assertEqual(calls[1].kwargs["max_frames"], 1)
        model.synthesize("Hello", max_duration_s=frame_seconds)
        self.assertEqual(self.backend.generate.call_args.kwargs["max_frames"], 1)
        self.backend.generate.reset_mock()
        for duration in (0.001, frame_seconds / 2):
            with (
                self.subTest(duration=duration),
                self.assertRaisesRegex(ValueError, "at least one codec frame"),
            ):
                model.synthesize("Hello", max_duration_s=duration)
        self.backend.generate.assert_not_called()
        model.synthesize(
            "Hello", max_duration_s=0.001, engine_options={"max_frames": 1}
        )
        self.assertEqual(self.backend.generate.call_args.kwargs["max_frames"], 1)

    def test_native_codes_must_fit_vocabulary_before_reuse_as_history(self):
        model = Model(self.root)
        size = model._engine._prompt.config["audio_decoder_config"]["vocab_size"]
        for code in (-1, size, np.iinfo(np.int32).max):
            for text in ("Hello", "Hello. Goodbye."):
                with self.subTest(code=code, text=text):
                    codes = np.zeros((1, 10), np.int32)
                    codes[0, -1] = code
                    self.backend.generate.return_value = dict(self.result, codes=codes)
                    self.backend.generate.reset_mock()
                    with self.assertRaisesRegex(
                        RuntimeError, "invalid audio/code metadata"
                    ):
                        model.synthesize(text)
                    self.assertEqual(self.backend.generate.call_count, 1)
        self.backend.generate.return_value = dict(
            self.result, codes=np.full((1, 10), size - 1, np.int32)
        )
        result = model.synthesize("Hello. Goodbye.")
        np.testing.assert_array_equal(
            result.codes, np.full((2, 10), size - 1, np.int32)
        )

    def test_native_results_reject_nan_wrong_shapes_and_inconsistent_length(self):
        model = Model(self.root)
        for update in (
            {"audio": np.full(2048, np.nan, np.float32)},
            {"audio": np.zeros((1, 2048), np.float32)},
            {"audio": np.zeros(2048, np.float64)},
            {"codes": np.zeros((1, 9), np.int32)},
            {"codes": np.zeros((1, 10), np.int64)},
            {"audio": np.zeros(2047, np.float32)},
            {"sample_rate": 22050},
            {"finish_reason": "unknown"},
        ):
            self.backend.generate.return_value = dict(self.result, **update)
            with self.subTest(update=list(update)), self.assertRaises(RuntimeError):
                model.synthesize("Hello", engine_options={"max_frames": 1})
        self.backend.generate.return_value = dict(self.result, finish_reason="length")
        self.assertEqual(
            model.synthesize("Hello", engine_options={"max_frames": 1}).finish_reason,
            "length",
        )
        self.backend.generate.return_value = {
            "audio": np.empty(0, np.float32),
            "codes": np.empty((0, 10), np.int32),
            "sample_rate": 44100,
            "finish_reason": "eos",
        }
        with self.assertRaisesRegex(RuntimeError, "no audio"):
            model.synthesize("Hello")

    def test_opt_in_retry_preserves_attempts_cost_prompt_and_seed_wrap(self):
        model = Model(self.root)
        first = dict(
            self.result, finish_reason="length", timings={"backbone": 2.0, "codec": 1.0}
        )
        second = dict(
            self.result,
            audio=np.ones(2048, np.float32),
            timings={"backbone": 3.0, "codec": 0.5},
        )
        self.backend.generate.side_effect = [first, second]
        with patch.object(
            model._engine._prompt, "prepare", wraps=model._engine._prompt.prepare
        ) as prepare:
            result = model.synthesize(
                "Hello",
                engine_options={"max_frames": 1},
                seed=2**64 - 1,
                max_retries=1,
            )
        prepare.assert_called_once_with("Hello", history=(), references=())
        calls = self.backend.generate.call_args_list
        self.assertIs(calls[0].args[0], calls[1].args[0])
        self.assertEqual([call.kwargs["seed"] for call in calls], [2**64 - 1, 0])
        self.assertEqual(result.finish_reason, "eos")
        np.testing.assert_array_equal(result.audio, second["audio"])
        self.assertEqual(result.timings, {"backbone": 5.0, "codec": 1.5})
        self.assertIsInstance(result.attempts, tuple)
        self.assertEqual(
            [a["finish_reason"] for a in result.attempts], ["length", "eos"]
        )
        self.assertEqual([a["seed"] for a in result.attempts], [2**64 - 1, 0])
        self.assertTrue(
            all(a["frames"] == 1 and a["seconds"] >= 0 for a in result.attempts)
        )
        self.assertTrue(
            all("audio" not in a and "codes" not in a for a in result.attempts)
        )

    def test_retry_is_bounded_returns_last_partial_and_default_or_greedy_skip(self):
        model = Model(self.root)
        self.backend.generate.return_value = dict(self.result, finish_reason="length")
        for kwargs in ({}, {"max_retries": 0}, {"max_retries": 1, "greedy": True}):
            self.backend.generate.reset_mock()
            result = model.synthesize(
                "Hello", engine_options={"max_frames": 1}, **kwargs
            )
            self.assertEqual(self.backend.generate.call_count, 1)
            self.assertEqual(len(result.attempts), 1)
        self.backend.generate.reset_mock()
        result = model.synthesize(
            "Hello", engine_options={"max_frames": 1}, seed=17, max_retries=1
        )
        self.assertEqual(self.backend.generate.call_count, 2)
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual([a["seed"] for a in result.attempts], [17, 18])
        self.backend.generate.reset_mock()
        self.backend.generate.return_value = self.result
        self.assertEqual(model.synthesize("Hello", max_retries=1).finish_reason, "eos")
        self.backend.generate.assert_called_once()

    def test_retry_validates_each_attempt_and_never_catches_native_errors(self):
        model = Model(self.root)
        for bad in (True, False, -1, 2, 1.0, None, "1"):
            with self.subTest(max_retries=bad), self.assertRaises(ValueError):
                model.synthesize("Hello", max_retries=bad)
        self.backend.generate.assert_not_called()
        for first in (
            dict(
                self.result,
                finish_reason="length",
                audio=np.full(2048, np.nan, np.float32),
            ),
            dict(self.result, finish_reason="length", timings={"codec": -1.0}),
            RuntimeError("native failure"),
        ):
            self.backend.generate.reset_mock()
            self.backend.generate.side_effect = [first, self.result]
            with self.subTest(first=type(first)), self.assertRaises(RuntimeError):
                model.synthesize(
                    "Hello", engine_options={"max_frames": 1}, max_retries=1
                )
            self.backend.generate.assert_called_once()
        self.backend.generate.reset_mock()
        self.backend.generate.side_effect = [
            dict(self.result, finish_reason="length"),
            dict(self.result, sample_rate=1),
        ]
        with self.assertRaises(RuntimeError):
            model.synthesize("Hello", engine_options={"max_frames": 1}, max_retries=1)
        self.assertEqual(self.backend.generate.call_count, 2)

    def test_cli_writes_mono_pcm16_with_clipping(self):
        output_path = self.root / "sample.wav"
        result = SynthesisResult(
            np.array([-2, -1, 0, 0.5, 1, 2], np.float32),
            44100,
            np.empty((0, 10), np.int32),
            "length",
        )
        fake = Mock()
        fake.return_value.synthesize.return_value = result
        arguments = [
            "faster-tts",
            str(self.root),
            "--text",
            "Hello",
            "--output",
            str(output_path),
        ]
        with (
            patch("faster_tts.__main__.Model", fake),
            patch.object(sys, "argv", arguments),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(SystemExit, "Incomplete synthesis"):
                main()
        with wave.open(str(output_path), "rb") as stream:
            self.assertEqual(
                (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()),
                (1, 2, 44100),
            )
            np.testing.assert_array_equal(
                np.frombuffer(stream.readframes(6), dtype="<i2"),
                [-32767, -32767, 0, 16384, 32767, 32767],
            )

    def test_cli_preserves_explicit_remote_identifier_and_cache_options(self):
        fake = Mock()
        fake.return_value.synthesize.return_value = SynthesisResult(
            np.zeros(2048, np.float32), 44100, np.zeros((1, 10), np.int32), "eos"
        )
        arguments = [
            "faster-tts",
            "hf://organization/converted-drama3",
            "--revision",
            "pinned-commit",
            "--cache-dir",
            str(self.root / "cache"),
            "--local-files-only",
            "--text",
            "Hello",
            "--output",
            str(self.root / "remote.wav"),
        ]
        with (
            patch("faster_tts.__main__.Model", fake),
            patch.object(sys, "argv", arguments),
            patch("builtins.print"),
        ):
            main()
        fake.assert_called_once_with(
            "hf://organization/converted-drama3",
            intra_threads=8,
            compute_type="default",
            device="cpu",
            device_index=0,
            revision="pinned-commit",
            cache_dir=self.root / "cache",
            local_files_only=True,
            engine_options=None,
        )
        fake.return_value.synthesize.assert_called_once_with(
            "Hello",
            max_duration_s=24.0,
            temperature=0.7,
            top_k=30,
            top_p=0.7,
            seed=0,
            greedy=False,
            split_sentences=True,
        )

    def test_rejects_zero_stride_native_audio_instead_of_repeating_first_sample(self):
        # Reproduces pybind11 2.11's rank-one allocation metadata under NumPy 2.
        # The underlying buffer can contain correct samples while its public
        # stride makes every element expose the first sample.
        model = Model(self.root)
        samples = np.arange(2048, dtype=np.float32)
        broken = np.lib.stride_tricks.as_strided(samples, shape=(2048,), strides=(0,))
        self.assertEqual(broken.strides, (0,))
        self.backend.generate.return_value = dict(self.result, audio=broken)
        with self.assertRaisesRegex(RuntimeError, "invalid audio/code metadata"):
            model.synthesize("Hello", engine_options={"max_frames": 1})

    def test_directed_events_use_empty_history_and_never_condition_speech(self):
        model = Model(self.root)
        event = dict(
            self.result,
            audio=np.full(2048, 0.1, np.float32),
            codes=np.full((1, 10), 3, np.int32),
            finish_reason="length",
        )
        self.backend.generate.side_effect = [self.result, event, self.result]
        result = model.synthesize(
            "Hello. [breathing] Hello!",
            engine_options={"directed_spans": True, "max_frames": 600},
        )
        calls = self.backend.generate.call_args_list
        begin = model._engine._prompt.config["semantic_start_token_id"]
        self.assertEqual(
            [np.count_nonzero(c.args[0][:, 0] == begin) for c in calls], [0, 0, 1]
        )
        self.assertFalse(np.any(calls[2].args[0][:, 1:] == 3))
        self.assertEqual([c.kwargs["max_frames"] for c in calls], [256, 22, 256])
        self.assertEqual(result.finish_reason, "eos")
        self.assertEqual(
            [a["segment_kind"] for a in result.attempts], ["speech", "event", "speech"]
        )
        self.assertEqual(result.attempts[1]["finish_reason"], "length")
        self.assertTrue(result.attempts[1]["event_length_cap_explicit"])
        self.assertEqual(result.codes.shape, (3, 10))

    def test_directed_budget_cannot_skip_an_event_or_a_speech_segment(self):
        model = Model(self.root)
        result = model.synthesize(
            "Hello. [breathing] Hello!",
            engine_options={"directed_spans": True, "max_frames": 22},
        )
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(self.backend.generate.call_count, 1)
        self.backend.generate.reset_mock()
        result = model.synthesize(
            "Hello. Hello!", engine_options={"directed_spans": True, "max_frames": 1}
        )
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(len(result.codes), 1)
        self.backend.generate.reset_mock()
        self.backend.generate.return_value = dict(self.result, finish_reason="length")
        result = model.synthesize(
            "Hello. Hello!", engine_options={"directed_spans": True}
        )
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(self.backend.generate.call_count, 1)

    def test_directed_rejects_silent_events_and_validates_before_generation(self):
        model = Model(self.root)
        with self.assertRaisesRegex(RuntimeError, "silent audio"):
            model.synthesize(
                "Hello. [breathing] Hello!", engine_options={"directed_spans": True}
            )
        self.backend.generate.reset_mock()
        for kw in (
            {"engine_options": {"directed_spans": 1}},
            {"engine_options": {"directed_spans": True}, "split_sentences": False},
        ):
            with self.assertRaises(ValueError):
                model.synthesize("Hello.", **kw)
        with self.assertRaises(ValueError):
            model.synthesize(
                "Hello. <|im_end|>", engine_options={"directed_spans": True}
            )
        self.backend.generate.assert_not_called()

    def test_directed_speaker_turns_retain_tags_and_history_is_request_local(self):
        model = Model(self.root)
        result = model.synthesize(
            "<|speaker:0|> Hello. <|speaker:1|> Hello!",
            engine_options={"directed_spans": True},
        )
        self.assertEqual(
            [a["text"] for a in result.attempts],
            ["<|speaker:0|> Hello.", "<|speaker:1|> Hello!"],
        )
        self.backend.generate.reset_mock()
        model.synthesize("Hello.", engine_options={"directed_spans": True})
        begin = model._engine._prompt.config["semantic_start_token_id"]
        self.assertFalse(np.any(self.backend.generate.call_args.args[0][:, 0] == begin))

    def test_cli_directed_spans_forwarding_and_conflict_before_load(self):
        fake = Mock()
        fake.return_value.synthesize.return_value = SynthesisResult(
            audio=np.zeros(2048, np.float32),
            sample_rate=44100,
            codes=np.zeros((1, 10), np.int32),
            finish_reason="eos",
        )
        argv = [
            "faster-tts",
            "--text",
            "Hello.",
            "--output",
            str(self.root / "out.wav"),
            "--directed-spans",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("faster_tts.__main__.Model", fake),
            patch("builtins.print"),
        ):
            main()
        self.assertEqual(
            fake.return_value.synthesize.call_args.kwargs["engine_options"],
            {"directed_spans": True},
        )
        fake.reset_mock()
        with (
            patch.object(sys, "argv", argv + ["--no-split-sentences"]),
            patch("faster_tts.__main__.Model", fake),
            patch("sys.stderr"),
        ):
            with self.assertRaises(SystemExit):
                main()
        fake.assert_not_called()


if __name__ == "__main__":
    unittest.main()
