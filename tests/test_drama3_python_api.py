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
        options = {"gpu_expert_cache_gb": "auto"}
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

    def test_cache_environment_inherited_without_override(self):
        key = "CT2_DRAMA3_GPU_EXPERT_CACHE_GB"
        with patch.dict(os.environ, {key: "3"}):
            model = Model(self.root, device="cuda", num_workers=2)
            self.assertEqual(os.environ[key], "3")
        self.assertEqual(model.info()["gpu_expert_cache_gb"], 6)

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
        self.assertEqual(model.info()["codec_device"], "cpu")

    def test_auto_selects_gpu_when_available_and_cpu_otherwise(self):
        for count, expected in ((0, "cpu"), (2, "cuda")):
            with (
                self.subTest(count=count),
                patch(
                    "faster_tts.engines.drama3.import_module",
                    return_value=Mock(get_cuda_device_count=Mock(return_value=count)),
                ),
            ):
                self.constructor.reset_mock()
                model = Model(self.root, device="auto")
                self.assertEqual(self.constructor.call_args.kwargs["device"], expected)
                self.assertEqual(model.info()["backend"], expected)

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
                "prefill_chunk_size": 16,
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
                RuntimeError, "experimental CTranslate2 Drama3 binding"
            ):
                Model(self.root)
        self.native_patch.start()

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
        prepare.assert_called_once_with("Hello", history=())
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


if __name__ == "__main__":
    unittest.main()
