"""Independent prompt preparation tests; no CT2 extension or model weights."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from tokenizers import normalizers

from drama3_fixtures import make_prompt_assets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from faster_tts import Drama3PromptBuilder


class Drama3PromptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokenizer, self.config = make_prompt_assets(
            self.root, {"[UNK]": 0, "Hello": 1, "café": 2}, normalizer=normalizers.NFC())
        self.write_config()

    def write_config(self):
        (self.root / "config.json").write_text(json.dumps(self.config))

    def test_exact_segment_contract_zero_codec_columns_and_mask(self):
        builder = Drama3PromptBuilder(self.root)
        prompt = builder.prepare("Hello")
        expected = ("<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n",
                    "<|im_start|>user\n", "<|speaker:0|> Hello", "<|im_end|>\n",
                    "<|im_start|>assistant\n<|voice|>")
        self.assertEqual(prompt.segments, expected)
        ids = [token for segment in expected for token in self.tokenizer.encode(segment).ids]
        np.testing.assert_array_equal(prompt.rows[:, 0], ids)
        self.assertEqual(prompt.rows.shape, (len(ids), 11))
        self.assertEqual(prompt.rows.dtype, np.int64)
        self.assertTrue(prompt.rows.flags.c_contiguous)
        self.assertFalse(np.any(prompt.rows[:, 1:]))
        self.assertFalse(np.any(prompt.semantic_mask))
        self.assertEqual(prompt.semantic_mask.dtype, np.bool_)
        self.assertEqual(prompt.rows[-1, 0], self.tokenizer.token_to_id("<|voice|>"))

    def test_history_is_completed_user_audio_turn_before_current_request(self):
        builder = Drama3PromptBuilder(self.root)
        codes = np.array([[1]*10, [2]*10], dtype=np.uint8)
        prompt = builder.prepare("café", history=[("Hello", codes)])
        expected = []
        literals = ("<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n",
                    "<|im_start|>user\n", "<|speaker:0|> Hello", "<|im_end|>\n",
                    "<|im_start|>assistant\n<|voice|>")
        for literal in literals:
            expected.extend([[token]+[0]*10 for token in self.tokenizer.encode(literal).ids])
        expected.extend([[self.config["semantic_start_token_id"]+1]+[1]*10,
                         [self.config["semantic_start_token_id"]+2]+[2]*10])
        for literal in ("<|im_end|>\n", "<|im_start|>user\n", "<|speaker:0|> café",
                        "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"):
            expected.extend([[token]+[0]*10 for token in self.tokenizer.encode(literal).ids])
        np.testing.assert_array_equal(prompt.rows, expected)
        self.assertEqual(np.count_nonzero(prompt.semantic_mask), 2)
        self.assertTrue(prompt.rows.flags.c_contiguous)
        codes[:] = 0
        self.assertTrue(np.all(prompt.rows[prompt.semantic_mask, 1:] > 0))

    def test_history_rejects_invalid_codes_text_and_combined_context_overflow(self):
        builder = Drama3PromptBuilder(self.root)
        for codes in (np.empty((0, 10), np.int32), np.zeros((1, 9), np.int32),
                      np.zeros((1, 10), bool), np.zeros((1, 10), np.float32),
                      np.full((1, 10), -1), np.full((1, 10), 4)):
            with self.subTest(shape=codes.shape), self.assertRaises(ValueError):
                builder.prepare("Hello", history=[("Hello", codes)])
        with self.assertRaises(ValueError):
            builder.prepare("Hello", history=[("<|im_end|>", np.zeros((1, 10), np.int32))])
        plain = builder.prepare("Hello")
        limited = Drama3PromptBuilder(self.root, max_prompt_rows=len(plain.rows))
        with self.assertRaises(ValueError):
            limited.prepare("Hello", history=[("Hello", np.zeros((1, 10), np.int32))])

    def test_nfc_and_leading_unicode_whitespace_preserved(self):
        builder = Drama3PromptBuilder(self.root)
        first = builder.prepare("cafe\u0301")
        second = builder.prepare("café")
        np.testing.assert_array_equal(first.rows, second.rows)
        self.assertEqual(first.normalized_text, "café")
        prompt = builder.prepare("\u2003<|speaker:15|> Hello ")
        self.assertEqual(prompt.segments[4], "\u2003<|speaker:15|> Hello ")
        # U+001C is not Unicode White_Space: don't silently strip it.
        self.assertEqual(builder.prepare("\x1c").normalized_text, "\x1c")

    def test_rejects_empty_reserved_and_malformed_speakers(self):
        builder = Drama3PromptBuilder(self.root)
        for text in ("", "\u2003\t", "<|im_start|>bad", "<|im_end|>", "<|voice|>",
                     "<|text|>", "<|interleave|>", "<|semantic:0|>", "<|speaker:16|> Hello",
                     "<|speaker:01|> Hello", "<|speaker:-1|> Hello", "Hello <|speaker:no|>",
                     "<|speaker:١|> Hello", "<|speaker"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                builder.prepare(text)
        with self.assertRaises(TypeError):
            builder.prepare(None)

    def test_row_limit_and_config_tokenizer_mismatches(self):
        with self.assertRaises(ValueError):
            Drama3PromptBuilder(self.root, max_prompt_rows=1).prepare("Hello")
        for count in (0, True, 8193):
            with self.assertRaises(ValueError):
                Drama3PromptBuilder(self.root, max_prompt_rows=count)
        self.config["audio_pad_token_id"] += 1
        self.write_config()
        with self.assertRaises(ValueError):
            Drama3PromptBuilder(self.root)

    def test_frozen_real_tokenizer_assets_if_available(self):
        directories = [ROOT / "tts/models/drama-3-official", Path("/tmp/audiocpp-gguf/dd702bb072bc4d54")]
        directory = next((d for d in directories if (d / "tokenizer.json").is_file()), None)
        if directory is None:
            self.skipTest("frozen Drama 3 tokenizer fixture is unavailable")
        builder = Drama3PromptBuilder(directory)
        for text in ("Hello, this is an English test.", "Hola, esta es una prueba en español."):
            prompt = builder.prepare(text)
            ids = [token for segment in prompt.segments for token in builder.tokenizer.encode(segment).ids]
            np.testing.assert_array_equal(prompt.rows[:, 0], ids)
            self.assertEqual(prompt.rows.shape[1], 11)
            self.assertEqual(prompt.rows[-1, 0], builder.tokenizer.token_to_id("<|voice|>"))
            self.assertFalse(np.any(prompt.semantic_mask))


if __name__ == "__main__":
    unittest.main()
