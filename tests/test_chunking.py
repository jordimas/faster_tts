"""Sentence-boundary regressions, independent of model inference."""

import unittest

from faster_tts.chunking import split_text


class ChunkingTest(unittest.TestCase):
    def test_unicode_sentence_boundaries_with_and_without_spaces(self):
        cases = (
            ("你好。世界！再见？", ["你好。", "世界！", "再见？"]),
            ("「こんにちは。」次です。", ["「こんにちは。」", "次です。"]),
            ("안녕하세요! 다음 문장입니다.", ["안녕하세요!", "다음 문장입니다."]),
            ("مرحبا؟ نعم!", ["مرحبا؟", "نعم!"]),
            ("یہ پہلا جملہ ہے۔ یہ دوسرا ہے۔", ["یہ پہلا جملہ ہے۔", "یہ دوسرا ہے۔"]),
            ("नमस्ते। फिर मिलेंगे॥", ["नमस्ते।", "फिर मिलेंगे॥"]),
            ("မင်္ဂလာပါ။နောက်တစ်ကြောင်း။", ["မင်္ဂလာပါ။", "နောက်တစ်ကြောင်း။"]),
            ("Բարեւ։ Հաջորդը։", ["Բարեւ։", "Հաջորդը։"]),
            ("ሰላም። እንዴት ነህ፧", ["ሰላም።", "እንዴት ነህ፧"]),
            ("Привет. До свидания.", ["Привет.", "До свидания."]),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(split_text(text), expected)

    def test_multilingual_abbreviations_and_unicode_initials(self):
        for first in (
            "Mme. Dupont arrive.",
            "Das ist z.B. richtig.",
            "El núm. és correcte.",
            "Ж. Иванов пришёл.",
            "Он живёт на ул. Ленина.",
        ):
            with self.subTest(first=first):
                self.assertEqual(
                    split_text(first + " Next sentence."), [first, "Next sentence."]
                )

    def test_unicode_clauses_preserve_every_character(self):
        for clause in (
            "这是一个很长的分句，",
            "هذه عبارة طويلة،",
            "これは長い節です；",
        ):
            text = clause * 40
            chunks = split_text(text)
            self.assertGreater(len(chunks), 1)
            self.assertEqual("".join(chunks), text)
            self.assertTrue(all(chunk for chunk in chunks))

    def test_unpunctuated_scripts_remain_intact(self):
        for text in ("สวัสดีทุกคนยินดีต้อนรับ", "שלום לכולם", "ខ្ញុំសូមស្វាគមន៍"):
            self.assertEqual(split_text(text), [text])

    def test_unicode_speaker_tags_and_cues(self):
        self.assertEqual(
            split_text("<|speaker:2|> 你好。再见。"),
            ["<|speaker:2|> 你好。", "<|speaker:2|> 再见。"],
        )
        self.assertEqual(
            split_text("[轻声] 你好。[悲伤] 再见。"), ["[轻声] 你好。", "[悲伤] 再见。"]
        )

    def test_long_sentence_groups_clauses_without_changing_spoken_text(self):
        first = "The visitor waited quietly beside the door, "
        second = "watching the clouds move slowly across the sky; "
        text = first * 3 + second * 3 + "until the rain finally stopped."
        chunks = split_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 200 for chunk in chunks))
        for chunk in chunks[:-1]:
            self.assertIn(chunk[-1], ",;:")

    def test_clause_target_is_soft_and_never_splits_arbitrary_words_or_markup(self):
        text = "word " * 60 + "end."
        self.assertEqual(split_text(text), [text])
        for text in (
            "[whisper] " + "a clause, " * 30 + "end.",
            "<|speaker:2|> " + "a clause, " * 30 + "<|speaker:3|> End.",
        ):
            self.assertEqual(split_text(text), [text])
        short = "First clause,   another clause; the final clause."
        self.assertEqual(split_text(short), [short])

    def test_plain_sentences_quotes_newlines_and_unicode(self):
        cases = {
            "Wait! The light is green now; we can cross.": [
                "Wait!",
                "The light is green now; we can cross.",
            ],
            "¡Espera! Ya podemos cruzar.": ["¡Espera!", "Ya podemos cruzar."],
            "She said “hello.” Then she left.": ["She said “hello.”", "Then she left."],
            "First line\nSecond line": ["First line", "Second line"],
            "  Café.\n\n Otra frase.  ": ["Café.", "Otra frase."],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(split_text(text), expected)

    def test_abbreviations_initials_and_decimals_are_not_sentence_boundaries(self):
        for text in (
            "Dr. Smith paid 3.14 dollars. Next sentence.",
            "La Dra. Pérez llegó. Otra frase.",
            "A. Smith lives in the U.S. today. Next sentence.",
        ):
            chunks = split_text(text)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(" ".join(chunks), text)

    def test_markup_speakers_and_cues_preserve_the_original_request(self):
        for text in (
            "<|speaker:2|> Hello. <|speaker:3|> Goodbye.",
            "[whisper] Hello. Goodbye.",
            "<whisper>Hello. Goodbye.</whisper>",
        ):
            self.assertEqual(split_text(text), [text])

    def test_each_sentence_keeps_its_own_emotional_cue(self):
        first = "[amb un nus a la gola, plorant en veu baixa] He deixat la seva jaqueta al costat de la porta."
        second = "[amb la veu trencada] No paro de pensar que entrarà i em demanarà la jaqueta."
        self.assertEqual(split_text(first + " " + second), [first, second])
        first = "[quietly. Pause! Then whisper?] Hello."
        second = "[sad; hesitant, but clear] Goodbye."
        self.assertEqual(split_text(first + "\n" + second), [first, second])
        self.assertEqual(
            split_text("[softly] Dr. Smith arrived. [loudly] He left."),
            ["[softly] Dr. Smith arrived.", "[loudly] He left."],
        )

    def test_uncertain_cue_scope_and_malformed_markup_stay_intact(self):
        for text in (
            "[whisper] Hello. Goodbye.",
            "Hello. [whisper] Goodbye.",
            "[sad] Hello [whisper] there. [sad] Goodbye.",
            "[sad [whisper]] Hello. [sad] Goodbye.",
            "[sad Hello. [whisper] Goodbye.",
            "[sad] Hello. [whisper Goodbye.",
            "[] Hello. [sad] Goodbye.",
            "[sad] Hello. [sad]",
            "<|speaker:2|> [sad] Hello. <|speaker:3|> [sad] Goodbye.",
            "<|speaker:2|> [sad] Hello. [sad] Goodbye.",
        ):
            with self.subTest(text=text):
                self.assertEqual(split_text(text), [text])

    def test_single_leading_speaker_is_preserved_on_every_chunk(self):
        for speaker in (0, 2, 15):
            tag = f"<|speaker:{speaker}|>"
            self.assertEqual(
                split_text("  " + tag + " Hello. Goodbye."),
                [tag + " Hello.", tag + " Goodbye."],
            )
            text = "\t" + tag + " Hello."
            self.assertEqual(split_text(text), [text])
        for text in (
            "<|speaker:02|> Hello. Goodbye.",
            "<|speaker:16|> Hello. Goodbye.",
            "<|speaker:2|> [whisper] Hello. Goodbye.",
        ):
            self.assertEqual(split_text(text), [text])
