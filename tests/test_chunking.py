"""Sentence-boundary regressions, independent of model inference."""
import unittest
from faster_tts.chunking import split_text


class ChunkingTest(unittest.TestCase):
    def test_plain_sentences_quotes_newlines_and_unicode(self):
        cases = {
            'Wait! The light is green now; we can cross.': ['Wait!', 'The light is green now; we can cross.'],
            '¡Espera! Ya podemos cruzar.': ['¡Espera!', 'Ya podemos cruzar.'],
            'She said “hello.” Then she left.': ['She said “hello.”', 'Then she left.'],
            'First line\nSecond line': ['First line', 'Second line'],
            '  Café.\n\n Otra frase.  ': ['Café.', 'Otra frase.'],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(split_text(text), expected)

    def test_abbreviations_initials_and_decimals_are_not_sentence_boundaries(self):
        for text in ('Dr. Smith paid 3.14 dollars. Next sentence.',
                     'La Dra. Pérez llegó. Otra frase.',
                     'A. Smith lives in the U.S. today. Next sentence.'):
            chunks = split_text(text)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(' '.join(chunks), text)

    def test_markup_speakers_and_cues_preserve_the_original_request(self):
        for text in ('<|speaker:2|> Hello. Goodbye.', '[whisper] Hello. Goodbye.',
                     '<whisper>Hello. Goodbye.</whisper>'):
            self.assertEqual(split_text(text), [text])
