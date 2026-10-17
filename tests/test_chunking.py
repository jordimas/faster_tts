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
        for text in ('<|speaker:2|> Hello. <|speaker:3|> Goodbye.', '[whisper] Hello. Goodbye.',
                     '<whisper>Hello. Goodbye.</whisper>'):
            self.assertEqual(split_text(text), [text])

    def test_single_leading_speaker_is_preserved_on_every_chunk(self):
        for speaker in (0, 2, 15):
            tag = f'<|speaker:{speaker}|>'
            self.assertEqual(split_text('  ' + tag + ' Hello. Goodbye.'),
                             [tag + ' Hello.', tag + ' Goodbye.'])
            text = '\t' + tag + ' Hello.'
            self.assertEqual(split_text(text), [text])
        for text in ('<|speaker:02|> Hello. Goodbye.', '<|speaker:16|> Hello. Goodbye.',
                     '<|speaker:2|> [whisper] Hello. Goodbye.'):
            self.assertEqual(split_text(text), [text])
