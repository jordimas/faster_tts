"""Audio preparation without native model weights."""

import io
import unittest
from unittest.mock import patch

import numpy as np

from faster_tts.references import load_references


class ReferenceAudioTest(unittest.TestCase):
    def test_files_and_bytes_mix_and_resample(self):
        import soundfile as sf

        samples = np.column_stack((np.full(2400, 0.5), np.zeros(2400)))
        audio = io.BytesIO()
        sf.write(audio, samples, 24000, format="WAV", subtype="FLOAT")
        references = load_references(
            [{"audio": audio.getvalue(), "text": "Hello", "speaker": 3}]
        )
        text, waveform = references[0]
        self.assertEqual(text, "<|speaker:3|> Hello")
        self.assertEqual(waveform.shape, (4410,))
        self.assertEqual(waveform.dtype, np.float32)
        self.assertTrue(waveform.flags.c_contiguous)
        np.testing.assert_allclose(waveform[100:-100], 0.25, atol=0.001)
        with patch(
            "soundfile.read", return_value=(samples.astype(np.float32), 44100)
        ) as read:
            text, waveform = load_references(
                [{"audio": "voice.wav", "text": "<|speaker:2|> Hello", "speaker": 3}]
            )[0]
        self.assertEqual(read.call_args.args[0], "voice.wav")
        self.assertEqual(text, "<|speaker:2|> Hello")
        np.testing.assert_allclose(waveform, 0.25)

    def test_invalid_references(self):
        for entry in (
            {"audio": "voice.wav", "text": ""},
            {"text": "Hello"},
            {"audio": "voice.wav", "text": "Hello", "speaker": True},
            {"audio": "voice.wav", "text": "Hello", "speaker": 16},
            {"audio": "voice.wav", "text": "Hello", "typo": 1},
        ):
            with self.subTest(entry=entry), self.assertRaises((ValueError, TypeError)):
                load_references([entry])

    def test_empty_and_nonfinite_audio(self):
        for waveform in (
            np.empty((0, 1), np.float32),
            np.array([[np.nan]], np.float32),
        ):
            with patch("soundfile.read", return_value=(waveform, 44100)):
                with self.assertRaisesRegex(ValueError, "nonempty and finite"):
                    load_references([{"audio": "voice.wav", "text": "Hello"}])
