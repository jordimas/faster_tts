"""Drama 3 prompt preparation without a model runtime.

Protocol evidence: Fish Speech's fish_omni/generate.py build_drama3_prompt:
https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/fish_omni/generate.py
The checkpoint's generic chat_template.jinja is NOT the TTS prompt builder.
Literals are encoded separately; concatenating them before encoding changes BPE
boundaries. Optional history contains completed text/audio turns; reference voice
encoding and waveform synthesis are outside this module.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

_WHITE_SPACE = "".join(
    chr(value)
    for value in list(range(9, 14))
    + [32, 133, 160, 5760]
    + list(range(8192, 8203))
    + [8232, 8233, 8239, 8287, 12288]
)
_SPEAKER = re.compile(r"<\|speaker:(?:[0-9]|1[0-5])\|>")
_RESERVED = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
)


@dataclass(frozen=True)
class Drama3Prompt:
    """CPU contiguous int64 rows [N,11], plus a bool semantic mask [N].

    Column zero holds tokenizer IDs; columns 1..10 are zero for text rows, NOT
    audio_pad_token_id. Audio rows would hold ten codec indices and an offset
    semantic ID. semantic_mask identifies audio rows supplied through history.
    audio_pad_token_id names a tokenizer special token;
    it is exposed as metadata and is not the filler for codec columns.

    Attributes:
        rows: Contiguous int64 prompt array of shape `(N, 11)` on CPU.
        semantic_mask: Boolean array of shape `(N,)` identifying history audio rows.
        normalized_text: Current input text normalized to Unicode NFC, before
            any default speaker tag is added.
        segments: Ordered protocol strings and history codec arrays used to
            construct the rows. Strings are encoded separately to preserve BPE boundaries.
        semantic_start_token_id: First tokenizer ID representing a semantic code.
        semantic_end_token_id: Last tokenizer ID representing a semantic code.
        eos_token_id: Tokenizer ID for the end-of-turn token.
        audio_pad_token_id: Tokenizer audio-padding ID; not a codec-column filler.
    """

    rows: np.ndarray
    semantic_mask: np.ndarray
    normalized_text: str
    segments: tuple
    semantic_start_token_id: int
    semantic_end_token_id: int
    eos_token_id: int
    audio_pad_token_id: int


class Drama3PromptBuilder:
    """Prepare prompts from local tokenizer/config files without loading a model.

    Most applications only need `faster_tts.Model`. Use this class to inspect
    prompt rows or prepare inputs for a native Drama 3 runtime.

    Args:
        checkpoint_dir: Local directory containing `tokenizer.json` and
            `config.json`, either a source fish_omni configuration or a converted
            configuration wrapping it in `source_config`. No downloads occur.
        max_prompt_rows: Maximum combined text/history rows, in `1..8192`.
            Defaults to 8192. This does not reserve space for generated frames.

    Raises:
        ValueError: Invalid row limit, incompatible configuration, or tokenizer mismatch.
        OSError: Required checkpoint files cannot be read.
    """

    def __init__(self, checkpoint_dir, max_prompt_rows=8192):
        if (
            not isinstance(max_prompt_rows, int)
            or isinstance(max_prompt_rows, bool)
            or not 1 <= max_prompt_rows <= 8192
        ):
            raise ValueError("max_prompt_rows must be an integer in 1..8192")
        self.max_prompt_rows = max_prompt_rows
        directory = Path(checkpoint_dir)
        self.config = json.loads(
            (directory / "config.json").read_text(encoding="utf-8")
        )
        if isinstance(self.config, dict) and "source_config" in self.config:
            self.config = self.config["source_config"]
        if (
            not isinstance(self.config, dict)
            or self.config.get("model_type") != "fish_omni"
        ):
            raise ValueError("expected a fish_omni checkpoint")
        audio = self.config.get("audio_decoder_config", {})
        if not isinstance(audio, dict) or audio.get("num_codebooks") != 10:
            raise ValueError("plain Drama 3 prompt requires ten codebooks")
        size = audio.get("vocab_size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("invalid audio codebook vocabulary size")
        for field in (
            "semantic_start_token_id",
            "semantic_end_token_id",
            "eos_token_id",
            "audio_pad_token_id",
        ):
            value = self.config.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"missing or invalid {field}")
        if (
            self.config["semantic_end_token_id"]
            - self.config["semantic_start_token_id"]
            + 1
            != size
        ):
            raise ValueError("semantic ID range disagrees with codebook size")
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        expected = {
            "<|im_end|>": self.config["eos_token_id"],
            "<|audio_pad|>": self.config["audio_pad_token_id"],
        }
        expected.update(
            {
                f"<|semantic:{index}|>": self.config["semantic_start_token_id"] + index
                for index in range(size)
            }
        )
        for literal, token_id in expected.items():
            if self.tokenizer.token_to_id(literal) != token_id:
                raise ValueError(f"tokenizer/config mismatch: {literal}")
        for literal in ("<|im_start|>", "<|voice|>") + tuple(
            f"<|speaker:{i}|>" for i in range(16)
        ):
            if self.tokenizer.token_to_id(literal) is None:
                raise ValueError(f"missing tokenizer special token: {literal}")

    def prepare(self, text, *, history=(), references=()):
        """Prepare text with optional (text, integer codes[frames,10]) history.

        Include only completed generations. History is conditioning context,
        not output audio, and counts towards the prompt/context limit.

        Args:
            text: Nonempty text normalized to Unicode NFC. Canonical speaker tags
                with IDs 0 through 15 are accepted; other protocol tags are rejected.
                A default speaker-0 tag is added if no leading speaker tag exists.
            references: Iterable of `(transcript, codes)` voice examples placed
                in the system message, separate from completed history turns.
                Codes follow the same shape and vocabulary rules as history.
            history: Iterable of completed `(text, codes)` turns. Each code array
                must be nonempty, integer, shape `(frames, 10)`, and contain only
                indices within the configured vocabulary.

        Returns:
            Drama3Prompt: Encoded rows, semantic mask, normalized text, and metadata.

        Raises:
            TypeError: Text is not a string.
            ValueError: Empty text, forbidden tags, invalid history codes, or a
                combined prompt exceeding `max_prompt_rows`.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        text = unicodedata.normalize("NFC", text)
        if not text.strip(_WHITE_SPACE):
            raise ValueError("text must contain non-whitespace characters")
        if any(marker in text for marker in _RESERVED):
            raise ValueError("text contains reserved prompt structure")
        for marker in re.finditer(re.escape("<|speaker"), text):
            if _SPEAKER.match(text, marker.start()) is None:
                raise ValueError("speaker tags must use canonical IDs 0..15")
        spoken = (
            text
            if _SPEAKER.match(text.lstrip(_WHITE_SPACE))
            else "<|speaker:0|> " + text
        )
        segments = (
            "<|im_start|>system\n",
            "convert the provided text to speech",
            "<|im_end|>\n",
            "<|im_start|>user\n",
            spoken,
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        )
        reference_segments = []
        for index, (transcript, codes) in enumerate(references, 1):
            validated = self.prepare(transcript, history=[(transcript, codes)])
            spoken_reference = validated.segments[4].lstrip(_WHITE_SPACE)
            reference_segments.extend(
                (
                    f"\n\n[Reference {index}]\nText: {spoken_reference}\nSpeech: ",
                    np.asarray(codes, dtype=np.int64),
                )
            )
        if reference_segments:
            segments = (
                segments[0],
                "convert the provided text to speech reference to the following:",
                *reference_segments,
                *segments[2:],
            )
        system_end = 3 + len(reference_segments)
        turns = []
        for previous_text, codes in history:
            previous = self.prepare(previous_text)
            codes = np.asarray(codes)
            audio = self.config["audio_decoder_config"]
            if (
                codes.ndim != 2
                or codes.shape[1] != audio["num_codebooks"]
                or not codes.size
                or not np.issubdtype(codes.dtype, np.integer)
                or np.any(codes < 0)
                or np.any(codes >= audio["vocab_size"])
            ):
                raise ValueError(
                    "history codes must be nonempty integer [frames,10] within vocabulary"
                )
            if len(codes) > self.max_prompt_rows:
                raise ValueError("history exceeds prompt row limit")
            turns.extend(
                (
                    *previous.segments[3:],
                    codes.astype(np.int64, copy=False),
                    "<|im_end|>\n",
                )
            )
        segments = segments[:system_end] + tuple(turns) + segments[system_end:]
        blocks, count = [], 0
        for segment in segments:
            if isinstance(segment, str):
                ids = self.tokenizer.encode(segment).ids
                block = np.zeros((len(ids), 11), dtype=np.int64)
                block[:, 0] = ids
            else:
                block = np.column_stack(
                    (segment[:, 0] + self.config["semantic_start_token_id"], segment)
                )
            count += len(block)
            if count > self.max_prompt_rows:
                raise ValueError(f"prompt exceeds {self.max_prompt_rows} rows")
            blocks.append(block)
        rows = np.ascontiguousarray(np.concatenate(blocks), dtype=np.int64)
        semantic = (rows[:, 0] >= self.config["semantic_start_token_id"]) & (
            rows[:, 0] <= self.config["semantic_end_token_id"]
        )
        return Drama3Prompt(
            rows=rows,
            semantic_mask=semantic,
            normalized_text=text,
            segments=segments,
            **{
                field: self.config[field]
                for field in (
                    "semantic_start_token_id",
                    "semantic_end_token_id",
                    "eos_token_id",
                    "audio_pad_token_id",
                )
            },
        )
