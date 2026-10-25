"""Conservative Unicode sentence and clause boundaries for synthesis."""

import re
from itertools import pairwise

_ABBREVIATIONS = frozenset(
    (
        "mr",
        "mrs",
        "ms",
        "dr",
        "dra",
        "sr",
        "sra",
        "srta",
        "prof",
        "jr",
        "vs",
        "etc",
        "approx",
        "no",
        "núm",
        "ud",
        "uds",
    )
)
# Period abbreviations are inherently ambiguous without language context. Keep
# familiar forms and short initials together; unknown forms remain heuristic.
_ABBREVIATIONS = _ABBREVIATIONS | frozenset(
    (
        "mme",
        "mmes",
        "mlle",
        "mlles",
        "ém",
        "av",
        "bd",
        "bzw",
        "ca",
        "dipl",
        "ing",
        "nr",
        "str",
        "vgl",
        "z.b",
        "d.h",
        "sig",
        "sig.ra",
        "dott",
        "dott.ssa",
        "ecc",
        "ex",
        "p.ex",
        "sra",
        "sta",
        "sant",
        "pàg",
        "pág",
        "tel",
        "г",
        "гг",
        "ул",
        "д",
        "им",
        "рис",
        "стр",
        "т.д",
        "т.п",
    )
)
_CLOSERS = r"[\"\'”’»）)\]}」』】》]*"
# These marks can end sentences without whitespace (e.g. Chinese and Japanese).
_TERMINALS = "。！？؟۔।॥።፧፨։။᙮꓿"
_END = re.compile(
    r"[.!?]+" + _CLOSERS + r"(?=\s|$)|[" + _TERMINALS + r"]+" + _CLOSERS + r"|\n+"
)
_CLAUSE = re.compile(r"[,;:](?=\s)|[，、；：،؛]")


def split_text(text):
    """Keep abbreviations together and preserve a single leading speaker tag.

    This is a Unicode punctuation heuristic, not a linguistic parser. Text in
    any language is accepted; text without clear boundaries stays together.
    Long sentences group at clause punctuation with a 200-character soft target.
    Sentence-attached bracket cues are kept when every sentence has its own cue.
    Other markup stays intact. Spoken spans retain their punctuation and internal
    whitespace; a leading speaker tag is repeated on each generated chunk.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    original = text
    speaker = re.match(r"[ \t\r\n]*(<\|speaker:(?:[0-9]|1[0-5])\|>)[ \t\r\n]*", text)
    if speaker:
        text = text[speaker.end() :]
    if any(mark in text for mark in ("<", ">")):
        return [original]
    cued = "[" in text or "]" in text
    boundaries = text
    if cued:
        if speaker:
            return [original]
        boundaries = re.sub(r"\[[^\[\]<>]+\]", lambda match: " " * len(match[0]), text)
        if "[" in boundaries or "]" in boundaries:
            return [original]
    chunks, start = [], 0
    for match in _END.finditer(boundaries):
        if match.group().startswith("."):
            prefix = boundaries[: match.start()]
            word = re.search(r"([\w.]+)$", prefix)
            word = word.group(1) if word else ""
            if (
                word.casefold() in _ABBREVIATIONS
                or len(word) == 1
                and word.isalpha()
                or re.fullmatch(r"(?:[^\W\d_]\.)+[^\W\d_]", word)
            ):
                continue
        chunk = text[start : match.end()].strip()
        if chunk:
            chunks.append(chunk)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        chunks.append(tail)
    if cued:
        # A cue's scope across sentences is unspecified: only split when each
        # sentence carries exactly one leading cue of its own.
        if not all(
            re.match(r"^\[[^\[\]<>]+\]\s+[^\[\]]", chunk) and chunk.count("[") == 1
            for chunk in chunks
        ):
            return [original]
        return chunks if len(chunks) > 1 else [original]
    bounded = []
    for chunk in chunks:
        start = previous = 0
        ends = [match.end() for match in _CLAUSE.finditer(chunk)] + [len(chunk)]
        for end in ends:
            if end - start > 200 and previous > start:
                bounded.append(chunk[start:previous].strip())
                start = previous
            previous = end
        bounded.append(chunk[start:].strip())
    chunks = bounded
    if speaker:
        return (
            [speaker[1] + " " + chunk for chunk in chunks]
            if len(chunks) > 1
            else [original]
        )
    return chunks or [text]


_CUE = re.compile(r"\[[^\[\]<>\n]+\]")
_SPEAKER = re.compile(r"<\|speaker:(?:[0-9]|1[0-5])\|>")
_BREATH = re.compile(r"\[breathing\]", re.IGNORECASE)
_TERMINAL = re.compile(r"[.!?" + _TERMINALS + r"]" + _CLOSERS + r"$")


def _terminal_boundary(text):
    if "\n" in text[len(text.rstrip()) :]:
        return True
    value = text.rstrip()
    terminal = _TERMINAL.search(value)
    if terminal is None:
        return False
    if value[terminal.start()] == ".":
        word = re.search(r"([\w.]+)$", value[: terminal.start()])
        word = word[1] if word else ""
        if (
            word.casefold() in _ABBREVIATIONS
            or len(word) == 1
            and word.isalpha()
            or re.fullmatch(r"(?:[^\W\d_]\.)+[^\W\d_]", word)
        ):
            return False
    return True


def _directed_chunks(text):
    # Retain ambiguous scopes and malformed directions as a single span.
    cues = list(_CUE.finditer(text))
    masked = _CUE.sub("", text)
    if "<" in text or ">" in text or "[" in masked or "]" in masked:
        return [text]
    if not cues:
        return split_text(text)
    prefix = text[: cues[0].start()]
    if prefix.strip() and (
        not any(c.isalnum() for c in prefix) or not _terminal_boundary(prefix)
    ):
        return [text]
    if len(cues) < 2 and not prefix.strip():
        return split_text(text)
    for index, cue in enumerate(cues):
        end = cues[index + 1].start() if index + 1 < len(cues) else len(text)
        spoken = text[cue.end() : end]
        if (
            not any(c.isalpha() for c in cue[0][1:-1])
            or not any(c.isalnum() for c in spoken)
            or index + 1 < len(cues)
            and not _terminal_boundary(spoken)
        ):
            return [text]
    starts = [c.start() for c in cues] + [len(text)]
    return ([prefix.strip()] if prefix.strip() else []) + [
        text[a:b].strip() for a, b in pairwise(starts)
    ]


def directed_segments(text):
    """Plan opt-in speech/event spans, retaining ambiguous markup intact.

    Only explicit breathing between complete speech clauses is an event.
    Speaker turns keep their original tag; cue scopes are never inferred.
    Returns a list of `(kind, text)` pairs without generating audio.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    # Unknown angle markup must be validated by the prompt builder unchanged.
    if "<" in _SPEAKER.sub("", text) or ">" in _SPEAKER.sub("", text):
        return [("speech", text)]
    turns = [
        t.strip()
        for t in re.split(r"(?=<\|speaker:(?:[0-9]|1[0-5])\|>)", text)
        if t.strip()
    ]
    segments = []
    for turn in turns:
        leading = _SPEAKER.match(turn)
        tag = leading[0] + " " if leading else ""
        body = turn[leading.end() :].strip() if leading else turn
        events = list(_BREATH.finditer(body))
        if events:
            masked = _CUE.sub("", body)
            clauses = _BREATH.split(body)
            if (
                "[" not in masked
                and "]" not in masked
                and all(
                    any(c.isalnum() for c in _CUE.sub("", clause)) for clause in clauses
                )
                and all(_terminal_boundary(clause) for clause in clauses[:-1])
            ):
                for index, clause in enumerate(clauses):
                    segments.append(("speech", tag + clause.strip()))
                    if index < len(events):
                        segments.append(("event", tag + events[index][0]))
                continue
        segments.extend(("speech", tag + chunk) for chunk in _directed_chunks(body))
    return segments or [("speech", text)]
