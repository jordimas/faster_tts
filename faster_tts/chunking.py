"""Conservative Unicode sentence and clause boundaries for synthesis."""
import re


_ABBREVIATIONS = frozenset(('mr', 'mrs', 'ms', 'dr', 'dra', 'sr', 'sra', 'srta',
                            'prof', 'jr', 'vs', 'etc', 'approx', 'no', 'núm', 'ud', 'uds'))
# Period abbreviations are inherently ambiguous without language context. Keep
# familiar forms and short initials together; unknown forms remain heuristic.
_ABBREVIATIONS = _ABBREVIATIONS | frozenset((
    'mme', 'mmes', 'mlle', 'mlles', 'ém', 'av', 'bd', 'bzw', 'ca', 'dipl',
    'ing', 'nr', 'str', 'vgl', 'z.b', 'd.h', 'sig', 'sig.ra', 'dott', 'dott.ssa',
    'ecc', 'ex', 'p.ex', 'sra', 'sta', 'sant', 'pàg', 'pág', 'tel',
    'г', 'гг', 'ул', 'д', 'им', 'рис', 'стр', 'т.д', 'т.п',
))
_CLOSERS = r'[\"\'”’»）)\]}」』】》]*'
# These marks can end sentences without whitespace (e.g. Chinese and Japanese).
_TERMINALS = '。！？؟۔।॥።፧፨։။᙮꓿'
_END = re.compile(
    r'[.!?]+' + _CLOSERS + r'(?=\s|$)|[' + _TERMINALS + r']+'
    + _CLOSERS + r'|\n+'
)
_CLAUSE = re.compile(r'[,;:](?=\s)|[，、；：،؛]')


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
        raise TypeError('text must be a string')
    original = text
    speaker = re.match(r'[ \t\r\n]*(<\|speaker:(?:[0-9]|1[0-5])\|>)[ \t\r\n]*', text)
    if speaker:
        text = text[speaker.end():]
    if any(mark in text for mark in ('<', '>')):
        return [original]
    cued = '[' in text or ']' in text
    boundaries = text
    if cued:
        if speaker:
            return [original]
        boundaries = re.sub(r'\[[^\[\]<>]+\]', lambda match: ' ' * len(match[0]), text)
        if '[' in boundaries or ']' in boundaries:
            return [original]
    chunks, start = [], 0
    for match in _END.finditer(boundaries):
        if match.group().startswith('.'):
            prefix = boundaries[:match.start()]
            word = re.search(r'([\w.]+)$', prefix)
            word = word.group(1) if word else ''
            if (word.casefold() in _ABBREVIATIONS
                    or len(word) == 1 and word.isalpha()
                    or re.fullmatch(r'(?:[^\W\d_]\.)+[^\W\d_]', word)):
                continue
        chunk = text[start:match.end()].strip()
        if chunk:
            chunks.append(chunk)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        chunks.append(tail)
    if cued:
        # A cue's scope across sentences is unspecified: only split when each
        # sentence carries exactly one leading cue of its own.
        if not all(re.match(r'^\[[^\[\]<>]+\]\s+[^\[\]]', chunk)
                   and chunk.count('[') == 1 for chunk in chunks):
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
        return [speaker[1] + ' ' + chunk for chunk in chunks] if len(chunks) > 1 else [original]
    return chunks or [text]
