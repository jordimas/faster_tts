"""Conservative English/Spanish sentence boundaries for plain-text synthesis."""
import re


_ABBREVIATIONS = frozenset(('mr', 'mrs', 'ms', 'dr', 'dra', 'sr', 'sra', 'srta',
                            'prof', 'jr', 'vs', 'etc', 'approx', 'no', 'núm', 'ud', 'uds'))
_END = re.compile(r'[.!?]+[\"\'”’»)]*(?=\s|$)|\n+')


def split_text(text):
    """Keep decimals/initials/abbreviations together; leave markup untouched.

    This is a bounded plain-text heuristic, not a multilingual linguistic parser.
    Each output is a contiguous span of the input, with boundary whitespace removed.
    """
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    if any(mark in text for mark in ('<', '>', '[', ']')):
        return [text]
    chunks, start = [], 0
    for match in _END.finditer(text):
        if match.group().startswith('.'):
            prefix = text[:match.start()]
            word = re.search(r'([\w.]+)$', prefix)
            word = word.group(1) if word else ''
            if (word.casefold() in _ABBREVIATIONS
                    or len(word) == 1 and word.isalpha()
                    or re.fullmatch(r'(?:[A-Za-z]\.)+[A-Za-z]', word)):
                continue
        chunk = text[start:match.end()].strip()
        if chunk:
            chunks.append(chunk)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        chunks.append(tail)
    return chunks or [text]
