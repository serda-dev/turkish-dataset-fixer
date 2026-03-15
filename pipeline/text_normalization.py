"""
text_normalization.py — Stage 1: Safe, minimal text normalization.

Preserves Turkish characters (ç, ğ, ı, İ, ö, ş, ü).
Does NOT aggressively rewrite content.
"""

import re
import unicodedata


# Null bytes and broken control chars (keep \n, \t, \r)
_CONTROL_CHAR_RE = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]'
)

# Various Unicode whitespace characters → standard space
_UNICODE_WHITESPACE_RE = re.compile(
    r'[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000\ufeff]'
)

# Collapse excessive blank lines (3+ consecutive → 2)
_EXCESSIVE_BLANK_LINES_RE = re.compile(r'\n{4,}')

# Collapse excessive spaces (4+ consecutive → single space)
_EXCESSIVE_SPACES_RE = re.compile(r' {4,}')

# Zero-width chars that serve no purpose in corpus text
_ZERO_WIDTH_RE = re.compile(r'[\u200b\u200c\u200d\u2060\ufeff]')


def normalize_text(text: str) -> str:
    """
    Apply safe, minimal normalization to text.

    Operations (in order):
    1. Remove null bytes and broken control characters
    2. Remove zero-width characters
    3. Normalize Unicode whitespace to standard space
    4. Collapse excessive blank lines (4+ → 2 newlines)
    5. Collapse excessive spaces (4+ → 1 space)
    6. Strip leading/trailing whitespace

    Does NOT:
    - Change case
    - Remove diacriticals
    - Modify Turkish characters
    - Aggressively rewrite punctuation
    - Remove legitimate Unicode
    """
    if not text:
        return text

    # Step 1: Remove control characters (keep \n, \t, \r)
    text = _CONTROL_CHAR_RE.sub('', text)

    # Step 2: Remove zero-width characters
    text = _ZERO_WIDTH_RE.sub('', text)

    # Step 3: Normalize Unicode whitespace
    text = _UNICODE_WHITESPACE_RE.sub(' ', text)

    # Step 4: Collapse excessive blank lines
    text = _EXCESSIVE_BLANK_LINES_RE.sub('\n\n\n', text)

    # Step 5: Collapse excessive spaces
    text = _EXCESSIVE_SPACES_RE.sub(' ', text)

    # Step 6: Strip edges
    text = text.strip()

    return text
