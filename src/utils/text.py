"""Character vocabulary and text encoding utilities for SilentRead.

GRID Corpus sentences use a restricted vocabulary of 51 words and 26 lower-case
letters + space.  We work at the character level and reserve two special tokens:

  PAD_IDX  (0) — used to pad label tensors to a fixed batch length
  BLANK_IDX       — CTC blank token, appended at the end of the alphabet
"""

import re
from typing import List

# --------------------------------------------------------------------- #
# Alphabet: lower-case letters + space, then CTC blank at the very end   #
# --------------------------------------------------------------------- #
_CHARS = list("abcdefghijklmnopqrstuvwxyz ")  # 27 printable chars

PAD_IDX: int = 0                           # padding index (not a real char)
# Chars are indexed 1 … 27 so that 0 is reserved for padding
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(_CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(_CHARS)}

BLANK_IDX: int = len(_CHARS) + 1          # 28 — CTC blank
VOCAB_SIZE: int = BLANK_IDX + 1           # 29 total output classes

# Public alias used elsewhere
VOCAB = _CHARS


def encode_text(text: str) -> List[int]:
    """Convert a string to a list of integer indices (1-based, no blank/pad).

    Unknown characters are silently dropped.

    >>> encode_text("bin blue")
    [2, 9, 14, 27, 2, 12, 21, 5]
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode_indices(indices: List[int], *, remove_duplicates: bool = True,
                   remove_blank: bool = True) -> str:
    """Convert a list of integer indices back to a string.

    Args:
        indices: Raw sequence from the model (may contain blank / duplicates).
        remove_duplicates: Collapse consecutive identical tokens (CTC-style).
        remove_blank: Drop BLANK_IDX tokens after duplicate removal.

    >>> decode_indices([2, 2, 9, 28, 14, 27, 2, 12, 21, 5])
    'bin blue'
    """
    if remove_duplicates:
        deduped: List[int] = []
        prev = None
        for idx in indices:
            if idx != prev:
                deduped.append(idx)
            prev = idx
        indices = deduped

    if remove_blank:
        indices = [i for i in indices if i != BLANK_IDX]

    chars = [IDX_TO_CHAR.get(i, "") for i in indices]
    return "".join(chars).strip()
