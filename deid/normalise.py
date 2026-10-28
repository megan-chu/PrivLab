"""
normalise.py - fold look-alike characters before scanning, one character for one character.

Zero-width and no-break spaces become spaces, dashes become "-", and Cyrillic/Greek letters
that look Latin become Latin ("Ѕ7654321F" -> "S7654321F"). Length never changes, so every
offset found in the folded text points at the same place in the original.
"""
SPACES = "              　"
ZERO_WIDTH = "​‌‍⁠﻿"
DASHES = "‐‑‒–—―−"
HOMOGLYPHS = {  # Cyrillic and Greek letters that look like Latin ones
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J", "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x",
    "у": "y", "ѕ": "s", "і": "i", "ј": "j", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "ο": "o",
}
TABLE = str.maketrans({**{c: " " for c in SPACES + ZERO_WIDTH}, **{c: "-" for c in DASHES}, **HOMOGLYPHS})


def fold(text: str) -> str:
    return text.translate(TABLE)
