"""
checksums.py - check digits for national ID numbers.

Used by the pattern recognisers to drop look-alikes (a random letter + 7 digits
+ letter is not an NRIC unless its check letter is right).
"""
import re

NRIC_WEIGHTS = (2, 7, 6, 5, 4, 3, 2)
NRIC_TABLES = {
    "S": "JZIHGFEDCBA",
    "T": "JZIHGFEDCBA",
    "F": "XWUTRQPNMLK",
    "G": "XWUTRQPNMLK",
}
NRIC_OFFSET = {"S": 0, "F": 0, "T": 4, "G": 4}


def nric_valid(s: str) -> bool:
    """Singapore NRIC/FIN, e.g. S1234567D. M-series FINs are accepted on format only."""
    s = s.strip().upper()
    if not re.fullmatch(r"[STFGM]\d{7}[A-Z]", s):
        return False
    prefix = s[0]
    if prefix == "M":
        # PLACEHOLDER: M-series (2022+) check-letter table not verified; none in the dataset.
        return True
    total = sum(int(d) * w for d, w in zip(s[1:8], NRIC_WEIGHTS)) + NRIC_OFFSET[prefix]
    return NRIC_TABLES[prefix][total % 11] == s[8]


def luhn_valid(s: str) -> bool:
    """Payment card number: 13-19 digits (spaces or hyphens allowed), Luhn check digit."""
    digits = re.sub(r"[ -]", "", s.strip())
    if not re.fullmatch(r"\d{13,19}", digits) or len(set(digits)) == 1:
        return False
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2) for i, d in enumerate(int(c) for c in reversed(digits)))
    return total % 10 == 0


def iban_valid(s: str) -> bool:
    """IBAN, e.g. GB82 WEST 1234 5698 7654 32: country code, 2 check digits, mod-97 == 1."""
    s = re.sub(r"[ -]", "", s.strip()).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    return int("".join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1


def nhs_valid(s: str) -> bool:
    """UK NHS number: 10 digits (spaces or hyphens allowed), mod-11 check digit."""
    digits = re.sub(r"[ -]", "", s.strip())
    if not re.fullmatch(r"\d{10}", digits):
        return False
    remainder = sum(int(d) * w for d, w in zip(digits[:9], range(10, 1, -1))) % 11
    check = 11 - remainder
    if check == 11:
        check = 0
    return check != 10 and check == int(digits[9])
