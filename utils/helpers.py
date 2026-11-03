"""Small, deterministic helpers: number parsing, text matching, hashing, logging."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

_LOG_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# --------------------------------------------------------------------------- #
# Number parsing
# --------------------------------------------------------------------------- #
# Something that looks like a printed amount/quantity: digits with optional
# thousands separators (, . space ' or thin space) and an optional decimal part.
NUMBER_TOKEN_RE = re.compile(r"(?<![\w.,])[-−(]?\d{1,3}(?:[ ,.'’  ]\d{3})*(?:[.,]\d+)?\)?(?![\w])|(?<![\w.,])[-−(]?\d+(?:[.,]\d+)?\)?(?![\w])")

_CURRENCY_RE = re.compile(
    r"(?i)(?:US\$|USD|EUR|INR|GBP|CHF|SEK|NOK|DKK|AUD|CAD|JPY|CNY|RS\.?|RUPEES|[$€£¥₹₩₽])"
)


def detect_decimal_separator(text: str) -> str:
    """Guess the document's decimal separator ('.' or ',') from its amounts.

    Counts patterns such as ``1,234.56`` / ``12.50`` (dot-decimal) versus
    ``1.234,56`` / ``12,50`` (comma-decimal). Amounts with exactly two decimal
    digits are by far the most common on invoices, so they are a reliable vote.
    """
    dot = len(re.findall(r"\d,\d{3}\.\d{1,2}\b", text)) * 3 + len(re.findall(r"(?<![\d.,])\d+\.\d{2}(?![\d.,])", text))
    comma = len(re.findall(r"\d\.\d{3},\d{1,2}\b", text)) * 3 + len(re.findall(r"(?<![\d.,])\d+,\d{2}(?![\d.,])", text))
    return "," if comma > dot else "."


def parse_number(raw: Optional[str], decimal_sep: str = ".") -> Optional[float]:
    """Parse a printed number into a float, deterministically.

    Handles currency symbols/codes, thousands separators (``,`` ``.`` space ``'``),
    comma decimals, trailing/leading minus, parentheses for negatives and unit
    suffixes (``"10,400 pcs"``). Returns ``None`` when the text does not contain
    exactly one unambiguous number – we never guess.
    """
    if raw is None:
        return None
    s = unicodedata.normalize("NFKC", str(raw)).strip()
    if not s:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1]
    s = s.replace("−", "-").replace("–", "-")
    s = _CURRENCY_RE.sub(" ", s).strip()
    if s.endswith("-") and not s.startswith("-"):  # "9,32-" (common in ERP prints)
        negative, s = True, s[:-1]
    if s.startswith("-"):
        negative, s = True, s[1:]
    # Drop trailing unit words like "pcs", "kg", "Units" (keep the number).
    s = re.sub(r"(?i)\s*[a-z%][a-z.%/]*\s*$", "", s).strip()
    s = s.lstrip("+ ").strip()
    if not s or not re.fullmatch(r"[\d ,.'’  ]+", s):
        return None
    s = re.sub(r"[ '’  ]", "", s)  # spaces/apostrophes are only ever grouping
    if not re.search(r"\d", s):
        return None

    has_dot, has_comma = "." in s, ","in s
    if has_dot and has_comma:
        # The right-most separator is the decimal one.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_comma or has_dot:
        sep = "," if has_comma else "."
        parts = s.split(sep)
        if len(parts) > 2:
            # "1,234,567" or "1.234.567" -> grouping only
            if all(len(p) == 3 for p in parts[1:]):
                s = "".join(parts)
            else:
                return None
        else:
            frac = parts[1]
            if len(frac) == 3 and sep != decimal_sep:
                s = "".join(parts)  # e.g. "10,400" in a dot-decimal document
            elif len(frac) == 3 and sep == decimal_sep and len(parts[0]) <= 3 and parts[0] != "0":
                # "1.500" in a dot-decimal document is still ambiguous only if the
                # doc *also* groups with dots; trust the document convention.
                s = parts[0] + "." + frac
            else:
                s = parts[0] + "." + frac
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def number_variants(value: float) -> set[str]:
    """Printed forms a value could take, used to look for it in source text."""
    out = set()
    for decimals in (0, 1, 2, 3, 4):
        if decimals == 0 and value != int(value):
            continue
        base = f"{abs(value):,.{decimals}f}"  # 1,234.50
        out.add(base)
        out.add(base.replace(",", ""))  # 1234.50
        swapped = base.replace(",", "§").replace(".", ",").replace("§", ".")  # 1.234,50
        out.add(swapped)
        out.add(swapped.replace(".", ""))  # 1234,50
        out.add(base.replace(",", " "))  # 1 234.50
        out.add(swapped.replace(".", " "))  # 1 234,50
    return out


def number_in_text(value: Optional[float], text: str, decimal_sep: str = ".") -> bool:
    """True if ``value`` appears as a number token in ``text``."""
    if value is None or not text:
        return False
    for tok in NUMBER_TOKEN_RE.findall(text):
        parsed = parse_number(tok, decimal_sep)
        if parsed is not None and abs(abs(parsed) - abs(value)) < 1e-6:
            return True
    return False


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def clean_whitespace(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _norm_for_match(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def text_support_ratio(needle: Optional[str], haystack: str) -> float:
    """Share of the needle's words that can be found (fuzzily) in the haystack.

    Used to check that an extracted material/vendor name really occurs in the
    document text, i.e. that it was read rather than invented. Fuzzy matching
    tolerates small OCR errors ("Industrivagen" vs "Industrivägen").
    """
    if not needle:
        return 0.0
    words = [w for w in _norm_for_match(needle).split() if len(w) > 1 or w.isdigit()]
    if not words:
        return 0.0
    hay = _norm_for_match(haystack)
    hay_words = set(hay.split())
    found = 0
    for w in words:
        if w in hay_words or (len(w) >= 4 and w in hay):
            found += 1
        elif len(w) >= 4 and any(SequenceMatcher(None, w, h).ratio() >= 0.8 for h in hay_words if abs(len(h) - len(w)) <= 2):
            found += 1
    return found / len(words)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_for_match(a), _norm_for_match(b)).ratio()
