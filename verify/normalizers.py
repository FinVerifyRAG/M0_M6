"""
verify/normalizers.py
---------------------
Canonicalize numbers, units, dates, and section references so that
V1 deterministic matching works across textual variants.

All public functions return None when the input cannot be parsed.

Examples tested in tests/test_verify_normalizers.py
"""
import re
from typing import Optional
from datetime import date


# ---------------------------------------------------------------------------
# Numeric / monetary normalization
# ---------------------------------------------------------------------------

_LAKH_MAP = {
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
    "crore": 1e7, "crores": 1e7, "cr": 1e7, "cr.": 1e7,
    "thousand": 1e3, "thousands": 1e3,
    "million": 1e6, "billion": 1e9,
}

_AMOUNT_RE = re.compile(
    r"(?:₹|Rs\.?|INR|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*"
    r"(lakh|lakhs?|lac|lacs?|crore|crores?|cr\.?|thousand|thousands?|million|billion)?",
    re.IGNORECASE,
)


def norm_amount(s: str) -> Optional[float]:
    """
    Normalize a rupee amount to a plain float.
    '₹5 lakh' -> 500000.0, '₹1.5 crore' -> 15000000.0, '5,00,000' -> 500000.0
    """
    s = s.strip()
    m = _AMOUNT_RE.search(s)
    if not m:
        return None
    num_str = m.group(1).replace(",", "")
    try:
        value = float(num_str)
    except ValueError:
        return None
    unit = (m.group(2) or "").lower().rstrip(".")
    multiplier = _LAKH_MAP.get(unit, 1.0)
    return value * multiplier


# ---------------------------------------------------------------------------
# Percentage normalization
# ---------------------------------------------------------------------------

_PCT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent|percent)(?:\s*(?:per\s*annum|p\.?a\.?))?",
    re.IGNORECASE,
)


def norm_pct(s: str) -> Optional[float]:
    """
    Normalize a percentage to a plain float.
    '5%' -> 5.0, '5 per cent' -> 5.0, '5.5% p.a.' -> 5.5
    """
    m = _PCT_RE.search(s.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# dd/mm/yyyy or dd-mm-yyyy
_DMY_SLASH = re.compile(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})")
# dd Month yyyy  e.g. "12 March 2024"
_DMY_WORD = re.compile(
    r"(\d{1,2})\s+"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+(\d{4})",
    re.IGNORECASE,
)
# yyyy-mm-dd ISO
_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

# FY 2023-24 → we store the start year
_FY_RE = re.compile(r"FY\s*(\d{4})[-–](\d{2,4})", re.IGNORECASE)


def norm_date(s: str) -> Optional[date]:
    """
    Parse an Indian date string and return a datetime.date object.
    Returns None when parsing fails.

    Supports:
      - 'dd/mm/yyyy', 'dd-mm-yyyy'
      - 'dd Month yyyy' ('12 March 2024')
      - 'yyyy-mm-dd'
      - 'FY 2023-24' → 01 Apr 2023
    """
    s = s.strip()

    m = _ISO.fullmatch(s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    m = _DMY_WORD.search(s)
    if m:
        day = int(m.group(1))
        mon = _MONTH_MAP.get(m.group(2).lower())
        yr = int(m.group(3))
        if mon:
            try:
                return date(yr, mon, day)
            except ValueError:
                pass

    m = _DMY_SLASH.search(s)
    if m:
        d_, mo_, y_ = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y_ < 100:
            y_ += 2000
        try:
            return date(y_, mo_, d_)
        except ValueError:
            # maybe it was already mm/dd/yyyy; try swapping
            try:
                return date(y_, d_, mo_)
            except ValueError:
                pass

    m = _FY_RE.search(s)
    if m:
        fy_start = int(m.group(1))
        try:
            return date(fy_start, 4, 1)  # 1st April of the financial year
        except ValueError:
            pass

    return None


def dates_equal(a: str, b: str) -> Optional[bool]:
    """
    Return True if both strings normalise to the same date, False if they
    differ, None if either cannot be parsed.
    """
    da, db = norm_date(a), norm_date(b)
    if da is None or db is None:
        return None
    return da == db


# ---------------------------------------------------------------------------
# Section / regulation reference normalization
# ---------------------------------------------------------------------------

_SEC_RE = re.compile(
    r"\b(?:section|regulation|clause|rule|article|para(?:graph)?|schedule|"
    r"circular|master\s+direction)\s*(\d+[A-Z]?)(?:\((\w+)\))*",
    re.IGNORECASE,
)


def norm_section(s: str) -> Optional[str]:
    """
    Normalise a section reference to lowercase canonical form.
    'Section 80C(2)' -> 'section 80c(2)'
    'Regulation 52 (4)(a)' -> 'regulation 52(4)(a)'
    '80C(2)' is passed through as-is (no leading keyword).
    """
    s = s.strip()
    return s.lower().replace(" ", "")


def sections_equal(a: str, b: str) -> bool:
    """Return True if two section references normalise to the same string."""
    return norm_section(a) == norm_section(b)


# ---------------------------------------------------------------------------
# Generic numeric comparison (tolerant of floating-point noise)
# ---------------------------------------------------------------------------

def numerically_equal(a: Optional[float], b: Optional[float],
                      rel_tol: float = 1e-6) -> Optional[bool]:
    """Compare two float values with relative tolerance. Returns None if either is None."""
    if a is None or b is None:
        return None
    if a == 0 and b == 0:
        return True
    return abs(a - b) / max(abs(a), abs(b)) < rel_tol
