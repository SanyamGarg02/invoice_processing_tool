"""Deterministic normalisation of raw extraction output into internal records."""

from __future__ import annotations

import re

from models.invoice_schema import LineItemRecord, LLMInvoiceExtraction, RawLineItem
from utils.helpers import clean_whitespace, parse_number

KNOWN_UNITS = {
    "pcs", "pc", "pce", "pces", "piece", "pieces", "ea", "each", "unit", "units", "nos", "no", "stk", "st", "stuks",
    "kg", "g", "gr", "mg", "t", "ton", "tons", "tonne", "lb", "lbs", "oz",
    "l", "lt", "ltr", "litre", "litres", "liter", "liters", "ml", "m3", "gal",
    "m", "mm", "cm", "km", "m2", "sqm", "sqft", "ft", "in", "yd",
    "roll", "rolls", "box", "boxes", "set", "sets", "pack", "packs", "pkt", "pair", "pairs", "bag", "bags",
    "carton", "cartons", "ctn", "drum", "drums", "pallet", "pallets", "sheet", "sheets", "coil", "coils",
    "h", "hr", "hrs", "hour", "hours", "day", "days", "month", "months", "night", "nights", "lot", "job",
}
COUNT_UNITS = {"pcs", "pc", "pce", "pces", "piece", "pieces", "ea", "each", "nos", "no", "stk", "st", "stuks",
               "box", "boxes", "set", "sets", "pack", "packs", "pkt", "pair", "pairs", "bag", "bags", "carton",
               "cartons", "ctn", "drum", "drums", "pallet", "pallets", "sheet", "sheets", "roll", "rolls", "coil", "coils"}

_CURRENCY_HINTS = [("€", "EUR"), ("$", "USD"), ("£", "GBP"), ("₹", "INR"), ("rs", "INR"), ("eur", "EUR"),
                   ("usd", "USD"), ("inr", "INR"), ("gbp", "GBP"), ("sek", "SEK"), ("chf", "CHF")]


def normalize_vendor(name: str | None) -> str | None:
    name = clean_whitespace(name)
    if not name:
        return None
    name = name.strip(" ,;:-|*")
    return name or None


def normalize_material(name: str | None) -> str | None:
    """Tidy whitespace and stray OCR/table characters without changing meaning."""
    name = clean_whitespace(name)
    if not name:
        return None
    name = re.sub(r"^[|•·\-–—:;,.\s]+", "", name)
    name = re.sub(r"[|\s]+$", "", name)
    return name or None


def normalize_unit(unit: str | None) -> str | None:
    unit = clean_whitespace(unit)
    if not unit:
        return None
    u = unit.strip(" .|").lower()
    # OCR often glues a table border to the unit: "Jroll", "|pcs", "froll".
    if u not in KNOWN_UNITS and len(u) > 2 and u[0] in "j|f[(l1i!" and u[1:] in KNOWN_UNITS:
        u = u[1:]
    if u in KNOWN_UNITS:
        return "L" if u == "l" else u  # a lone lowercase "l" reads like the digit 1
    return unit.strip(" .|") or None


def detect_currency(*texts: str | None) -> str | None:
    for t in texts:
        if not t:
            continue
        low = t.lower()
        for hint, code in _CURRENCY_HINTS:
            if hint in low:
                return code
    return None


_CURRENCY_TOKENS = {"€": "EUR", "$": "USD", "£": "GBP", "₹": "INR", "¥": "JPY"}
_ISO_CURRENCIES = {"EUR", "USD", "GBP", "INR", "SEK", "NOK", "DKK", "CHF", "JPY", "CNY", "AUD", "CAD", "PLN",
                   "CZK", "HUF", "SGD", "AED", "ZAR", "BRL", "MXN", "NZD", "HKD"}


def detect_document_currency(text: str | None) -> str | None:
    """The currency printed most often on the invoice (ISO codes as words, or symbols)."""
    if not text:
        return None
    counts: dict[str, int] = {}
    for code in re.findall(r"\b[A-Z]{3}\b", text):
        if code in _ISO_CURRENCIES:
            counts[code] = counts.get(code, 0) + 1
    for sym, code in _CURRENCY_TOKENS.items():
        if sym in text:
            counts[code] = counts.get(code, 0) + text.count(sym)
    for _ in re.findall(r"\bRs\.?(?=\s*[-−]?\d)", text):
        counts["INR"] = counts.get("INR", 0) + 1
    return max(counts, key=counts.get) if counts else None


def format_quantity(qty: float | None, unit: str | None, grouping: bool = True) -> str:
    """Quantity with its unit for display: "10 kg", "2.5 L", "1,000 pcs" (no unit printed: "10")."""
    if qty is None:
        return ""
    num = f"{qty:{',' if grouping else ''}.4f}".rstrip("0").rstrip(".")
    return f"{num} {unit}" if unit else num


def format_value(value: float | None, currency: str | None, grouping: bool = True) -> str:
    """Value with its currency for display: "EUR 1,234.50" (no currency identified: "1,234.50")."""
    if value is None:
        return ""
    num = f"{value:{',' if grouping else ''}.2f}"
    return f"{currency} {num}" if currency else num


_LEADING_NUMBER = re.compile(r"[-−]?\d[\d.,'’ ]*\d|[-−]?\d")


def split_labelled_number(text: str | None) -> tuple[float | None, str | None]:
    """Reviewer input such as "10 kg", "EUR 44.99" or "44,99 €" -> (number, label)."""
    text = clean_whitespace(text)
    if not text:
        return None, None
    m = _LEADING_NUMBER.search(text)
    if not m:
        return None, text
    label = (text[:m.start()] + " " + text[m.end():]).strip(" :") or None
    return parse_number(m.group().strip()), label


def normalize_currency(label: str | None) -> str | None:
    label = clean_whitespace(label)
    if not label:
        return None
    if re.fullmatch(r"[A-Za-z]{3}", label):
        return label.upper()
    return detect_currency(label) or label


def normalize_item(
    raw: RawLineItem,
    *,
    invoice_id: str,
    source_file: str,
    line_index: int,
    vendor_name: str | None,
    currency: str | None,
    decimal_sep: str,
    method: str,
) -> LineItemRecord:
    qty_raw = clean_whitespace(raw.quantity_text)
    unit = raw.unit
    if qty_raw and not unit:
        # Keep a unit printed inside the quantity cell ("1.00 kg") – never drop it.
        m = re.match(r"^\s*[-−]?[\d.,' ]+\s*([A-Za-z][A-Za-z.]{0,8})\s*$", qty_raw)
        if m:
            unit = m.group(1)
    return LineItemRecord(
        invoice_id=invoice_id,
        source_file=source_file,
        line_index=line_index,
        vendor_name=normalize_vendor(vendor_name),
        material_name=normalize_material(raw.material_name),
        material_quantity=parse_number(qty_raw.replace("|", " ") if qty_raw else None, decimal_sep),
        material_value=parse_number(raw.value_text.replace("|", " ") if raw.value_text else None, decimal_sep),
        unit=normalize_unit(unit),
        currency=currency or detect_currency(raw.value_text),
        raw_quantity=qty_raw,
        raw_value=clean_whitespace(raw.value_text),
        source_page=raw.source_page,
        source_line=clean_whitespace(raw.source_line),
        extraction_method=method,
        record_issues=[f"Extractor note: {raw.ambiguity_note}"] if raw.ambiguity_note else [],
    )


def normalize_extraction(
    extraction: LLMInvoiceExtraction,
    *,
    invoice_id: str,
    source_file: str,
    decimal_sep: str,
    method: str,
) -> list[LineItemRecord]:
    vendor = normalize_vendor(extraction.vendor_name)
    currency = (extraction.currency or "").upper() or None
    if currency and not re.fullmatch(r"[A-Z]{3}", currency):
        currency = detect_currency(currency)
    return [
        normalize_item(item, invoice_id=invoice_id, source_file=source_file, line_index=i, vendor_name=vendor,
                       currency=currency, decimal_sep=decimal_sep, method=method)
        for i, item in enumerate(extraction.items)
    ]
