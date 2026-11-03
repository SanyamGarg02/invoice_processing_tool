"""Deterministic validation and uncertainty detection.

Nothing here trusts the extractor. Each record is checked for:

* presence and numeric validity of the four business fields,
* evidence: the material name, quantity and value must be found in the source
  page text, and quantity/value must sit on (or next to) the same row as the
  material name – this is the line-item association check,
* OCR quality of the source page and OCR-noise characters,
* duplicates within the invoice,
* reconciliation: the sum of line values must match a total printed on the
  invoice. This is how missing, extra or misread lines are caught without
  assuming value = quantity x unit price (discounts, taxes, levies...).

Any finding puts the record into "Needs Review" with a human-readable reason.
"""

from __future__ import annotations

import re

from models.invoice_schema import (
    FIELD_NAMES,
    DocumentContent,
    InvoiceResult,
    InvoiceStatus,
    LineItemRecord,
    LLMInvoiceExtraction,
    RecordStatus,
)
from services.normalizer import COUNT_UNITS
from utils.helpers import NUMBER_TOKEN_RE, number_in_text, parse_number, text_support_ratio

LOW_OCR_CONFIDENCE = 70.0
MIN_TEXT_SUPPORT = 0.6
MAX_PLAUSIBLE_VALUE = 1e9
MAX_PLAUSIBLE_QTY = 1e7
TOTAL_LINE_RE = re.compile(
    r"(?i)(total|sub-?total|net|amount due|balance due|sum|totaal|subtotaal|summe|gesamt|zwischensumme|betrag|"
    r"montant|somme|importe|totale|grand|invoice amount|charges)"
)
OCR_NOISE_RE = re.compile(r"[|~^{}\\<>]|[^\w\s]{3,}")


def validate_fields(rec: LineItemRecord) -> dict[str, list[str]]:
    """Field-level checks that only need the record itself (re-used after manual edits)."""
    issues: dict[str, list[str]] = {f: [] for f in FIELD_NAMES}

    if not rec.vendor_name:
        issues["vendor_name"].append("Vendor name is missing.")
    elif len(re.findall(r"[^\W\d_]", rec.vendor_name)) < 2:
        issues["vendor_name"].append("Vendor name looks invalid.")

    if not rec.material_name:
        issues["material_name"].append("Material name is missing.")
    elif len(re.findall(r"[^\W\d_]", rec.material_name)) < 2:
        issues["material_name"].append("Material name contains no meaningful text.")

    q = rec.material_quantity
    if q is None:
        if rec.raw_quantity:
            issues["material_quantity"].append(f"Quantity '{rec.raw_quantity}' could not be read as a number.")
        else:
            issues["material_quantity"].append("Quantity is missing on the invoice line.")
    else:
        if q <= 0:
            issues["material_quantity"].append("Quantity is zero or negative.")
        if abs(q) > MAX_PLAUSIBLE_QTY:
            issues["material_quantity"].append("Quantity is implausibly large.")
        if rec.unit and rec.unit.lower() in COUNT_UNITS and q != int(q):
            issues["material_quantity"].append(f"Fractional quantity for a countable unit ({rec.unit}).")

    v = rec.material_value
    if v is None:
        if rec.raw_value:
            issues["material_value"].append(f"Value '{rec.raw_value}' could not be read as a number.")
        else:
            issues["material_value"].append("Line value is missing on the invoice line.")
    else:
        if v < 0:
            issues["material_value"].append("Negative value (credit note or discount) – please confirm.")
        elif v == 0:
            issues["material_value"].append("Zero value – confirm this is a billable line.")
        if abs(v) > MAX_PLAUSIBLE_VALUE:
            issues["material_value"].append("Value is implausibly large.")
    return issues


def _page_text(doc: DocumentContent, page: int | None) -> str:
    if page and 1 <= page <= doc.page_count:
        return doc.pages[page - 1].text
    return doc.full_text


def _row_window(material: str, page_text: str) -> str:
    """Text of the row(s) that best match the material name.

    Neighbouring rows are only included when the matched row carries no numbers
    of its own (a description wrapped onto several lines); otherwise a value
    from the next item's row could wrongly "confirm" a swapped association.
    """
    lines = page_text.splitlines()
    scored = [(text_support_ratio(material, ln), i) for i, ln in enumerate(lines)]
    if not scored:
        return ""
    best = max(s for s, _ in scored)
    if best < 0.5:
        return ""
    material_tokens = set(NUMBER_TOKEN_RE.findall(material))
    window = set()
    for s, i in scored:
        if s < best - 0.05:
            continue
        window.add(i)
        own_numbers = [t for t in NUMBER_TOKEN_RE.findall(lines[i]) if t not in material_tokens]
        if not own_numbers:
            window.update({i - 1, i + 1})
    return "\n".join(lines[i] for i in sorted(window) if 0 <= i < len(lines))


def _evidence_checks(rec: LineItemRecord, doc: DocumentContent, decimal_sep: str) -> None:
    page_text = _page_text(doc, rec.source_page)
    full = doc.full_text
    if rec.source_page is None or not (1 <= rec.source_page <= doc.page_count):
        rec.record_issues.append("Source page could not be determined.")

    if rec.material_name and text_support_ratio(rec.material_name, full) < MIN_TEXT_SUPPORT:
        rec.field_issues["material_name"].append("Material name not found in the document text (possible misread).")

    window = _row_window(rec.material_name, page_text) if rec.material_name else ""
    # Also accept the evidence line quoted by the extractor if it really exists on the page.
    if (rec.source_line and rec.material_name and text_support_ratio(rec.source_line, page_text) >= 0.8
            and text_support_ratio(rec.material_name, rec.source_line) >= MIN_TEXT_SUPPORT):
        window = window + "\n" + rec.source_line

    for field, value in (("material_value", rec.material_value), ("material_quantity", rec.material_quantity)):
        if value is None:
            continue
        if not number_in_text(value, full, decimal_sep):
            rec.field_issues[field].append("Number not found anywhere in the document (possible misread or invented value).")
        elif window and not number_in_text(value, window, decimal_sep):
            rec.field_issues[field].append("Number is not on the same row as the material – association uncertain.")
        elif not window:
            rec.field_issues[field].append("Could not locate the material's row to confirm this number.")

    page = doc.pages[rec.source_page - 1] if rec.source_page and 1 <= rec.source_page <= doc.page_count else None
    if page is not None and page.method == "ocr":
        if page.ocr_confidence is not None and page.ocr_confidence < LOW_OCR_CONFIDENCE:
            rec.record_issues.append(f"Source page {page.page_number} has low OCR confidence ({page.ocr_confidence:.0f}%).")
        if rec.material_name and OCR_NOISE_RE.search(rec.material_name):
            rec.field_issues["material_name"].append("Contains characters typical of OCR noise.")


def _vendor_checks(records: list[LineItemRecord], doc: DocumentContent) -> None:
    if not records or not records[0].vendor_name:
        return
    vendor = records[0].vendor_name
    if text_support_ratio(vendor, doc.full_text) < MIN_TEXT_SUPPORT:
        for r in records:
            r.field_issues["vendor_name"].append("Vendor name not found in the document text.")


def _duplicate_checks(records: list[LineItemRecord], doc: DocumentContent, decimal_sep: str) -> None:
    """Flag items extracted more often than they actually occur in the document.

    Identical lines are legitimate (e.g. the same licence billed for six users),
    so a duplicate is only suspicious when the extractor returned a line more
    times than there are matching rows in the source text.
    """
    groups: dict[tuple, list[LineItemRecord]] = {}
    for r in records:
        if r.material_name is not None:
            groups.setdefault((r.material_name.lower(), r.material_quantity, r.material_value), []).append(r)
    lines = doc.full_text.splitlines()
    for (name, _qty, value), recs in groups.items():
        if len(recs) < 2:
            continue
        occurrences = sum(
            1 for ln in lines
            if text_support_ratio(name, ln) >= 0.8 and (value is None or number_in_text(value, ln, decimal_sep))
        )
        if len(recs) > max(occurrences, 1):
            for r in recs:
                r.record_issues.append(
                    f"Possible duplicate: extracted {len(recs)} times but found on {occurrences} row(s) of the invoice."
                )


def printed_totals(doc: DocumentContent, decimal_sep: str) -> list[float]:
    """Amounts printed on total-like lines (subtotal, net total, Totaal, Summe, ...)."""
    totals = []
    for line in doc.full_text.splitlines():
        if TOTAL_LINE_RE.search(line):
            for tok in NUMBER_TOKEN_RE.findall(line):
                v = parse_number(tok, decimal_sep)
                if v is not None and v != 0:
                    totals.append(abs(v))
    return totals


def reconcile(records: list[LineItemRecord], doc: DocumentContent, items_total_text: str | None,
              decimal_sep: str) -> tuple[bool | None, str]:
    """Compare the sum of line values with totals printed on the invoice."""
    values = [r.material_value for r in records if r.material_value is not None]
    if not values:
        return None, "No line values to reconcile."
    s = round(sum(values), 2)
    tol = max(0.02, 0.005 * len(values))
    candidates = printed_totals(doc, decimal_sep)
    stated = parse_number(items_total_text, decimal_sep) if items_total_text else None
    if stated is not None and number_in_text(stated, doc.full_text, decimal_sep):
        candidates.insert(0, abs(stated))
    for c in candidates:
        if abs(abs(s) - c) <= tol:
            return True, f"Sum of line values ({s:,.2f}) matches a printed invoice total ({c:,.2f})."
    if not candidates:
        return None, f"No printed total found to reconcile the line values (sum {s:,.2f})."
    ref = f"{stated:,.2f}" if stated is not None else ", ".join(f"{c:,.2f}" for c in sorted(set(candidates))[-3:])
    return False, (f"Sum of extracted line values ({s:,.2f}) does not match the printed total ({ref}) – "
                   "a line may be missing, duplicated or misread.")


def finalize_status(rec: LineItemRecord) -> None:
    if rec.status in (RecordStatus.VERIFIED, RecordStatus.REJECTED):
        return
    rec.status = RecordStatus.NEEDS_REVIEW if rec.all_issues else RecordStatus.VALIDATED


def invoice_status(inv: InvoiceResult) -> InvoiceStatus:
    if inv.status == InvoiceStatus.FAILED:
        return InvoiceStatus.FAILED
    active = [r for r in inv.records if r.status != RecordStatus.REJECTED]
    if not active:
        return InvoiceStatus.NEEDS_REVIEW if inv.records else InvoiceStatus.FAILED
    if any(r.status == RecordStatus.NEEDS_REVIEW for r in active):
        return InvoiceStatus.NEEDS_REVIEW
    return InvoiceStatus.COMPLETED


def validate_invoice(inv: InvoiceResult, doc: DocumentContent, extraction: LLMInvoiceExtraction,
                     decimal_sep: str) -> None:
    """Run all checks on an invoice's records, set field issues and statuses in place."""
    records = inv.records
    for r in records:
        for f, msgs in validate_fields(r).items():
            r.field_issues[f].extend(msgs)
        _evidence_checks(r, doc, decimal_sep)
    _vendor_checks(records, doc)
    _duplicate_checks(records, doc, decimal_sep)

    ok, msg = reconcile(records, doc, extraction.items_total_text, decimal_sep)
    if ok is False:
        inv.invoice_issues.append(msg)
        for r in records:
            r.record_issues.append("Invoice totals do not reconcile (see invoice details).")
    elif ok is None:
        inv.warnings.append(msg)
    else:
        inv.warnings.append(msg)
    inv.items_total = parse_number(extraction.items_total_text, decimal_sep) if extraction.items_total_text else None

    for r in records:
        # de-duplicate messages while keeping order
        r.record_issues = list(dict.fromkeys(r.record_issues))
        r.field_issues = {f: list(dict.fromkeys(m)) for f, m in r.field_issues.items()}
        finalize_status(r)
