"""Human-in-the-loop review: apply corrections, verify/reject records, build export data."""

from __future__ import annotations

import pandas as pd

from models.invoice_schema import EXPORT_COLUMNS, InvoiceResult, LineItemRecord, RecordStatus
from services.normalizer import (
    format_quantity,
    format_value,
    normalize_currency,
    normalize_material,
    normalize_unit,
    normalize_vendor,
    split_labelled_number,
)
from services.validator import finalize_status, invoice_status, validate_fields
from utils.helpers import parse_number


def _to_number(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return parse_number(str(value))


def apply_correction(
    rec: LineItemRecord,
    *,
    vendor_name: str | None,
    material_name: str | None,
    material_quantity,
    material_value,
    verified: bool,
) -> list[str]:
    """Apply a reviewer's edits; returns blocking problems (empty list == saved).

    A record becomes VERIFIED only when the reviewer ticks "verified" AND the
    corrected values pass the basic field checks (present, numeric). Warnings
    such as "negative value" do not block a reviewer from confirming a credit
    line – the human decision overrides soft checks, never missing data.

    Text input carries its unit / currency ("10 kg", "EUR 44.99"); what the
    reviewer typed replaces them. A plain number keeps the stored unit / currency.
    """
    rec.vendor_name = normalize_vendor(vendor_name)
    rec.material_name = normalize_material(material_name)
    if isinstance(material_quantity, str):
        rec.material_quantity, unit = split_labelled_number(material_quantity)
        rec.unit = normalize_unit(unit)
    else:
        rec.material_quantity = _to_number(material_quantity)
    if isinstance(material_value, str):
        rec.material_value, currency = split_labelled_number(material_value)
        rec.currency = normalize_currency(currency)
    else:
        rec.material_value = _to_number(material_value)
    rec.raw_quantity = None if rec.material_quantity is None else rec.raw_quantity
    rec.raw_value = None if rec.material_value is None else rec.raw_value
    rec.reviewed_by_user = True

    issues = validate_fields(rec)
    blocking = []
    if not rec.vendor_name:
        blocking.append("Vendor Name is required.")
    if not rec.material_name:
        blocking.append("Material Name is required.")
    if rec.material_quantity is None:
        blocking.append("Material Quantity must be a number.")
    if rec.material_value is None:
        blocking.append("Material Value must be a number.")

    rec.field_issues = issues
    if verified and not blocking:
        rec.status = RecordStatus.VERIFIED
    else:
        rec.status = RecordStatus.NEEDS_REVIEW
    return blocking


def reject_record(rec: LineItemRecord) -> None:
    """Exclude a record (e.g. a subtotal the extractor mistook for an item)."""
    rec.status = RecordStatus.REJECTED
    rec.reviewed_by_user = True


def restore_record(rec: LineItemRecord) -> None:
    rec.status = RecordStatus.NEEDS_REVIEW


def apply_vendor_to_invoice(inv: InvoiceResult, vendor_name: str) -> None:
    """Set the vendor on every record of one invoice (vendor is invoice-level)."""
    vendor = normalize_vendor(vendor_name)
    inv.vendor_name = vendor
    for r in inv.records:
        r.vendor_name = vendor
        r.field_issues["vendor_name"] = validate_fields(r)["vendor_name"]
        if r.status == RecordStatus.NEEDS_REVIEW and not r.reviewed_by_user:
            # A record whose only problem was the vendor is now valid.
            finalize_status(r)


def add_manual_record(inv: InvoiceResult) -> LineItemRecord:
    """Let a reviewer add a line the extractor missed."""
    rec = LineItemRecord(
        invoice_id=inv.invoice_id,
        source_file=inv.filename,
        line_index=len(inv.records),
        vendor_name=inv.vendor_name,
        currency=inv.currency,
        extraction_method="manual",
        record_issues=["Added manually by reviewer."],
    )
    rec.field_issues = validate_fields(rec)
    inv.records.append(rec)
    return rec


def refresh_invoice(inv: InvoiceResult) -> None:
    inv.status = invoice_status(inv)


def records_needing_review(invoices: list[InvoiceResult]) -> list[LineItemRecord]:
    return [r for inv in invoices for r in inv.records if r.status == RecordStatus.NEEDS_REVIEW]


def records_frame(invoices: list[InvoiceResult], include_meta: bool = False) -> pd.DataFrame:
    """All non-rejected records (for display), optionally with metadata columns."""
    rows = []
    for inv in invoices:
        for r in inv.records:
            if r.status == RecordStatus.REJECTED:
                continue
            row = {
                "Vendor Name": r.vendor_name,
                "Material Name": r.material_name,
                "Material Quantity": format_quantity(r.material_quantity, r.unit),
                "Material Value": format_value(r.material_value, r.currency),
            }
            if include_meta:
                row.update({
                    "Status": r.status.value,
                    "Invoice File": r.source_file,
                    "Page": r.source_page,
                    "Issues": "; ".join(r.all_issues),
                })
            rows.append(row)
    cols = EXPORT_COLUMNS + (["Status", "Invoice File", "Page", "Issues"] if include_meta else [])
    return pd.DataFrame(rows, columns=cols)


def export_records(invoices: list[InvoiceResult]) -> list[LineItemRecord]:
    """Only validated or manually verified records, in invoice/line order."""
    return [r for inv in invoices for r in sorted(inv.records, key=lambda x: x.line_index) if r.exportable]


def export_frame(invoices: list[InvoiceResult], display: bool = False) -> pd.DataFrame:
    """Exactly the four business columns. Numbers stay numeric unless ``display`` is set,
    which shows quantity with unit and value with currency ("10 kg", "EUR 44.99")."""
    rows = [
        [r.vendor_name, r.material_name,
         format_quantity(r.material_quantity, r.unit) if display else r.material_quantity,
         format_value(r.material_value, r.currency) if display else r.material_value]
        for r in export_records(invoices)
    ]
    return pd.DataFrame(rows, columns=EXPORT_COLUMNS)


def export_labels(invoices: list[InvoiceResult]) -> tuple[list[str | None], list[str | None]]:
    """(units, currencies) of the exported rows, aligned with ``export_frame``."""
    recs = export_records(invoices)
    return [r.unit for r in recs], [r.currency for r in recs]


def excluded_count(invoices: list[InvoiceResult]) -> int:
    return sum(1 for inv in invoices for r in inv.records if r.status == RecordStatus.NEEDS_REVIEW)

