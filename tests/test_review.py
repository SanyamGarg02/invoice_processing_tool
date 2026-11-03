from models.invoice_schema import InvoiceResult, InvoiceStatus, LineItemRecord, RecordStatus
from services.review_service import (
    add_manual_record,
    apply_correction,
    apply_vendor_to_invoice,
    excluded_count,
    export_frame,
    refresh_invoice,
    reject_record,
)
from services.validator import validate_fields


def _inv():
    inv = InvoiceResult(invoice_id="inv1", filename="a.pdf", file_hash="h", status=InvoiceStatus.NEEDS_REVIEW)
    inv.records = [
        LineItemRecord(invoice_id="inv1", source_file="a.pdf", line_index=0, vendor_name=None,
                       material_name="Plywood Sheets", material_quantity=None, material_value=250000.0,
                       status=RecordStatus.NEEDS_REVIEW),
        LineItemRecord(invoice_id="inv1", source_file="a.pdf", line_index=1, vendor_name=None,
                       material_name="Hardwood Panels", material_quantity=250, material_value=125000.0,
                       status=RecordStatus.NEEDS_REVIEW),
    ]
    for r in inv.records:  # as the pipeline's validation would have done
        r.field_issues = validate_fields(r)
    return inv


def test_correction_and_verification_updates_export():
    inv = _inv()
    rec = inv.records[0]
    problems = apply_correction(rec, vendor_name="  ABC   Industries ", material_name="Plywood Sheets",
                                material_quantity="500", material_value="2,50,000.00".replace(",", ""),
                                verified=True)
    assert problems == []
    assert rec.status == RecordStatus.VERIFIED
    assert rec.vendor_name == "ABC Industries" and rec.material_quantity == 500
    df = export_frame([inv])
    assert df.values.tolist() == [["ABC Industries", "Plywood Sheets", 500.0, 250000.0]]
    assert excluded_count([inv]) == 1


def test_saving_without_verify_keeps_needs_review():
    rec = _inv().records[1]
    apply_correction(rec, vendor_name="ABC", material_name="Hardwood Panels", material_quantity=250,
                     material_value=125000, verified=False)
    assert rec.status == RecordStatus.NEEDS_REVIEW


def test_cannot_verify_with_missing_values():
    rec = _inv().records[0]
    problems = apply_correction(rec, vendor_name="ABC", material_name="Plywood", material_quantity="",
                                material_value="abc", verified=True)
    assert rec.status == RecordStatus.NEEDS_REVIEW
    assert any("Quantity" in p for p in problems) and any("Value" in p for p in problems)


def test_reviewer_can_confirm_credit_line():
    rec = _inv().records[1]
    problems = apply_correction(rec, vendor_name="ABC", material_name="Credit: damaged panels",
                                material_quantity=1, material_value="-500.00", verified=True)
    assert problems == [] and rec.status == RecordStatus.VERIFIED
    assert rec.field_issues["material_value"]  # the soft warning stays visible


def test_reject_and_invoice_status():
    inv = _inv()
    for r in inv.records:
        apply_correction(r, vendor_name="ABC", material_name=r.material_name, material_quantity=1,
                         material_value=r.material_value, verified=True)
    refresh_invoice(inv)
    assert inv.status == InvoiceStatus.COMPLETED
    reject_record(inv.records[0])
    refresh_invoice(inv)
    assert inv.status == InvoiceStatus.COMPLETED
    assert len(export_frame([inv])) == 1


def test_vendor_applies_to_all_records_and_manual_add():
    inv = _inv()
    apply_vendor_to_invoice(inv, "XYZ Suppliers")
    assert {r.vendor_name for r in inv.records} == {"XYZ Suppliers"}
    assert not inv.records[0].field_issues["vendor_name"]
    assert inv.records[1].status == RecordStatus.VALIDATED  # vendor was its only issue
    assert inv.records[0].status == RecordStatus.NEEDS_REVIEW  # still missing a quantity
    new = add_manual_record(inv)
    assert new.vendor_name == "XYZ Suppliers" and new.status == RecordStatus.NEEDS_REVIEW
    assert not new.exportable


def test_reviewer_edits_unit_and_currency_in_the_same_field():
    inv = _inv()
    rec = inv.records[1]
    rec.unit, rec.currency = "pcs", "EUR"
    assert apply_correction(rec, vendor_name="ABC", material_name="Hardwood Panels", material_quantity="12.5 kg",
                            material_value="44,99 €", verified=True) == []
    assert (rec.material_quantity, rec.unit, rec.material_value, rec.currency) == (12.5, "kg", 44.99, "EUR")
    # A plain number from bulk verify keeps the unit and currency already stored.
    apply_correction(rec, vendor_name="ABC", material_name="Hardwood Panels", material_quantity=12.5,
                     material_value=44.99, verified=True)
    assert (rec.unit, rec.currency) == ("kg", "EUR")
    # Typing a bare number in the field removes the unit.
    apply_correction(rec, vendor_name="ABC", material_name="Hardwood Panels", material_quantity="3",
                     material_value="INR 10", verified=True)
    assert (rec.unit, rec.currency) == (None, "INR")
