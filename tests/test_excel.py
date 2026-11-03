import io

import pandas as pd
import pytest
from openpyxl import load_workbook

from models.invoice_schema import EXPORT_COLUMNS, InvoiceResult, LineItemRecord, RecordStatus
from services.excel_generator import build_excel
from services.review_service import export_frame


def _inv(vendor, items, status=RecordStatus.VALIDATED):
    inv = InvoiceResult(invoice_id=vendor[:3], filename=f"{vendor}.pdf", file_hash=vendor)
    inv.records = [
        LineItemRecord(invoice_id=inv.invoice_id, source_file=inv.filename, line_index=i, vendor_name=vendor,
                       material_name=n, material_quantity=q, material_value=v, status=status, source_page=1,
                       unit="pcs", record_issues=["internal note"])
        for i, (n, q, v) in enumerate(items)
    ]
    return inv


def test_row_per_line_item_and_exactly_four_columns():
    invoices = [
        _inv("ABC Industries", [("Plywood Sheets", 500, 250000), ("Industrial Adhesive", 100, 45000),
                                ("Hardwood Panels", 250, 125000)]),
        _inv("XYZ Suppliers", [("Industrial Resin", 50, 30000), ("MDF Boards", 200, 80000)]),
    ]
    df = export_frame(invoices)
    data = build_excel(df)
    wb = load_workbook(io.BytesIO(data))
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    assert list(rows[0]) == EXPORT_COLUMNS
    assert len(rows) == 6
    assert rows[1] == ("ABC Industries", "Plywood Sheets", 500, 250000)
    assert rows[5] == ("XYZ Suppliers", "MDF Boards", 200, 80000)
    assert ws.max_column == 4  # no metadata columns leaked


def test_formatting():
    df = export_frame([_inv("ABC", [("Bolt M8", 1000, 1234.5), ("Oil", 2.5, 10)])])
    ws = load_workbook(io.BytesIO(build_excel(df))).active
    assert all(c.font.bold for c in ws[1])
    assert ws.freeze_panes == "A2"
    assert ws["D2"].number_format == "#,##0.00"
    assert ws["C2"].number_format == "#,##0"
    assert isinstance(ws["D2"].value, (int, float))
    assert ws.column_dimensions["B"].width >= 16


def test_extra_columns_are_refused():
    df = pd.DataFrame([["A", "B", 1, 2, "file.pdf"]], columns=EXPORT_COLUMNS + ["Invoice File"])
    with pytest.raises(ValueError):
        build_excel(df)


def test_only_validated_or_verified_records_are_exported():
    ok = _inv("ABC", [("Bolt", 1, 1.0)])
    pending = _inv("XYZ", [("Nut", 1, 2.0)], status=RecordStatus.NEEDS_REVIEW)
    verified = _inv("QRS", [("Washer", 3, 3.0)], status=RecordStatus.VERIFIED)
    rejected = _inv("TUV", [("Subtotal", 1, 6.0)], status=RecordStatus.REJECTED)
    df = export_frame([ok, pending, verified, rejected])
    assert df["Material Name"].tolist() == ["Bolt", "Washer"]


def test_empty_export_still_has_headers():
    ws = load_workbook(io.BytesIO(build_excel(export_frame([])))).active
    assert [c.value for c in ws[1]] == EXPORT_COLUMNS


def test_unit_and_currency_shown_in_the_same_numeric_cells():
    from services.review_service import export_labels

    inv = InvoiceResult(invoice_id="mix", filename="mix.pdf", file_hash="mix")
    rows = [("Sugar", 10, "g", 5.5, "EUR"), ("Cement", 10, "kg", 120, "INR"), ("Oil", 10, "L", 44.99, "USD"),
            ("Bolts", 10, "units", 3, None), ("Service", 1.5, None, 99.9, "EUR")]
    inv.records = [
        LineItemRecord(invoice_id="mix", source_file="mix.pdf", line_index=i, vendor_name="V", material_name=n,
                       material_quantity=q, unit=u, material_value=v, currency=c, status=RecordStatus.VALIDATED)
        for i, (n, q, u, v, c) in enumerate(rows)
    ]
    units, currencies = export_labels([inv])
    ws = load_workbook(io.BytesIO(build_excel(export_frame([inv]), units=units, currencies=currencies))).active
    assert ws.max_column == 4  # no separate unit / currency columns
    assert [c.value for c in ws["C"][1:]] == [10, 10, 10, 10, 1.5]  # still numbers Finance can sum
    assert [c.number_format for c in ws["C"][1:]] == [
        '#,##0 "g"', '#,##0 "kg"', '#,##0 "L"', '#,##0 "units"', "#,##0.####"]
    assert [c.number_format for c in ws["D"][1:]] == [
        '"EUR "#,##0.00', '"INR "#,##0.00', '"USD "#,##0.00', "#,##0.00", '"EUR "#,##0.00']
    assert export_frame([inv], display=True)["Material Quantity"].tolist() == ["10 g", "10 kg", "10 L", "10 units", "1.5"]
    assert export_frame([inv], display=True)["Material Value"].tolist()[:2] == ["EUR 5.50", "INR 120.00"]
