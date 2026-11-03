"""Consolidated Excel export: exactly the four business columns, nothing else.

Quantity and value cells stay real numbers (so Finance can sum and filter them);
the unit and currency are shown in the same cell through its number format,
e.g. 10 displayed as "10 kg" and 44.99 as "EUR 44.99".
"""

from __future__ import annotations

import io

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from models.invoice_schema import EXPORT_COLUMNS

SHEET_NAME = "Invoice Line Items"


def _quoted(label: str) -> str:
    return '"' + label.replace('"', "'") + '"'


def build_excel(df: pd.DataFrame, units: list | None = None, currencies: list | None = None) -> bytes:
    """Return .xlsx bytes for the consolidated records.

    ``units`` / ``currencies`` are aligned with the rows of ``df`` (None = not printed).

    Raises ValueError if the frame does not have exactly the export columns, so
    internal metadata can never leak into the Finance file by accident.
    """
    if list(df.columns) != EXPORT_COLUMNS:
        raise ValueError(f"Export must have exactly these columns: {EXPORT_COLUMNS}")
    df = df.copy()
    df["Material Quantity"] = pd.to_numeric(df["Material Quantity"], errors="raise")
    df["Material Value"] = pd.to_numeric(df["Material Value"], errors="raise")
    units = list(units) if units is not None else [None] * len(df)
    currencies = list(currencies) if currencies is not None else [None] * len(df)
    if len(units) != len(df) or len(currencies) != len(df):
        raise ValueError("units and currencies must have one entry per row")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=SHEET_NAME)
        ws = writer.sheets[SHEET_NAME]
        header_fill = PatternFill("solid", fgColor="DDE7F3")
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.freeze_panes = "A2"

        shown = {col: [str(v) for v in df[col].tolist()] for col in EXPORT_COLUMNS}
        for i, (qty_cell, value_cell) in enumerate(ws.iter_rows(min_row=2, min_col=3, max_col=4)):
            q = qty_cell.value
            fmt = "#,##0" if isinstance(q, (int, float)) and float(q).is_integer() else "#,##0.####"
            qty_cell.number_format = f"{fmt} {_quoted(units[i])}" if units[i] else fmt
            value_cell.number_format = f"{_quoted(currencies[i] + ' ')}#,##0.00" if currencies[i] else "#,##0.00"
            shown["Material Quantity"][i] += f" {units[i]}" if units[i] else ""
            shown["Material Value"][i] = f"{currencies[i] or ''} {value_cell.value:,.2f}"

        for idx, col in enumerate(EXPORT_COLUMNS, start=1):
            values = [str(col)] + shown[col]
            width = min(max(len(v) for v in values) + 3, 60)
            ws.column_dimensions[get_column_letter(idx)].width = max(width, 16)
        for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
    return buf.getvalue()
