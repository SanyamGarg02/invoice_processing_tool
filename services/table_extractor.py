"""Optional ruled-table detection with pdfplumber (digital pages only).

pdfplumber's table finder works well when an invoice draws cell borders, and
then gives cleanly separated cells. Many invoices have no ruling lines, in which
case it finds nothing (or junk), so tables are only used as *supplementary hints*
for the LLM next to the layout text – never as the sole source.
"""

from __future__ import annotations

import io

import pdfplumber

from utils.helpers import get_logger

log = get_logger(__name__)


def _to_markdown(rows: list[list[str | None]]) -> str:
    clean = [[(c or "").replace("\n", " ").strip() for c in row] for row in rows]
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    header, body = clean[0], clean[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def extract_tables(pdf_bytes: bytes, page_numbers: list[int]) -> dict[int, list[str]]:
    """Return {1-based page number: [markdown table, ...]} for plausible tables."""
    found: dict[int, list[str]] = {}
    if not page_numbers:
        return found
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for pno in page_numbers:
                page = pdf.pages[pno - 1]
                tables = page.extract_tables({"vertical_strategy": "lines", "horizontal_strategy": "lines"})
                good = []
                for t in tables:
                    rows = [r for r in t if r and any((c or "").strip() for c in r)]
                    # A line-item table needs a header plus at least one row with
                    # >= 3 filled cells (description, quantity, amount). Tables
                    # where everything collapsed into one cell are useless.
                    dense = [r for r in rows if sum(1 for c in r if (c or "").strip()) >= 3]
                    if len(dense) >= 2:
                        good.append(_to_markdown(rows))
                if good:
                    found[pno] = good
    except Exception as exc:  # table detection is best-effort
        log.warning("pdfplumber table extraction failed: %s", exc)
    return found
