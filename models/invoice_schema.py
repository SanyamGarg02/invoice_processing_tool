"""Data models shared across the pipeline.

Two layers of models live here:

1. ``LLMInvoiceExtraction`` – the *raw* structure the LLM (or the rule-based
   fallback) must return. Numbers are kept as the verbatim strings printed on
   the invoice; deterministic Python code parses them later. This keeps the LLM
   in the role of "reader/mapper" and lets us verify every value against the
   source text.

2. ``LineItemRecord`` / ``InvoiceResult`` – the internal, normalised
   representation used by validation, the review UI and Excel export. They carry
   processing metadata (filename, page, evidence, field status...) that is shown
   in the app but never exported to Excel.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# The four business columns, in export order. Single source of truth.
EXPORT_COLUMNS = ["Vendor Name", "Material Name", "Material Quantity", "Material Value"]


# --------------------------------------------------------------------------- #
# Raw extraction schema (LLM structured output)
# --------------------------------------------------------------------------- #
class RawLineItem(BaseModel):
    """One line item exactly as read from the document."""

    model_config = ConfigDict(extra="forbid")

    material_name: Optional[str] = Field(
        description="Product/material/service description of this line, copied from the invoice. null if absent."
    )
    quantity_text: Optional[str] = Field(
        description="Quantity exactly as printed (e.g. '10,400', '1.00', '2'). null if the line has no quantity."
    )
    unit: Optional[str] = Field(
        description="Unit of measure printed next to the quantity (e.g. 'pcs', 'kg', 'roll', 'L'). null if none."
    )
    value_text: Optional[str] = Field(
        description="Line amount / line total for this item exactly as printed incl. sign (e.g. '5,126.16', '€ 399,00', '-9.32'). "
        "This is the line's total value, NOT the unit price. null if not printed."
    )
    source_page: Optional[int] = Field(description="1-based page number where this line item appears.")
    source_line: Optional[str] = Field(
        description="The verbatim text line(s) from the document that contain this item (used as evidence)."
    )
    ambiguity_note: Optional[str] = Field(
        description="Short note if anything about this line is unclear or ambiguous, else null."
    )


class LLMInvoiceExtraction(BaseModel):
    """Structured output requested from the LLM for one invoice (or one page chunk)."""

    model_config = ConfigDict(extra="forbid")

    vendor_name: Optional[str] = Field(
        description="Legal/trading name of the company that ISSUED the invoice (seller/supplier), not the customer. null if unclear."
    )
    vendor_evidence: Optional[str] = Field(
        description="Verbatim text from the document from which the vendor name was taken."
    )
    currency: Optional[str] = Field(description="ISO currency code if identifiable (e.g. EUR, USD, INR), else null.")
    items_total_text: Optional[str] = Field(
        description="The printed total that the line item values should add up to (usually the net subtotal "
        "before tax, or the grand total if line values include tax), exactly as printed. null if not present."
    )
    items: list[RawLineItem]


# --------------------------------------------------------------------------- #
# Document extraction structures
# --------------------------------------------------------------------------- #
class PageContent(BaseModel):
    page_number: int  # 1-based
    text: str  # layout-reconstructed text
    method: str  # "digital" | "ocr" | "empty"
    ocr_confidence: Optional[float] = None  # mean word confidence 0-100 (OCR pages only)
    tables_markdown: list[str] = Field(default_factory=list)
    # Visual rows as (x0, x1, text) segments – used by the rule-based fallback
    # to map numbers to table columns by horizontal position.
    rows: list[list[tuple[float, float, str]]] = Field(default_factory=list)


class DocumentContent(BaseModel):
    filename: str
    page_count: int
    pages: list[PageContent]
    warnings: list[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    @property
    def has_text(self) -> bool:
        return any(p.text.strip() for p in self.pages)


# --------------------------------------------------------------------------- #
# Internal normalised records
# --------------------------------------------------------------------------- #
class RecordStatus(str, Enum):
    VALIDATED = "Validated"  # passed all deterministic checks
    NEEDS_REVIEW = "Needs Review"  # at least one issue, awaiting a human
    VERIFIED = "Verified"  # a Finance user reviewed and approved it
    REJECTED = "Rejected"  # a Finance user excluded it (e.g. not a real item)


class InvoiceStatus(str, Enum):
    UPLOADED = "Uploaded"
    EXTRACTING = "Extracting"
    PROCESSING = "Processing"
    VALIDATING = "Validating"
    COMPLETED = "Completed"
    NEEDS_REVIEW = "Needs Review"
    FAILED = "Failed"


FIELD_NAMES = ("vendor_name", "material_name", "material_quantity", "material_value")


class LineItemRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    invoice_id: str
    source_file: str
    line_index: int  # order within invoice

    # --- the four business fields (normalised) ---
    vendor_name: Optional[str] = None
    material_name: Optional[str] = None
    material_quantity: Optional[float] = None
    material_value: Optional[float] = None

    # --- supporting metadata (never exported) ---
    unit: Optional[str] = None
    currency: Optional[str] = None
    raw_quantity: Optional[str] = None
    raw_value: Optional[str] = None
    source_page: Optional[int] = None
    source_line: Optional[str] = None
    extraction_method: str = "llm"  # "llm" | "rule-based"

    status: RecordStatus = RecordStatus.NEEDS_REVIEW
    # field name -> list of issues for that field (empty list == OK)
    field_issues: dict[str, list[str]] = Field(default_factory=lambda: {f: [] for f in FIELD_NAMES})
    record_issues: list[str] = Field(default_factory=list)  # issues not tied to a single field
    reviewed_by_user: bool = False

    @property
    def all_issues(self) -> list[str]:
        out = list(self.record_issues)
        for f, issues in self.field_issues.items():
            out.extend(f"{f.replace('_', ' ').title()}: {i}" for i in issues)
        return out

    @property
    def flagged_fields(self) -> list[str]:
        return [f for f, issues in self.field_issues.items() if issues]

    @property
    def exportable(self) -> bool:
        return self.status in (RecordStatus.VALIDATED, RecordStatus.VERIFIED)


class InvoiceResult(BaseModel):
    invoice_id: str
    filename: str
    file_hash: str
    status: InvoiceStatus = InvoiceStatus.UPLOADED
    error: Optional[str] = None  # user-friendly failure message
    warnings: list[str] = Field(default_factory=list)
    invoice_issues: list[str] = Field(default_factory=list)  # e.g. totals don't reconcile
    vendor_name: Optional[str] = None
    currency: Optional[str] = None
    items_total: Optional[float] = None
    page_count: int = 0
    extraction_methods: list[str] = Field(default_factory=list)  # per page
    extraction_engine: Optional[str] = None
    records: list[LineItemRecord] = Field(default_factory=list)
    document: Optional[DocumentContent] = None  # kept for evidence display
