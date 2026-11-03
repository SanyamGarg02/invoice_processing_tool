"""End-to-end processing of one invoice and of a batch.

PDF -> page text (digital or OCR) -> structured extraction (LLM, or rule-based
fallback) -> normalisation -> deterministic validation -> InvoiceResult.

Every invoice is processed independently and every exception is converted into
a "Failed" result with a user-friendly message, so one bad file never stops the
batch.
"""

from __future__ import annotations

import os
import uuid
from typing import Callable

from models.invoice_schema import InvoiceResult, InvoiceStatus
from services.llm_extractor import LLMConfig, LLMExtractionError, extract_with_llm
from services.normalizer import detect_document_currency, normalize_extraction
from services.pdf_extractor import PDFValidationError, extract_document
from services.rule_based_extractor import extract_rule_based
from services.validator import invoice_status, validate_invoice
from utils.helpers import detect_decimal_separator, file_sha256, get_logger

log = get_logger(__name__)

StatusCallback = Callable[[InvoiceStatus, str], None]


def extraction_mode(cfg: LLMConfig | None = None) -> str:
    """'llm' when an API key is configured (unless forced off), else 'rule-based'."""
    cfg = cfg or LLMConfig.from_env()
    forced = os.getenv("EXTRACTION_MODE", "auto").lower()
    if forced == "rule-based":
        return "rule-based"
    return "llm" if cfg.enabled else "rule-based"


def process_invoice(
    filename: str,
    data: bytes,
    *,
    on_status: StatusCallback | None = None,
    llm_config: LLMConfig | None = None,
    llm_client=None,
) -> InvoiceResult:
    notify = on_status or (lambda s, m: None)
    inv = InvoiceResult(invoice_id=uuid.uuid4().hex[:12], filename=filename, file_hash=file_sha256(data))
    try:
        # 1. Extract text page by page (digital text layer or OCR).
        inv.status = InvoiceStatus.EXTRACTING
        notify(inv.status, "Reading PDF pages (digital text / OCR)...")
        doc = extract_document(data, filename)
        inv.document = doc
        inv.page_count = doc.page_count
        inv.extraction_methods = [p.method for p in doc.pages]
        inv.warnings.extend(doc.warnings)
        if not doc.has_text:
            raise PDFValidationError("No readable text was found in this PDF, even after OCR.")

        # 2. Structured extraction.
        inv.status = InvoiceStatus.PROCESSING
        decimal_sep = detect_decimal_separator(doc.full_text)
        cfg = llm_config or LLMConfig.from_env()
        mode = "llm" if llm_client is not None else extraction_mode(cfg)
        fallback_reason = None
        if mode == "llm":
            notify(inv.status, f"AI is reading the invoice ({cfg.model})...")
            try:
                extraction = extract_with_llm(doc, cfg, client=llm_client)
                inv.extraction_engine = f"AI ({cfg.label} {cfg.model})"
            except LLMExtractionError as exc:
                log.warning("%s: LLM failed (%s); using rule-based fallback", filename, exc)
                fallback_reason = str(exc)
                mode = "rule-based"
        if mode == "rule-based":
            notify(inv.status, "Extracting line items with the rule-based parser...")
            extraction = extract_rule_based(doc, decimal_sep)
            inv.extraction_engine = "Rule-based fallback"
            if fallback_reason:
                inv.warnings.append(f"AI extraction failed ({fallback_reason}) – used rule-based fallback.")

        if not extraction.currency:
            # Not stated by the extractor: use the currency printed on the invoice itself.
            extraction.currency = detect_document_currency(doc.full_text)
        inv.vendor_name = extraction.vendor_name
        inv.currency = extraction.currency
        inv.records = normalize_extraction(
            extraction, invoice_id=inv.invoice_id, source_file=filename, decimal_sep=decimal_sep, method=mode
        )
        if fallback_reason:
            for r in inv.records:
                r.record_issues.append("AI extraction failed; extracted by the rule-based fallback.")

        # 3. Deterministic validation.
        inv.status = InvoiceStatus.VALIDATING
        notify(inv.status, "Validating extracted records...")
        if not inv.records:
            inv.status = InvoiceStatus.FAILED
            inv.error = "No material line items could be identified in this invoice."
            notify(inv.status, inv.error)
            return inv
        validate_invoice(inv, doc, extraction, decimal_sep)
        inv.status = invoice_status(inv)
        notify(inv.status, f"{len(inv.records)} line item(s) extracted.")
    except PDFValidationError as exc:
        inv.status, inv.error = InvoiceStatus.FAILED, str(exc)
        notify(inv.status, inv.error)
    except Exception:  # never let one invoice crash the batch / leak a trace
        log.exception("Unexpected error while processing %s", filename)
        inv.status = InvoiceStatus.FAILED
        inv.error = "An unexpected error occurred while processing this invoice. See the application log for details."
        notify(inv.status, inv.error)
    return inv


def dedupe_uploads(files: list[tuple[str, bytes]]) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Drop byte-identical files uploaded twice in one batch; returns (unique, skipped names)."""
    seen: set[str] = set()
    unique, skipped = [], []
    for name, data in files:
        h = file_sha256(data)
        if h in seen:
            skipped.append(name)
            continue
        seen.add(h)
        unique.append((name, data))
    return unique, skipped


def process_batch(files: list[tuple[str, bytes]], **kwargs) -> tuple[list[InvoiceResult], list[str]]:
    unique, skipped = dedupe_uploads(files)
    return [process_invoice(name, data, **kwargs) for name, data in unique], skipped
