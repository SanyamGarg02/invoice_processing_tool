"""Streamlit front end: upload -> process -> review -> export.

All business logic lives in ``services/``; this file only handles UI and
session state.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from models.invoice_schema import FIELD_NAMES, InvoiceResult, InvoiceStatus, RecordStatus  # noqa: E402
from services.excel_generator import build_excel  # noqa: E402
from services.llm_extractor import LLMConfig  # noqa: E402
from services.normalizer import format_quantity, format_value  # noqa: E402
from services.ocr_service import ocr_available  # noqa: E402
from services.pdf_extractor import render_page_png  # noqa: E402
from services.pipeline import dedupe_uploads, extraction_mode, process_invoice  # noqa: E402
from services.review_service import (  # noqa: E402
    add_manual_record,
    apply_correction,
    apply_vendor_to_invoice,
    excluded_count,
    export_frame,
    export_labels,
    records_frame,
    refresh_invoice,
    reject_record,
    restore_record,
)
from utils.helpers import human_size  # noqa: E402

st.set_page_config(page_title="AI-Powered Invoice Processing System", page_icon="🧾", layout="wide")

FIELD_LABELS = {
    "vendor_name": "Vendor Name",
    "material_name": "Material Name",
    "material_quantity": "Material Quantity",
    "material_value": "Material Value",
}
STATUS_ICON = {
    InvoiceStatus.UPLOADED: "📄", InvoiceStatus.EXTRACTING: "🔎", InvoiceStatus.PROCESSING: "🤖",
    InvoiceStatus.VALIDATING: "🧮", InvoiceStatus.COMPLETED: "✅", InvoiceStatus.NEEDS_REVIEW: "⚠️",
    InvoiceStatus.FAILED: "❌",
}
MAX_FILE_MB = 25

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .flag {background:#fff4e5;border-left:4px solid #f0a020;padding:.35rem .6rem;margin:.15rem 0;border-radius:4px;color:#5c3b00;font-size:.9rem}
      .ok {background:#eaf7ee;border-left:4px solid #2e9e57;padding:.35rem .6rem;border-radius:4px;color:#14532d;font-size:.9rem}
      .evidence {font-family:ui-monospace,Menlo,monospace;font-size:.8rem;white-space:pre-wrap;background:#f6f8fa;color:#1f2328;padding:.5rem;border-radius:4px;max-height:220px;overflow:auto}
    </style>
    """,
    unsafe_allow_html=True,
)

ss = st.session_state
ss.setdefault("invoices", [])  # list[InvoiceResult]
ss.setdefault("skipped_duplicates", [])
ss.setdefault("rejected_uploads", [])
ss.setdefault("uploader_key", 0)
ss.setdefault("pdf_bytes", {})  # invoice_id -> original PDF (for evidence view)
# Bumped whenever records change outside their own form, so form widgets reload
# their values instead of keeping stale widget state.
ss.setdefault("form_version", 0)


def _invoices() -> list[InvoiceResult]:
    return ss["invoices"]


# --------------------------------------------------------------------------- #
# Header + sidebar
# --------------------------------------------------------------------------- #
st.title("🧾 AI-Powered Invoice Processing System")
st.caption("Automating manual invoice checking and data extraction for Finance operations.")

cfg = LLMConfig.from_env()
mode = extraction_mode(cfg)
with st.sidebar:
    st.header("System status")
    if mode == "llm":
        n_keys = len(cfg.key_pool)
        keys = f" · {n_keys} API keys (auto-rotation)" if n_keys > 1 else ""
        st.success(f"AI extraction enabled\n\nProvider: {cfg.label} · Model: `{cfg.model}`{keys}")
    else:
        st.warning(
            "AI extraction is **off** (no `OPENAI_API_KEY` or `GEMINI_API_KEY` in `.env`). "
            "Invoices are read by the rule-based fallback parser, which handles fewer layouts. "
            "All results are still validated."
        )
    if ocr_available():
        st.success("OCR (Tesseract) available for scanned PDFs")
    else:
        st.error("Tesseract not found – scanned PDFs cannot be read. Run `brew install tesseract`.")
    st.divider()
    st.markdown(
        "**How it works**\n\n1. Upload PDF invoices\n2. Click *Process Invoices*\n"
        "3. Review flagged records\n4. Download the consolidated Excel"
    )
    if _invoices() and st.button("Start a new batch", width="stretch"):
        ss["invoices"], ss["skipped_duplicates"], ss["rejected_uploads"], ss["pdf_bytes"] = [], [], [], {}
        ss["uploader_key"] += 1
        st.rerun()

# --------------------------------------------------------------------------- #
# A/B. Upload
# --------------------------------------------------------------------------- #
st.subheader("1. Upload invoices")
uploads = st.file_uploader(
    "Select one or more PDF invoices (other file types are not accepted)",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"uploader_{ss['uploader_key']}",
    help="You can remove a file with the ✕ next to it before processing.",
)

valid_files: list[tuple[str, bytes]] = []
if uploads:
    rows = []
    for f in uploads:
        data = f.getvalue()
        problem = None
        if not f.name.lower().endswith(".pdf"):
            problem = "Not a PDF file"
        elif len(data) == 0:
            problem = "Empty file"
        elif len(data) > MAX_FILE_MB * 1024 * 1024:
            problem = f"Larger than {MAX_FILE_MB} MB"
        elif b"%PDF" not in data[:1024]:
            problem = "File content is not a PDF"
        rows.append({"File name": f.name, "Size": human_size(len(data)),
                     "Check": "Ready" if problem is None else f"Rejected – {problem}"})
        if problem is None:
            valid_files.append((f.name, data))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    unique, dups = dedupe_uploads(valid_files)
    if dups:
        st.info(f"Duplicate upload(s) will be processed only once: {', '.join(dups)}")

# --------------------------------------------------------------------------- #
# C. Processing
# --------------------------------------------------------------------------- #
st.subheader("2. Process")
process = st.button("Process Invoices", type="primary", disabled=not valid_files, width="stretch")

if process and valid_files:
    unique, dups = dedupe_uploads(valid_files)
    ss["skipped_duplicates"] = dups
    results: list[InvoiceResult] = []
    overall = st.progress(0.0, text="Starting...")
    for i, (name, data) in enumerate(unique):
        box = st.status(f"{STATUS_ICON[InvoiceStatus.UPLOADED]} {name} — Uploaded", expanded=False)

        def on_status(status: InvoiceStatus, message: str, _box=box, _name=name) -> None:
            final = status in (InvoiceStatus.COMPLETED, InvoiceStatus.NEEDS_REVIEW, InvoiceStatus.FAILED)
            state = "complete" if status == InvoiceStatus.COMPLETED else (
                "error" if status == InvoiceStatus.FAILED else ("complete" if final else "running"))
            _box.update(label=f"{STATUS_ICON[status]} {_name} — {status.value}", state=state)
            _box.write(f"**{status.value}:** {message}")

        result = process_invoice(name, data, on_status=on_status)
        ss["pdf_bytes"][result.invoice_id] = data
        results.append(result)
        overall.progress((i + 1) / len(unique), text=f"Processed {i + 1} of {len(unique)} invoice(s)")
    overall.empty()
    ss["invoices"] = results
    ss["pdf_bytes"] = {k: v for k, v in ss["pdf_bytes"].items() if any(r.invoice_id == k for r in results)}
    st.rerun()

invoices = _invoices()
if not invoices:
    st.info("Upload PDF invoices and click **Process Invoices** to begin.")
    st.stop()

for inv in invoices:
    refresh_invoice(inv)

# --------------------------------------------------------------------------- #
# D. Results
# --------------------------------------------------------------------------- #
st.subheader("3. Results")
n_success = sum(1 for i in invoices if i.status == InvoiceStatus.COMPLETED)
n_failed = sum(1 for i in invoices if i.status == InvoiceStatus.FAILED)
all_records = [r for i in invoices for r in i.records if r.status != RecordStatus.REJECTED]
n_review = sum(1 for r in all_records if r.status == RecordStatus.NEEDS_REVIEW)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Invoices uploaded", len(invoices) + len(ss["skipped_duplicates"]))
c2.metric("Successfully processed", n_success, help="All records validated or verified")
c3.metric("Material line items", len(all_records))
c4.metric("Records needing review", n_review)
c5.metric("Failed invoices", n_failed)

summary = pd.DataFrame([
    {
        "Status": f"{STATUS_ICON[i.status]} {i.status.value}",
        "Invoice file": i.filename,
        "Vendor": i.vendor_name or "—",
        "Line items": len([r for r in i.records if r.status != RecordStatus.REJECTED]),
        "Pages": i.page_count or "—",
        "Read via": ", ".join(sorted(set(i.extraction_methods))) or "—",
        "Engine": i.extraction_engine or "—",
        "Message": i.error or ("; ".join(i.invoice_issues) if i.invoice_issues else ""),
    }
    for i in invoices
])
st.dataframe(summary, hide_index=True, width="stretch")

tab_records, tab_details = st.tabs(["Extracted records", "Records with processing details"])
with tab_records:
    st.dataframe(records_frame(invoices), hide_index=True, width="stretch")
with tab_details:
    st.dataframe(records_frame(invoices, include_meta=True), hide_index=True, width="stretch")

failed = [i for i in invoices if i.status == InvoiceStatus.FAILED]
if failed:
    with st.expander(f"❌ Failed invoices ({len(failed)})", expanded=True):
        for i in failed:
            st.error(f"**{i.filename}** — {i.error}")
            for w in i.warnings:
                st.caption(w)
            if i.document is not None and st.button("Enter line items manually", key=f"manual_{i.invoice_id}"):
                add_manual_record(i)
                i.status = InvoiceStatus.NEEDS_REVIEW
                st.rerun()

# --------------------------------------------------------------------------- #
# Manual review
# --------------------------------------------------------------------------- #
st.subheader("4. Manual review")
review_invoices = [i for i in invoices if i.status == InvoiceStatus.NEEDS_REVIEW or any(
    r.reviewed_by_user for r in i.records)]
if not review_invoices:
    st.markdown('<div class="ok">No records require manual review. 🎉</div>', unsafe_allow_html=True)


def _evidence_html(page_text: str, source_line: str | None) -> str:
    """Page text with the record's source row highlighted."""
    out = []
    target = (source_line or "").split()
    for line in page_text.splitlines():
        esc = html.escape(line)
        if target and all(tok in line for tok in target[: min(4, len(target))]):
            esc = f"<mark>{esc}</mark>"
        out.append(esc)
    return f'<div class="evidence">{"<br>".join(out)}</div>'


for inv in review_invoices:
    pending = [r for r in inv.records if r.status == RecordStatus.NEEDS_REVIEW]
    title = f"{STATUS_ICON[inv.status]} {inv.filename} — {len(pending)} record(s) need review"
    with st.expander(title, expanded=bool(pending)):
        st.caption(f"Vendor: **{inv.vendor_name or 'not identified'}** · Pages: {inv.page_count} · "
                   f"Read via: {', '.join(sorted(set(inv.extraction_methods)))} · Engine: {inv.extraction_engine}")
        for msg in inv.invoice_issues:
            st.markdown(f'<div class="flag">Invoice check: {html.escape(msg)}</div>', unsafe_allow_html=True)
        for msg in inv.warnings:
            st.caption(f"ℹ️ {msg}")

        with st.form(f"vendor_{inv.invoice_id}"):
            vc1, vc2 = st.columns([4, 1])
            new_vendor = vc1.text_input("Vendor name for all records of this invoice", value=inv.vendor_name or "")
            vc2.write("")
            if vc2.form_submit_button("Apply to all"):
                apply_vendor_to_invoice(inv, new_vendor)
                ss["form_version"] += 1
                st.rerun()

        if len(pending) > 1:
            with st.form(f"bulk_{inv.invoice_id}"):
                confirm = st.checkbox(
                    f"I have checked all {len(pending)} pending records of this invoice against the original PDF")
                if st.form_submit_button("Verify all pending records as shown"):
                    if not confirm:
                        st.error("Please confirm that you have checked the records first.")
                    else:
                        blocked = 0
                        for r in pending:
                            problems = apply_correction(
                                r, vendor_name=r.vendor_name, material_name=r.material_name,
                                material_quantity=r.material_quantity, material_value=r.material_value, verified=True)
                            blocked += bool(problems)
                        ss["form_version"] += 1
                        if blocked:
                            st.warning(f"{blocked} record(s) have missing values and must be corrected individually.")
                        else:
                            st.rerun()

        for rec in sorted(inv.records, key=lambda r: r.line_index):
            if rec.status == RecordStatus.VALIDATED and not rec.reviewed_by_user:
                continue
            st.markdown("---")
            head = f"**Line {rec.line_index + 1}** · Page {rec.source_page or '?'} · Status: **{rec.status.value}**"
            st.markdown(head)

            if rec.status == RecordStatus.REJECTED:
                st.caption("Excluded from export by reviewer.")
                if st.button("Restore record", key=f"restore_{rec.record_id}"):
                    restore_record(rec)
                    st.rerun()
                continue

            for msg in rec.record_issues:
                st.markdown(f'<div class="flag">{html.escape(msg)}</div>', unsafe_allow_html=True)

            fv = ss["form_version"]
            with st.form(f"form_{rec.record_id}_{fv}"):
                cols = st.columns([2, 3, 1.2, 1.2])
                values = {}
                current = {
                    "vendor_name": rec.vendor_name or "",
                    "material_name": rec.material_name or "",
                    "material_quantity": format_quantity(rec.material_quantity, rec.unit, grouping=False),
                    "material_value": format_value(rec.material_value, rec.currency, grouping=False),
                }
                for col, field in zip(cols, FIELD_NAMES):
                    issues = rec.field_issues.get(field, [])
                    label = ("⚠️ " if issues else "") + FIELD_LABELS[field]
                    values[field] = col.text_input(label, value=current[field], key=f"{field}_{rec.record_id}_{fv}")
                    for msg in issues:
                        col.markdown(f'<div class="flag">{html.escape(msg)}</div>', unsafe_allow_html=True)
                verified = st.checkbox("I have checked this record against the invoice – mark as verified",
                                       value=rec.status == RecordStatus.VERIFIED, key=f"ver_{rec.record_id}_{fv}")
                b1, b2, _ = st.columns([1, 1, 4])
                save = b1.form_submit_button("Save", type="primary")
                exclude = b2.form_submit_button("Exclude record")
                if save:
                    problems = apply_correction(
                        rec,
                        vendor_name=values["vendor_name"],
                        material_name=values["material_name"],
                        material_quantity=values["material_quantity"],
                        material_value=values["material_value"],
                        verified=verified,
                    )
                    if problems:
                        st.error("Not verified: " + " ".join(problems))
                    else:
                        st.rerun()
                if exclude:
                    reject_record(rec)
                    st.rerun()

            if inv.document is not None:
                with st.popover("Show source evidence"):
                    if rec.source_line:
                        st.markdown("**Extracted from this row:**")
                        st.code(rec.source_line, language=None)
                    if rec.raw_quantity or rec.raw_value:
                        st.caption(f"As printed – quantity: `{rec.raw_quantity}` · value: `{rec.raw_value}`")
                    page = rec.source_page if rec.source_page and 1 <= rec.source_page <= inv.page_count else 1
                    p = inv.document.pages[page - 1]
                    conf = f" · OCR confidence {p.ocr_confidence:.0f}%" if p.ocr_confidence else ""
                    st.markdown(f"**Page {page} text** ({p.method}{conf}):")
                    st.markdown(_evidence_html(p.text, rec.source_line), unsafe_allow_html=True)
                    png = render_page_png(ss["pdf_bytes"].get(inv.invoice_id, b""), page)
                    if png:
                        st.markdown(f"**Original page {page}:**")
                        st.image(png, width="stretch")

        if inv.document is not None and st.button("➕ Add a missing line item", key=f"add_{inv.invoice_id}"):
            add_manual_record(inv)
            st.rerun()

# --------------------------------------------------------------------------- #
# E. Export
# --------------------------------------------------------------------------- #
st.subheader("5. Export")
export_df = export_frame(invoices)
excluded = excluded_count(invoices)
if excluded:
    st.warning(f"{excluded} record(s) still need review and are **not** included in the Excel file. "
               "Verify them above to include them.")
if export_df.empty:
    st.info("No validated or verified records to export yet.")
else:
    st.success(f"{len(export_df)} validated/verified record(s) ready for export.")
    st.dataframe(export_frame(invoices, display=True), hide_index=True, width="stretch")
    units, currencies = export_labels(invoices)
    st.download_button(
        "Download Consolidated Excel",
        data=build_excel(export_df, units=units, currencies=currencies),
        file_name="consolidated_invoice_records.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        width="stretch",
    )
