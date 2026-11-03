"""Deterministic validation, uncertainty flags and full pipeline on real invoices."""

import pytest

from conftest import FakeOpenAIClient, sample_bytes
from models.invoice_schema import (
    DocumentContent,
    InvoiceStatus,
    LineItemRecord,
    LLMInvoiceExtraction,
    PageContent,
    RawLineItem,
    RecordStatus,
)
from services.llm_extractor import LLMConfig
from services.pipeline import dedupe_uploads, process_batch, process_invoice
from services.validator import reconcile, validate_fields


def _rec(**kw):
    base = dict(invoice_id="i", source_file="f.pdf", line_index=0, vendor_name="ABC Industries",
                material_name="Plywood Sheets", material_quantity=500, material_value=250000.0)
    base.update(kw)
    return LineItemRecord(**base)


def test_valid_record_has_no_field_issues():
    assert all(not v for v in validate_fields(_rec()).values())


@pytest.mark.parametrize(
    "kw,field",
    [
        (dict(vendor_name=None), "vendor_name"),
        (dict(material_name=None), "material_name"),
        (dict(material_quantity=None), "material_quantity"),
        (dict(material_quantity=0), "material_quantity"),
        (dict(material_quantity=2.5, unit="pcs"), "material_quantity"),
        (dict(material_value=None, raw_value="12,3,4"), "material_value"),
        (dict(material_value=-50.0), "material_value"),
        (dict(material_value=0.0), "material_value"),
    ],
)
def test_field_problems_are_flagged(kw, field):
    issues = validate_fields(_rec(**kw))
    assert issues[field], f"expected an issue on {field}"


def _doc(text):
    return DocumentContent(filename="t.pdf", page_count=1,
                           pages=[PageContent(page_number=1, text=text, method="digital")])


def test_reconciliation_detects_missing_line():
    doc = _doc("Item A  1  100.00\nItem B  1  50.00\nSubtotal  150.00")
    recs = [_rec(material_value=100.0)]
    ok, msg = reconcile(recs, doc, "150.00", ".")
    assert ok is False and "150.00" in msg
    ok, _ = reconcile([_rec(material_value=100.0), _rec(material_value=50.0)], doc, "150.00", ".")
    assert ok is True


# ----------------------------------------------------------------- the LLM is not trusted blindly
def _run_with_llm(name, extraction):
    cfg = LLMConfig(api_key="test", model="test-model", timeout=5, max_retries=0)
    return process_invoice(name, sample_bytes(name), llm_config=cfg, llm_client=FakeOpenAIClient([extraction]))


def _item(name, qty, value, page=1, line=None):
    return RawLineItem(material_name=name, quantity_text=qty, unit=None, value_text=value, source_page=page,
                       source_line=line, ambiguity_note=None)


COOLBLUE_ITEMS = [
    ("Apple iPad Air Wifi 16 GB Zilver", "1", "€ 399,00"),
    ("Decoded Leather Slim Cover Apple iPad Air 2 Zwart", "1", "€ 69,99"),
    ("Nintendo 3DS XL Wit + Blauw", "1", "€ 189,00"),
    ("Nintendo AC-adapter", "1", "€ 14,99"),
    ("Mario Kart 7 3DS", "1", "€ 44,99"),
]


def test_correct_llm_output_on_real_invoice_is_validated():
    ext = LLMInvoiceExtraction(vendor_name="Coolblue B.V.", vendor_evidence="Coolblue B.V.", currency="EUR",
                               items_total_text="€ 717,97", items=[_item(*i) for i in COOLBLUE_ITEMS])
    inv = _run_with_llm("coolblue1.pdf", ext)
    assert inv.status == InvoiceStatus.COMPLETED
    assert all(r.status == RecordStatus.VALIDATED for r in inv.records)
    assert inv.records[0].material_value == 399.0 and inv.records[0].currency == "EUR"


def test_hallucinated_value_is_flagged():
    items = [_item(*i) for i in COOLBLUE_ITEMS]
    items[2] = _item("Nintendo 3DS XL Wit + Blauw", "1", "€ 198,00")  # transposed digits
    ext = LLMInvoiceExtraction(vendor_name="Coolblue B.V.", vendor_evidence=None, currency="EUR",
                               items_total_text="€ 717,97", items=items)
    inv = _run_with_llm("coolblue1.pdf", ext)
    bad = inv.records[2]
    assert bad.status == RecordStatus.NEEDS_REVIEW
    assert any("not found" in m for m in bad.field_issues["material_value"])
    assert inv.invoice_issues  # totals no longer reconcile
    assert inv.status == InvoiceStatus.NEEDS_REVIEW


def test_swapped_association_is_flagged():
    items = [_item(*i) for i in COOLBLUE_ITEMS]
    items[3] = _item("Nintendo AC-adapter", "1", "€ 44,99")
    items[4] = _item("Mario Kart 7 3DS", "1", "€ 14,99")
    ext = LLMInvoiceExtraction(vendor_name="Coolblue B.V.", vendor_evidence=None, currency="EUR",
                               items_total_text="€ 717,97", items=items)
    inv = _run_with_llm("coolblue1.pdf", ext)
    for r in inv.records[3:]:
        assert any("same row" in m for m in r.field_issues["material_value"])


def test_invented_vendor_is_flagged():
    ext = LLMInvoiceExtraction(vendor_name="Mediamarkt GmbH", vendor_evidence=None, currency="EUR",
                               items_total_text="€ 717,97", items=[_item(*i) for i in COOLBLUE_ITEMS])
    inv = _run_with_llm("coolblue1.pdf", ext)
    assert all(r.field_issues["vendor_name"] for r in inv.records)


def test_llm_failure_falls_back_and_flags():
    import httpx2 as httpx
    import openai

    err = openai.APITimeoutError(request=httpx.Request("POST", "http://x"))
    cfg = LLMConfig(api_key="test", model="m", timeout=5, max_retries=0)
    inv = process_invoice("coolblue1.pdf", sample_bytes("coolblue1.pdf"), llm_config=cfg,
                          llm_client=FakeOpenAIClient([err]))
    assert inv.extraction_engine == "Rule-based fallback"
    assert inv.records and all(r.status == RecordStatus.NEEDS_REVIEW for r in inv.records)


# ----------------------------------------------------------------- batch behaviour on real invoices
def test_batch_isolates_failures_and_vendors():
    files = [
        ("QualityHosting.pdf", sample_bytes("QualityHosting.pdf")),
        ("broken.pdf", b"%PDF-1.4 garbage"),
        ("coolblue1.pdf", sample_bytes("coolblue1.pdf")),
        ("coolblue1_again.pdf", sample_bytes("coolblue1.pdf")),
    ]
    results, skipped = process_batch(files)
    assert skipped == ["coolblue1_again.pdf"]
    by_name = {r.filename: r for r in results}
    assert by_name["broken.pdf"].status == InvoiceStatus.FAILED
    assert "corrupted" in by_name["broken.pdf"].error
    assert {r.vendor_name for r in by_name["QualityHosting.pdf"].records} == {"QualityHosting AG"}
    assert {r.vendor_name for r in by_name["coolblue1.pdf"].records} == {"Coolblue B.V."}


def test_dedupe_uploads_by_content():
    unique, skipped = dedupe_uploads([("a.pdf", b"1"), ("b.pdf", b"1"), ("c.pdf", b"2")])
    assert [n for n, _ in unique] == ["a.pdf", "c.pdf"] and skipped == ["b.pdf"]


def test_zero_value_and_missing_quantity_need_review():
    inv = process_invoice("NetpresseInvoice.pdf", sample_bytes("NetpresseInvoice.pdf"))
    first = inv.records[0]
    assert first.material_quantity is None and first.field_issues["material_quantity"]
    assert inv.status == InvoiceStatus.NEEDS_REVIEW


def test_identical_legitimate_lines_are_not_duplicates():
    inv = process_invoice("QualityHosting.pdf", sample_bytes("QualityHosting.pdf"))
    assert inv.status == InvoiceStatus.COMPLETED
    assert not any("duplicate" in m.lower() for r in inv.records for m in r.record_issues)


def test_extracted_twice_is_flagged_as_duplicate():
    items = [_item(*i) for i in COOLBLUE_ITEMS] + [_item(*COOLBLUE_ITEMS[4])]
    ext = LLMInvoiceExtraction(vendor_name="Coolblue B.V.", vendor_evidence=None, currency="EUR",
                               items_total_text="€ 717,97", items=items)
    inv = _run_with_llm("coolblue1.pdf", ext)
    assert any("duplicate" in m.lower() for m in inv.records[-1].record_issues)
