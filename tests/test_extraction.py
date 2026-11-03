"""PDF reading, OCR routing, number parsing, rule-based and (mocked) LLM extraction."""

import httpx2 as httpx
import openai
import pymupdf
import pytest

from conftest import FakeOpenAIClient, sample_bytes
from models.invoice_schema import LLMInvoiceExtraction, RawLineItem
from services.layout import WordBox, words_to_text
from services.llm_extractor import LLMConfig, LLMExtractionError, chunk_pages, extract_with_llm
from services.ocr_service import ocr_available
from services.pdf_extractor import PDFValidationError, extract_document
from services.rule_based_extractor import extract_rule_based
from utils.helpers import detect_decimal_separator, parse_number


# --------------------------------------------------------------------------- number parsing
@pytest.mark.parametrize(
    "raw,sep,expected",
    [
        ("1,234.56", ".", 1234.56),
        ("1.234,56", ",", 1234.56),
        ("€ 2.321,00", ",", 2321.0),
        ("$4.11", ".", 4.11),
        ("5,39", ",", 5.39),
        ("10,400", ".", 10400.0),  # thousands separator in a dot-decimal document
        ("10,400 pcs", ".", 10400.0),
        ("1 234,50 €", ",", 1234.5),
        ("(100.00)", ".", -100.0),
        ("-9,32", ",", -9.32),
        ("9,32-", ",", -9.32),
        ("Rs 1939", ".", 1939.0),
        ("1.00 kg", ".", 1.0),
        ("0 €", ",", 0.0),
        ("1,234,567", ".", 1234567.0),
    ],
)
def test_parse_number(raw, sep, expected):
    assert parse_number(raw, sep) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "abc", "12/05/2024", "1.2.3,4.5", "see note"])
def test_parse_number_refuses_to_guess(raw):
    assert parse_number(raw) is None


def test_detect_decimal_separator():
    assert detect_decimal_separator("Totaal € 717,97  Subtotaal € 593,36") == ","
    assert detect_decimal_separator("Net total EUR 45,078.62  12.50") == "."


# --------------------------------------------------------------------------- layout
def test_layout_keeps_table_row_together():
    # Words deliberately out of reading order, as content streams often are.
    words = [
        WordBox(400, 100, 430, 110, "5,39"),
        WordBox(50, 100, 60, 110, "1"),
        WordBox(100, 101, 200, 111, "QualityExchange"),
        WordBox(50, 130, 120, 140, "Total"),
        WordBox(400, 130, 430, 140, "34,73"),
    ]
    lines = words_to_text(words).splitlines()
    assert lines[0].split() == ["1", "QualityExchange", "5,39"]
    assert "  " in lines[0]  # column gap preserved as a run of spaces
    assert lines[1].split() == ["Total", "34,73"]


# --------------------------------------------------------------------------- PDF validation
def test_rejects_empty_file():
    with pytest.raises(PDFValidationError, match="empty"):
        extract_document(b"", "empty.pdf")


def test_rejects_non_pdf_content():
    with pytest.raises(PDFValidationError, match="not a valid PDF"):
        extract_document(b"PK\x03\x04 this is a zip/docx", "renamed.pdf")


def test_rejects_corrupted_pdf():
    with pytest.raises(PDFValidationError):
        extract_document(b"%PDF-1.4\nthis is not really a pdf", "corrupted.pdf")


def test_rejects_password_protected_pdf():
    doc = pymupdf.open(stream=sample_bytes("saeco.pdf"), filetype="pdf")
    encrypted = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="secret")
    with pytest.raises(PDFValidationError, match="password"):
        extract_document(encrypted, "locked.pdf")


# --------------------------------------------------------------------------- digital vs OCR routing (real invoices)
def test_digital_pdf_uses_text_layer():
    doc = extract_document(sample_bytes("QualityHosting.pdf"), "QualityHosting.pdf")
    assert doc.page_count == 2
    assert [p.method for p in doc.pages] == ["digital", "digital"]
    # Layout reconstruction keeps description and amount of a row on one line.
    assert any("Small Business StandardExchange 2010" in ln and "3,89" in ln for ln in doc.pages[0].text.splitlines())


@pytest.mark.skipif(not ocr_available(), reason="Tesseract not installed")
def test_scanned_pdf_is_ocrd():
    doc = extract_document(sample_bytes("parseur_scanned_2page.pdf"), "scan.pdf")
    assert [p.method for p in doc.pages] == ["ocr", "ocr"]
    assert all(p.ocr_confidence and p.ocr_confidence > 60 for p in doc.pages)
    assert "45,078.62" in doc.full_text


@pytest.mark.skipif(not ocr_available(), reason="Tesseract not installed")
def test_text_inside_images_is_recovered_on_digital_page():
    # Saeco's invoice prints its company details only inside an image.
    doc = extract_document(sample_bytes("saeco.pdf"), "saeco.pdf")
    assert doc.pages[0].method == "digital"
    assert "Luscious" in doc.full_text


# --------------------------------------------------------------------------- rule-based extraction (real invoices)
@pytest.mark.parametrize("name", ["QualityHosting.pdf", "coolblue1.pdf", "coolblue2.pdf", "AzureInterior.pdf"])
def test_rule_based_matches_ground_truth(name, ground_truth):
    doc = extract_document(sample_bytes(name), name)
    sep = detect_decimal_separator(doc.full_text)
    ext = extract_rule_based(doc, sep)
    expected = ground_truth[name]["items"]
    assert len(ext.items) == len(expected)
    for item, exp in zip(ext.items, expected):
        assert parse_number(item.value_text, sep) == pytest.approx(exp["value"])
        assert parse_number(item.quantity_text, sep) == pytest.approx(exp["quantity"])


def test_rule_based_multi_page_table_continues():
    doc = extract_document(sample_bytes("QualityHosting.pdf"), "QualityHosting.pdf")
    ext = extract_rule_based(doc, ",")
    assert {i.source_page for i in ext.items} == {1, 2}


# --------------------------------------------------------------------------- LLM extraction (mocked client)
def _cfg():
    return LLMConfig(api_key="sk-secret-123", model="test-model", timeout=5, max_retries=0)


def _item(name, qty, value, page=1):
    return RawLineItem(material_name=name, quantity_text=qty, unit=None, value_text=value,
                       source_page=page, source_line=f"{name} {qty} {value}", ambiguity_note=None)


def test_llm_uses_structured_output_schema():
    doc = extract_document(sample_bytes("coolblue1.pdf"), "coolblue1.pdf")
    resp = LLMInvoiceExtraction(vendor_name="Coolblue B.V.", vendor_evidence="Coolblue B.V.", currency="EUR",
                                items_total_text="€ 717,97", items=[_item("Mario Kart 7 3DS", "1", "€ 44,99")])
    client = FakeOpenAIClient([resp])
    out = extract_with_llm(doc, _cfg(), client=client)
    assert out.vendor_name == "Coolblue B.V."
    call = client.calls[0]
    assert call["response_format"] is LLMInvoiceExtraction
    assert call["temperature"] == 0
    assert "Mario Kart 7 3DS" in call["messages"][1]["content"]  # invoice text is sent


def test_llm_long_invoice_is_chunked_by_page(monkeypatch):
    doc = extract_document(sample_bytes("QualityHosting.pdf"), "QualityHosting.pdf")
    assert len(chunk_pages(doc, max_chars=500)) == 2
    monkeypatch.setattr("services.llm_extractor.MAX_CHARS_PER_CALL", 500)
    r1 = LLMInvoiceExtraction(vendor_name="QualityHosting AG", vendor_evidence=None, currency="EUR",
                              items_total_text=None, items=[_item("A", "1", "3,89", 1)])
    r2 = LLMInvoiceExtraction(vendor_name=None, vendor_evidence=None, currency=None,
                              items_total_text="34,73", items=[_item("B", "1", "3,89", 2)])
    client = FakeOpenAIClient([r1, r2])
    out = extract_with_llm(doc, _cfg(), client=client)
    assert [i.material_name for i in out.items] == ["A", "B"]
    assert out.vendor_name == "QualityHosting AG" and out.items_total_text == "34,73"
    assert "part 2 of 2" in client.calls[1]["messages"][1]["content"]


@pytest.mark.parametrize(
    "exc,msg",
    [
        (openai.RateLimitError("rate", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None), "rate limit"),
        (openai.APITimeoutError(request=httpx.Request("POST", "http://x")), "timed out"),
        (openai.AuthenticationError("bad key", response=httpx.Response(401, request=httpx.Request("POST", "http://x")), body=None), "rejected"),
    ],
)
def test_llm_errors_become_friendly_messages(exc, msg, monkeypatch):
    monkeypatch.setattr("services.llm_extractor.time.sleep", lambda s: None)
    doc = extract_document(sample_bytes("coolblue1.pdf"), "coolblue1.pdf")
    with pytest.raises(LLMExtractionError, match=msg) as info:
        extract_with_llm(doc, _cfg(), client=FakeOpenAIClient([exc] * 3))  # rate limits are retried
    assert "sk-secret-123" not in str(info.value)  # the API key is never echoed


def test_schema_is_valid_for_openai_strict_structured_outputs():
    from openai.lib._parsing._completions import type_to_response_format_param

    fmt = type_to_response_format_param(LLMInvoiceExtraction)["json_schema"]
    assert fmt["strict"] is True
    item = fmt["schema"]["$defs"]["RawLineItem"]
    assert set(item["required"]) == set(item["properties"])  # strict mode: every field required (nullable)
    assert item["additionalProperties"] is False


# --------------------------------------------------------------------------- provider selection (Gemini)
def test_gemini_selected_when_only_gemini_key_is_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    cfg = LLMConfig.from_env()
    assert cfg.provider == "gemini" and cfg.enabled
    assert cfg.base_url.startswith("https://generativelanguage.googleapis.com/")
    assert cfg.model.startswith("gemini")


def test_openai_preferred_in_auto_mode_when_both_keys_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "o-test")
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    assert LLMConfig.from_env().provider == "openai"
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert LLMConfig.from_env().provider == "gemini"


def test_gemini_falls_back_to_json_mode_if_strict_schema_rejected():
    doc = extract_document(sample_bytes("coolblue1.pdf"), "coolblue1.pdf")
    cfg = LLMConfig(api_key="g", model="gemini-test", timeout=5, max_retries=0,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", provider="gemini")
    rejected = openai.BadRequestError(
        "schema not supported", response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None)
    good = LLMInvoiceExtraction(vendor_name="Coolblue B.V.", vendor_evidence=None, currency="EUR",
                                items_total_text="€ 717,97", items=[_item("Mario Kart 7 3DS", "1", "€ 44,99")])
    client = FakeOpenAIClient([rejected, good])
    out = extract_with_llm(doc, cfg, client=client)
    assert out.items[0].material_name == "Mario Kart 7 3DS"
    assert client.calls[1]["response_format"] == {"type": "json_object"}
    assert "JSON schema" in client.calls[1]["messages"][0]["content"]


def test_gemini_auth_error_names_the_right_variable():
    doc = extract_document(sample_bytes("coolblue1.pdf"), "coolblue1.pdf")
    cfg = LLMConfig(api_key="g", model="m", timeout=5, max_retries=0, provider="gemini")
    err = openai.AuthenticationError("bad", response=httpx.Response(401, request=httpx.Request("POST", "http://x")), body=None)
    with pytest.raises(LLMExtractionError, match="GEMINI_API_KEY"):
        extract_with_llm(doc, cfg, client=FakeOpenAIClient([err]))


def test_quota_error_is_waited_for_and_retried(monkeypatch):
    monkeypatch.setattr("services.llm_extractor.time.sleep", lambda s: None)
    doc = extract_document(sample_bytes("coolblue1.pdf"), "coolblue1.pdf")
    body = [{"error": {"code": 429, "details": [{"retryDelay": "7s"}]}}]
    quota = openai.RateLimitError("quota", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=body)
    good = LLMInvoiceExtraction(vendor_name="Coolblue B.V.", vendor_evidence=None, currency="EUR",
                                items_total_text=None, items=[_item("Mario Kart 7 3DS", "1", "€ 44,99")])
    from services.llm_extractor import _retry_delay
    assert _retry_delay(quota) == 8.0
    out = extract_with_llm(doc, _cfg(), client=FakeOpenAIClient([quota, good]))
    assert out.items[0].material_name == "Mario Kart 7 3DS"


def test_overloaded_model_gives_clear_message(monkeypatch):
    monkeypatch.setattr("services.llm_extractor.time.sleep", lambda s: None)
    doc = extract_document(sample_bytes("coolblue1.pdf"), "coolblue1.pdf")
    busy = lambda: openai.InternalServerError(
        "busy", response=httpx.Response(503, request=httpx.Request("POST", "http://x")), body=None)
    with pytest.raises(LLMExtractionError, match="overloaded"):
        extract_with_llm(doc, _cfg(), client=FakeOpenAIClient([busy(), busy(), busy()]))


def test_daily_quota_fails_fast_without_waiting(monkeypatch):
    waits = []
    monkeypatch.setattr("services.llm_extractor.time.sleep", waits.append)
    doc = extract_document(sample_bytes("coolblue1.pdf"), "coolblue1.pdf")
    body = [{"error": {"code": 429, "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]},
        {"retryDelay": "80246s"}]}}]
    quota = openai.RateLimitError("quota", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=body)
    with pytest.raises(LLMExtractionError, match="daily .* quota") as info:
        extract_with_llm(doc, _cfg(), client=FakeOpenAIClient([quota]))
    assert "22 hours" in str(info.value)
    assert waits == []


# --------------------------------------------------------------------------- API key rotation
def _rotation_cfg():
    return LLMConfig(api_key="k1", model="m", timeout=5, max_retries=0, provider="gemini",
                     keys=[("GEMINI_API_KEY", "k1"), ("GEMINI_API_KEY_2", "k2"), ("GEMINI_API_KEY_3", "k3")])


def _err(cls, status, body=None):
    return cls("x", response=httpx.Response(status, request=httpx.Request("POST", "http://x")), body=body)


_DAILY = [{"error": {"code": 429, "details": [
    {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}, {"retryDelay": "80000s"}]}}]


def _good():
    return LLMInvoiceExtraction(vendor_name="Coolblue B.V.", vendor_evidence=None, currency="EUR",
                                items_total_text=None, items=[_item("Mario Kart 7 3DS", "1", "€ 44,99")])


def test_all_numbered_keys_are_loaded_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "key-a")
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-b")
    monkeypatch.setenv("GEMINI_API_KEY_3", "key-a")  # duplicate is ignored
    monkeypatch.setenv("GEMINI_API_KEYS", "key-c, key-d")
    cfg = LLMConfig.from_env()
    assert [k for _, k in cfg.key_pool] == ["key-a", "key-b", "key-c", "key-d"]
    assert "key-" not in repr(cfg)  # keys never appear in logs/reprs


def test_daily_quota_rotates_to_next_key_and_later_invoices_skip_it(monkeypatch):
    monkeypatch.setattr("services.llm_extractor.time.sleep", lambda s: pytest.fail("must not wait"))
    doc = extract_document(sample_bytes("coolblue1.pdf"), "coolblue1.pdf")
    clients = {"k1": FakeOpenAIClient([_err(openai.RateLimitError, 429, _DAILY)]),
               "k2": FakeOpenAIClient([_good(), _good()]), "k3": FakeOpenAIClient([])}
    out = extract_with_llm(doc, _rotation_cfg(), client_factory=clients.get)
    assert out.items[0].material_name == "Mario Kart 7 3DS"
    extract_with_llm(doc, _rotation_cfg(), client_factory=clients.get)  # next invoice
    assert len(clients["k1"].calls) == 1 and len(clients["k2"].calls) == 2 and not clients["k3"].calls


def test_rejected_key_rotates_to_next_key():
    doc = extract_document(sample_bytes("coolblue1.pdf"), "coolblue1.pdf")
    clients = {"k1": FakeOpenAIClient([_err(openai.AuthenticationError, 401)]),
               "k2": FakeOpenAIClient([_err(openai.PermissionDeniedError, 403)]),
               "k3": FakeOpenAIClient([_good()])}
    assert extract_with_llm(doc, _rotation_cfg(), client_factory=clients.get).items


def test_overload_tries_other_keys_before_waiting(monkeypatch):
    waits = []
    monkeypatch.setattr("services.llm_extractor.time.sleep", waits.append)
    doc = extract_document(sample_bytes("coolblue1.pdf"), "coolblue1.pdf")
    clients = {"k1": FakeOpenAIClient([_err(openai.InternalServerError, 503)]),
               "k2": FakeOpenAIClient([_good()]), "k3": FakeOpenAIClient([])}
    assert extract_with_llm(doc, _rotation_cfg(), client_factory=clients.get).items
    assert waits == []


def test_all_keys_exhausted_names_each_key_without_leaking_it(monkeypatch):
    monkeypatch.setattr("services.llm_extractor.time.sleep", lambda s: None)
    doc = extract_document(sample_bytes("coolblue1.pdf"), "coolblue1.pdf")
    clients = {k: FakeOpenAIClient([_err(openai.RateLimitError, 429, _DAILY)]) for k in ("k1", "k2")}
    clients["k3"] = FakeOpenAIClient([_err(openai.AuthenticationError, 401)])
    with pytest.raises(LLMExtractionError) as info:
        extract_with_llm(doc, _rotation_cfg(), client_factory=clients.get)
    msg = str(info.value)
    assert msg.startswith("All 3 Gemini API keys failed")
    assert "GEMINI_API_KEY_2: the daily" in msg and "GEMINI_API_KEY_3: the Gemini API key was rejected" in msg
    assert "k1" not in msg.replace("GEMINI_API_KEY", "")


def test_invoice_currency_detected_from_document_text():
    from services.normalizer import detect_document_currency

    assert detect_document_currency("Net total   EUR 45,078.62\nPlease transfer EUR 45,078.62") == "EUR"
    assert detect_document_currency("Promotional Discount of Rs -40.00") == "INR"
    assert detect_document_currency("Amount due $ 12.00") == "USD"
    assert detect_document_currency("Your orders: 3 items") is None  # "rs" inside words is not a currency


def test_gemini_invalid_key_400_rotates_instead_of_json_mode_retry():
    doc = extract_document(sample_bytes("coolblue1.pdf"), "coolblue1.pdf")
    bad = _err(openai.BadRequestError, 400,
               [{"error": {"code": 400, "message": "Please pass a valid API key", "status": "INVALID_ARGUMENT"}}])
    clients = {"k1": FakeOpenAIClient([bad]), "k2": FakeOpenAIClient([_good()]), "k3": FakeOpenAIClient([])}
    assert extract_with_llm(doc, _rotation_cfg(), client_factory=clients.get).items
    assert len(clients["k1"].calls) == 1  # no pointless JSON-mode retry with the same bad key
