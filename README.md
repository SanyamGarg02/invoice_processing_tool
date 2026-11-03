# AI-Powered Invoice Processing System

Automates a manual Finance task. Today, staff open supplier invoices from many vendors, read the vendor and each
material line, and type them into Excel. This app takes many PDF invoices in one upload and extracts four fields:
**Vendor Name, Material Name, Material Quantity and Material Value**. It checks every record with deterministic
rules and sends anything uncertain to a Finance reviewer. It then exports one consolidated Excel file with one row per
material line.

Nothing is built for a particular vendor: there are no templates, no vendor names in the code and no fixed column positions.

---

## 1. Setup (macOS)

```bash
brew install python@3.11 tesseract        # Tesseract = OCR engine for scanned PDFs
```

```bash
cd invoice_processing_tool
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Then put **one** API key in `.env`:
- **Google Gemini:** `GEMINI_API_KEY=...` (key from https://aistudio.google.com/apikey). You can change the model
  with `GEMINI_MODEL`.
- **OpenAI:** `OPENAI_API_KEY=...`. You can change the model with `OPENAI_MODEL`.

**Backup keys (optional):** add `GEMINI_API_KEY_2=`, `GEMINI_API_KEY_3=` … (up to `_9`), or the same with
`OPENAI_API_KEY_2` and so on. When a key is rejected, is not allowed to use the model, or runs out of quota, the app
sets it aside and continues with the next key. Later invoices in the batch skip a key whose daily quota is used up.
If the model is overloaded (503), the other keys are tried first, and the app waits only if they all fail. Keys never
appear in logs or messages; only their variable names do.

With `LLM_PROVIDER=auto` (the default), the app uses OpenAI when an OpenAI key is set, otherwise Gemini. Set
`LLM_PROVIDER=gemini` or `openai` to force one.

For non-English scanned invoices, run `brew install tesseract-lang` and set e.g. `OCR_LANG=eng+deu+fra+nld`.

## 2. Run

```bash
streamlit run app.py
```

Open http://localhost:8501. The sidebar shows whether AI extraction is on, which provider and model it uses, and
whether OCR is available.
**Without an API key the app still works.** It uses a rule-based fallback parser that handles fewer layouts, and the
same validation and review steps apply to its output.

## 3. Tests and evaluation

```bash
python -m pytest -q                         # 79 automated tests (no network, the AI API is mocked)
```

```bash
python scripts/evaluate.py                  # accuracy vs. hand-checked ground truth (uses the LLM if a key is set)
```

```bash
python scripts/evaluate.py --rule-based     # same, forcing the no-AI fallback
```

---

## 4. Project structure

```
app.py                       Streamlit UI only (upload, status, results, review, export)
models/invoice_schema.py     Pydantic models: LLM output schema + internal records + statuses
services/
  pdf_extractor.py           PDF validation, per-page digital-vs-OCR routing, image-region OCR
  layout.py                  Word boxes -> visual rows (keeps table rows together)
  ocr_service.py             Tesseract OCR with confidence scores
  table_extractor.py         pdfplumber ruled-table detection (hints for the LLM)
  llm_extractor.py           OpenAI / Gemini Structured Outputs, prompt, page chunking, error mapping
  rule_based_extractor.py    Template-free fallback (header synonyms + column x-positions)
  normalizer.py              Numbers, units, currency, whitespace -> common structure
  validator.py               Deterministic checks, evidence/association, reconciliation
  review_service.py          Corrections, verify/reject, export dataframe
  excel_generator.py         4-column formatted .xlsx
  pipeline.py                Orchestrates one invoice / a batch, isolates failures
utils/helpers.py             Number parsing, fuzzy text support, hashing, logging
scripts/evaluate.py          Ground-truth evaluation (counts "silent errors")
sample_invoices/             11 public invoices + ground_truth.json
tests/                       test_extraction / test_validation / test_excel / test_review
```

---

## 5. Architecture and data flow

```
 PDF upload (many)
     │  type/size/magic-byte check, SHA-256 de-duplication
     ▼
 ┌──────────────────── per invoice, isolated try/except ───────────────────────┐
 │ 1. EXTRACTING   pdf_extractor                                               │
 │    for each page:  text layer usable? ──yes──► PyMuPDF words                 │
 │                              └──no (scanned / broken encoding)──► Tesseract │
 │    digital pages: OCR embedded images too (logos with the vendor's details) │
 │    word boxes ──► layout.py: group into visual rows, keep column gaps       │
 │    (+ pdfplumber ruled tables as hints)                                     │
 │                                                                             │
 │ 2. PROCESSING   llm_extractor (or rule_based_extractor if no key/failure)   │
 │    page-tagged layout text ──► OpenAI Structured Outputs (Pydantic schema)  │
 │    returns vendor + items with numbers as *verbatim printed strings*,       │
 │    source page and source row                                               │
 │                                                                             │
 │    normalizer: parse numbers deterministically (locale-aware), units,       │
 │    currency, whitespace ──► LineItemRecord (+ metadata)                      │
 │                                                                             │
 │ 3. VALIDATING   validator                                                   │
 │    field checks · evidence in source text · same-row association ·          │
 │    OCR confidence · duplicate check · sum of lines vs printed total         │
 │    ──► each record: Validated | Needs Review (with reasons)                 │
 │    ──► invoice: Completed | Needs Review | Failed                           │
 └─────────────────────────────────────────────────────────────────────────────┘
     ▼
 Streamlit review: edit 4 fields, see reasons + source row + page text + page image,
 verify / exclude / add missing line ──► Verified
     ▼
 Export: only Validated + Verified records ──► 4-column Excel, one row per line item
```

### Role of each layer

| Layer | What it does | Why it is deterministic or AI |
|---|---|---|
| PDF text / OCR | Turns pages into text with layout | Deterministic. OCR runs only where needed |
| LLM | *Reads and maps* the text: works out who the vendor is, which rows are items, and which column holds quantity or amount, in any language or layout | This is where layouts vary without limit, so an LLM is the right tool |
| Normaliser | Parses `1.234,56`, `€ 2.321,00`, `10,400 pcs`, `(100.00)` and similar | Deterministic. Money is never parsed by an LLM |
| Validator | Decides whether a record can be trusted | Deterministic and explainable. LLM self-confidence is **not** used |
| Human review | Final authority on anything flagged | A person makes the call |
| Excel | Fixed four columns | Deterministic. Refuses any other columns |

---

## 6. Explanations

### Why these technologies
- **PyMuPDF**: fast, gives word-level boxes for layout reconstruction, renders pages for OCR and the review view.
- **pdfplumber**: good at ruled tables. Many invoices have no ruling lines, so its tables are only used as hints.
- **Tesseract** (over PaddleOCR): one `brew install`, no large deep-learning dependencies, and it reports per-word
  confidence, which feeds the uncertainty checks.
- **OpenAI Structured Outputs + Pydantic**: the response must match a JSON schema (`strict: true`), and it is validated
  again with Pydantic. The model is set by `OPENAI_MODEL`.
- **Gemini** goes through Google's OpenAI-compatible endpoint
  (`https://generativelanguage.googleapis.com/v1beta/openai/`), so the same client, prompt, schema and checks are
  used for both providers. Google calls this compatibility layer beta. If Gemini rejects the strict schema, the app
  retries once in plain JSON mode with the schema in the prompt and validates the reply with Pydantic.
- **Streamlit**: a web UI for non-technical users with no frontend build. **Pandas/OpenPyXL**: Excel with formatting.

### How unseen vendor layouts are handled
1. **Layout-preserving text.** Content-stream order often scatters a table row. In the QualityHosting sample, the
   quantity, description and amount come out far apart. `layout.py` groups words into rows by their y-position, so
   *description … qty … amount* stays on one line, and gaps of 2 or more spaces mark columns.
2. **The LLM maps by meaning, not column names.** The prompt gives examples ("Particulars", "Omschrijving",
   "Zeilenbetrag", "Extended price") but tells the model to interpret context. It also states general invoice rules
   that hold for any vendor: subtotals, carried-forward lines, tax lines and "incl." sub-lines are not items, and the
   issuer is not the bill-to party.
3. **Validation does not depend on the layout either**, so an unfamiliar invoice is still checked fully.
4. **The fallback parser is also template-free.** It finds the header row with a multilingual synonym vocabulary,
   then assigns each number to a column by its x-position under the header.

### OCR vs. digital extraction
Each page is checked on its own. If it has at least 40 non-space characters and the text is not garbage (broken font
encodings), the text layer is used: exact and instant. Otherwise the page is rendered at 300 DPI and run through
Tesseract (`psm 6`), and the word confidences are kept. Mixed PDFs work page by page. On digital pages, embedded
images are OCR'd separately. In the Saeco invoice the issuer's name and address exist **only** inside an image.

### How LLM extraction works
- Input: page-tagged layout text, e.g. `===== PAGE 2 (OCR) =====`, plus any table hints.
- Output schema (`LLMInvoiceExtraction`): `vendor_name`, `vendor_evidence`, `currency`, `items_total_text`, and
  `items[]` with `material_name`, `quantity_text`, `unit`, `value_text`, `source_page`, `source_line`, `ambiguity_note`.
- **Numbers come back as the printed strings**, not as numbers. Python parses them, so every number can be traced
  back to the document.
- Settings: `temperature=0`, and missing values must be `null`. The prompt forbids calculating or guessing.
- **Long invoices:** pages are grouped into chunks of about 24k characters, and a page is never split. Each chunk gets
  page 1's header as read-only context (for the vendor) and returns items only from its own pages. Results are joined
  in page order, so items from different pages never get mixed. Short invoices go in one call, so a table that runs
  across pages stays together.
- API errors (auth, rate limit, timeout, connection, bad model, schema mismatch) become plain-language messages. The
  SDK retries with backoff (`OPENAI_MAX_RETRIES`). If the LLM fails, the invoice falls back to the rule-based parser
  and **all of its records are flagged**.

### How the structured JSON is validated
1. The OpenAI SDK enforces the JSON schema (strict mode; a test checks the schema is strict-compatible).
2. `LLMInvoiceExtraction.model_validate()` runs again on the result.
3. Deterministic Python checks run on every field (next section).

### How line-item relationships are preserved and checked
- Layout rows keep a description and its numbers on the same line, both for the LLM and for the checks.
- Each item carries its `source_page` and `source_line`.
- **Association check:** the validator finds the row(s) on the source page that best match the material name. The
  quantity and value must appear on that row. Neighbouring rows count only if the matched row has no numbers of its
  own, i.e. a wrapped description. A test confirms that swapping two items' values gets flagged.

### How uncertainty is detected (no reliance on LLM confidence)
A record goes to **Needs Review** when any of these checks fails:

| Check | Example |
|---|---|
| Field presence and validity | missing vendor, quantity not printed, value unparseable |
| Numeric sanity | zero or negative quantity, negative value (credit), zero value, fractional count for `pcs` |
| Evidence | the material name, or a number, does not appear in the document → possible misread or invented value |
| Association | the value is not on the same row as the material |
| Vendor evidence | the vendor name is not found in the document text |
| OCR quality | source page mean confidence < 70%, or OCR noise characters in the name |
| Duplicates | an item extracted more times than it actually appears (legitimate repeated lines are allowed) |
| **Reconciliation** | the sum of line values does not match any printed total (subtotal / net / grand total) |
| Extractor notes | the LLM or parser reported an ambiguity |

Reconciliation is the strongest check. It catches missing, extra or misread lines **without** assuming
value = quantity × unit price, which is often false because of discounts, taxes and levies.

### How human review works
For each invoice that needs review, the review panel shows:
- the source filename, the extraction engine, and the pages read via digital text or OCR
- invoice-level findings, e.g. totals that don't reconcile
- for each record: the reasons it was flagged, the uncertain fields marked ⚠️ with field-level explanations, the
  source page, the source row, the values as printed, the page text with the row highlighted, and an image of the
  original page

The reviewer can then:
- edit the four fields
- tick "verified" and save. The record becomes **Verified** only if all four fields are present and numeric
- exclude a record (e.g. a subtotal mistaken for an item), restore it, or add a missing line
- set the vendor for the whole invoice at once
- verify all pending records of an invoice in bulk, after ticking an explicit confirmation

Records stay in one of three groups: Validated / Needs Review / Failed, plus Verified and Excluded. **Only Validated
and Verified records are exported.** The app shows how many records were left out.

### How the Excel is generated
`review_service.export_frame()` builds a dataframe with exactly `Vendor Name | Material Name | Material Quantity |
Material Value`, one row per line item and vendor repeated on each row, in upload and line order.
`excel_generator.build_excel()` refuses any other columns. It writes real numbers (not text), bold headers, a frozen
header row and sensible column widths. **The unit and currency are shown in the same cell** through a per-cell number
format: the cell holds the number 10 but displays `10 kg`, and 44.99 displays as `EUR 44.99`. The values stay real
numbers, so Finance can still sum and filter them. When the invoice prints no unit or currency, the cell shows the plain
number. When the extractor reports no currency, the app uses the currency printed most often on the invoice. Filenames,
pages, statuses and issues stay in the app only.

---

## 7. Testing with real invoices

`sample_invoices/` contains 11 publicly available PDFs:
- **10** from the MIT-licensed test corpus of the open-source project
  [invoice2data](https://github.com/invoice-x/invoice2data/tree/master/tests/compare): AWS, Flipkart, Coolblue (2),
  Saeco/e-Luscious, QualityHosting, Free, Netpresse, OYO receipt and an Odoo-generated invoice. They cover English,
  German, Dutch and French; dot and comma decimals; 1- and 2-page invoices; tables spanning pages; and vendor details
  only in an image.
- **1 scanned invoice**: [Parseur's public OCR sample](https://parseur.com/samples/sample-invoice.pdf). It is 2 pages
  and image-only, with 18 line items, a table continued over a page break with a "carried forward" subtotal, stamps
  and handwriting. Its company data is fictional (Parseur made it as a test document), but it is a real public file;
  I did not create it.

`sample_invoices/ground_truth.json` holds the expected items, transcribed by hand from each PDF.

**Measured result, rule-based fallback (no API key available during development):**

```
Line items found: 49/53 | Quantity correct: 47/53 | Value correct: 49/53 | Vendor correct: 8/11
Auto-validated records: 21 | flagged for review: 30
Silent errors (auto-validated but wrong): 0
```

Every wrong or uncertain record was flagged, and none slipped into the export unreviewed. The 18 scanned line items
were all read correctly. They are still flagged, because the OCR'd header was unreadable and the fallback could only
infer the quantity column.

**Measured result, Gemini (`gemini-3.5-flash`, live API, 1 November 2026).** The free-tier key allows only 20
requests per model per day, so Gemini read 7 of the 11 invoices. The other 4 got temporary Google errors (503
"high demand", 429 quota) and fell back to the rule-based parser, as designed. On the 7 invoices Gemini read:

```
Invoices: parseur (scanned), QualityHosting, coolblue1, coolblue2, Netpresse, AzureInterior, Flipkart
Line items found: 45/45 | Quantity correct: 45/45 | Value correct: 45/45 | Vendor correct: 7/7
Extra rows: 2 (coolblue2 bundle components without their own price -> flagged, value missing)
Auto-validated records: 38 | Silent errors: 0
```

A separate single call on `free_fiber.pdf` was also correct: vendor "Free SAS", and the charge was extracted once
even though it appears on both pages. **Not yet measured with Gemini:** Saeco, AWS and OYO. Run
`python scripts/evaluate.py` again once the daily quota resets.

Notes on Gemini:
- On free-tier keys, `gemini-3.8-flash` was overloaded (503) and hung without answering, while `gemini-3.5-flash`
  answered in about 6 s. That is why the default is `gemini-3.5-flash`.
- Quota handling: short per-minute limits and 503 "busy" responses are waited out using Google's suggested
  `retryDelay`. A **daily** quota fails at once with a clear message, so the batch continues with the fallback parser
  and nobody waits for hours.
- The AI path also has offline tests: a mocked client covers request format, schema enforcement, page chunking,
  JSON-mode fallback, retries and error messages. Other tests feed LLM-style output for a real invoice through the full
  pipeline and show that a wrong value, a swapped association, a duplicate or an invented vendor all get flagged.

---

## 8. Limitations and trade-offs

- **AI accuracy so far is measured on 8 invoices with Gemini** (see above), which is a small sample. Free-tier Gemini keys allow only 20 requests per model per day, too few for real batch use; enable billing for production.
- The **rule-based fallback** needs a recognisable table header, or rows with ≥2 numbers. It cannot handle the AWS
  layout (no header, one amount per row), so that invoice fails without AI. It also guesses vendors only from legal
  suffixes (Ltd, GmbH, B.V., …).
- **Units and currencies are shown, never converted.** `10 g` and `10 kg` both hold the number 10, so summing a quantity
  column with mixed units, or a value column with mixed currencies, gives a meaningless total. Reviewers can change a
  unit or currency by typing it in the same field (`12.5 kg`, `EUR 44.99`).
- **Reconciliation** can be fooled if line values coincidentally add up to some other printed amount, or a vendor
  prints no total. In the second case it adds a warning and does not flag the records.
- Tesseract struggles with very poor scans, handwriting and stamps over text. Low confidence is flagged, not fixed.
- Session state lives in memory: a browser refresh or server restart loses unexported review work.
- Page text is sent to the AI provider (OpenAI or Google). Check data-processing terms for confidential invoices.

## 9. Possible production improvements

- Store invoices, records, corrections and an audit trail (who verified what, when) in a database. Add login and roles.
- Feed reviewer corrections back as few-shot examples, or as an evaluation set to track accuracy over time.
- Connect to the ERP: match against purchase orders and goods receipts (three-way match) and vendor master data.
- Send low-quality scans to a vision-capable LLM or a cloud document-AI OCR; add PaddleOCR for difficult scans.
- Process large batches asynchronously in a queue (Celery / RQ) with progress callbacks.
- Show per-vendor accuracy dashboards; use confidence thresholds tuned on labelled data.
- Compare the LLM's and the rule-based parser's output, and treat agreement as an extra trust signal.
