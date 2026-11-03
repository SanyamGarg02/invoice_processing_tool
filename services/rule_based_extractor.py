"""Rule-based, template-free fallback extractor.

Used when no OpenAI API key is configured or the LLM call fails. It is
deliberately generic – no vendor names or templates – and relies on:

* a multilingual vocabulary of column-header synonyms to find the line-item
  table header and the horizontal position of each column,
* x-coordinates of each row's text segments to assign numbers to columns,
* legal-entity suffixes and "sold by / supplier" cues to guess the vendor.

It is less capable than the LLM path (e.g. tables without a recognisable header
or single-amount summary lines), so its output goes through exactly the same
deterministic validation and anything uncertain is flagged for review.
"""

from __future__ import annotations

import re

from models.invoice_schema import DocumentContent, LLMInvoiceExtraction, RawLineItem
from utils.helpers import parse_number

# Column header vocabulary (lower-case, accent-free prefixes are fine).
HEADER_SYNONYMS: dict[str, list[str]] = {
    "desc": ["description", "item", "items", "product", "products", "particulars", "goods", "material", "article",
             "artikel", "omschrijving", "beschreibung", "bezeichnung", "designation", "désignation", "libelle",
             "libellé", "service", "services", "title", "details", "descripcion", "descrizione", "produkt"],
    "qty": ["qty", "quantity", "qte", "qté", "quantite", "quantité", "units", "menge", "anzahl", "aantal",
            "hoeveelheid", "nombre", "cantidad", "quantita"],
    "unit_price": ["price", "rate", "unit price", "unit cost", "prijs", "preis", "vk-preis", "einzelpreis",
                   "p.u.", "prix", "precio", "prezzo", "unit"],
    "amount": ["amount", "total", "line total", "net amount", "extended", "value", "bedrag", "betrag",
               "zeilenbetrag", "gesamt", "montant", "importe", "importo", "subtotal", "amt", "line amount",
               "prijs incl. btw"],
    "tax": ["tax", "vat", "btw", "mwst", "tva", "gst", "iva", "tax(%)", "taxes"],
    "discount": ["disc", "disc.%", "discount", "rabatt", "korting", "remise"],
    # Position / line-number columns – recognised so their numbers are not taken as quantities.
    "pos": ["pos", "pos.", "#", "no", "no.", "sr", "sr.", "s.no", "sl", "line", "nr", "nr.", "code", "sku", "hsn"],
}
# Rows that sit inside an item table but are not items (section subtotals, tax
# lines, page carry-overs, "incl." sub-lines): skipped, the table continues.
SKIP_ROW = re.compile(
    r"(?i)(\bsub-?total|subtotaal|zwischensumme|sous-total|carried (forward|over)|brought forward|übertrag|"
    r"^\W*(incl\.?|including|inkl\.?|inclusief)\s|^\W*(tax|vat|btw|mwst|tva|gst|iva)\b)"
)
# Rows that close the item table.
END_ROW = re.compile(
    r"(?i)\b(total|totaal|totale|summe|gesamt|grand total|amount due|net total|total ht|total ttc|balance|"
    r"montant total|endbetrag|to pay)\b"
)
LEGAL_SUFFIX = re.compile(
    r"(?i)\b(inc\.?|incorporated|ltd\.?|limited|llc|l\.l\.c\.|plc|gmbh|ag|kg|ab|oy|a/s|as|b\.v\.?|bv|n\.v\.?|"
    r"s\.a\.?|sa|sas|sarl|s\.r\.l\.?|srl|s\.p\.a\.?|spa|s\.l\.|pvt\.?|private limited|corp\.?|corporation|co\.)(?=$|[\s,.;:)])"
)
VENDOR_CUES = re.compile(r"(?i)(sold by|seller|supplier|vendor|service provider|issued by|from:|remit to|lieferant|leverancier|fournisseur)")
CUSTOMER_CUES = re.compile(r"(?i)(bill(ed)? to|ship(ped)? to|customer|buyer|attn|billing address|shipping address|"
                           r"factuuradres|afleveradres|rechnungsadresse|lieferadresse|kunde|client|deliver to|t\.a\.v\.)")
NUMERIC_SEG = re.compile(r"^[\s(+\-−]*(?:[$€£¥₹]|rs\.?|eur|usd|inr)?\s*[-−]?\d[\d.,'  ]*\s*(?:[$€£¥₹%]|[a-z]{1,6}\.?)?\)?$", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip(" :|"))


def _classify_header(seg: str) -> str | None:
    s = _norm(seg)
    best = None
    for col, words in HEADER_SYNONYMS.items():
        for w in words:
            if s == w or s.startswith(w + " ") or s.endswith(" " + w) or (len(w) > 4 and w in s):
                if best is None or len(w) > best[1]:
                    best = (col, len(w))
    return best[0] if best else None


def _is_numeric(seg: str) -> bool:
    s = seg.replace("|", " ").strip()
    return bool(NUMERIC_SEG.match(s)) and parse_number(s.replace("%", "")) is not None


def _find_header(rows) -> tuple[int, list[tuple[float, float, str]]] | None:
    """Return (row index, [(x0, x1, column type)]) of the most header-like row."""
    best = None
    for idx, row in enumerate(rows):
        cols = [(a, b, _classify_header(t)) for a, b, t in row]
        kinds = {c for _, _, c in cols if c}
        labelled = sum(1 for c in cols if c[2])
        line = " ".join(t for _, _, t in row)
        # A header row consists mostly of column labels and contains few digits.
        mostly_labels = labelled >= 0.6 * len(cols) and len(re.findall(r"\d", line)) <= 4
        if mostly_labels and "desc" in kinds and len(kinds & {"qty", "amount", "unit_price"}) >= 1:
            score = len(kinds)
            if best is None or score > best[0]:
                best = (score, idx, [c for c in cols if c[2]])
    return (best[1], best[2]) if best else None


def _assign(seg: tuple[float, float, str], header: list[tuple[float, float, str]]) -> str | None:
    """Assign a row segment to the header column it overlaps most (or is nearest to)."""
    a, b, _ = seg
    best, best_score = None, None
    for ha, hb, kind in header:
        overlap = min(b, hb) - max(a, ha)
        dist = abs((a + b) / 2 - (ha + hb) / 2)
        score = (overlap if overlap > 0 else 0) * 1000 - dist
        if best_score is None or score > best_score:
            best, best_score = kind, score
    return best


def _split_qty_unit(text: str) -> tuple[str, str | None]:
    t = text.replace("|", " ").strip()
    m = re.match(r"^([-−]?\d[\d.,' ]*)\s*([A-Za-z][A-Za-z.]{0,7})?$", t)
    if m:
        return m.group(1).strip(), (m.group(2) or None)
    return t, None


def _parse_rows(doc: DocumentContent, decimal_sep: str) -> list[RawLineItem]:
    items: list[RawLineItem] = []
    header_cols = None  # carried across pages for multi-page tables
    for page in doc.pages:
        rows = page.rows
        found = _find_header(rows)
        start = 0
        if found:
            start, header_cols = found[0] + 1, found[1]
        in_table = header_cols is not None
        for row in rows[start:]:
            line = "  ".join(t for _, _, t in row)
            if SKIP_ROW.search(line):
                continue
            if END_ROW.search(line):
                in_table = False
                continue
            nums = [s for s in row if _is_numeric(s[2])]
            texts = [s for s in row if not _is_numeric(s[2]) and len(re.findall(r"[^\W\d_]", s[2])) >= 3]
            if not nums or not texts:
                continue
            qty_t = val_t = unit = note = None
            if in_table and header_cols:
                by_col: dict[str, list[str]] = {}
                for s in row:
                    by_col.setdefault(_assign(s, header_cols) or "?", []).append(s[2])
                amount = [t for t in by_col.get("amount", []) if _is_numeric(t)]
                qty = [t for t in by_col.get("qty", []) if "%" not in t and _is_numeric(_split_qty_unit(t)[0])]
                desc = [t for t in by_col.get("desc", []) if not _is_numeric(t)]
                if amount:
                    val_t = amount[-1]
                if qty:
                    qty_t, unit = _split_qty_unit(qty[0])
                name = " ".join(desc) if desc else max((s[2] for s in texts), key=len)
            else:
                # No header: need description + quantity-like + amount-like numbers.
                money = [s for s in nums if "%" not in s[2]]
                if len(money) < 2:
                    continue
                name = max((s[2] for s in texts), key=len)
                after = [s for s in money if s[0] > max(t[0] for t in texts if t[2] == name)]
                if len(after) < 2:
                    continue
                qty_t, unit = _split_qty_unit(after[0][2])
                val_t = after[-1][2]
                note = "No table header recognised; quantity column inferred from position."
            if val_t is None:
                continue
            if unit is None:
                # A short word right after the quantity (e.g. "|pcs" split off by OCR).
                tail = [t for t in row if not _is_numeric(t[2]) and re.fullmatch(r"\|?\s*[A-Za-z]{1,6}\.?", t[2].strip())]
                unit = tail[0][2].strip("| ") if tail and qty_t else None
            items.append(RawLineItem(
                material_name=name.strip(" |"),
                quantity_text=qty_t,
                unit=unit,
                value_text=val_t.replace("|", "").strip(),
                source_page=page.page_number,
                source_line=re.sub(r"\s{2,}", "  ", line),
                ambiguity_note=note,
            ))
    return items


def _guess_vendor(doc: DocumentContent) -> tuple[str | None, str | None]:
    lines = [ln.strip() for p in doc.pages for ln in p.text.splitlines() if ln.strip()]
    candidates: dict[str, float] = {}
    names: dict[str, tuple[str, str]] = {}
    for i, ln in enumerate(lines):
        for part in re.split(r"\s{2,}| - | – |\|", ln):
            part = part.strip(" ,:;-")
            m = LEGAL_SUFFIX.search(part)
            if not m or len(part) > 60 or len(part) < 4:
                continue
            end = m.end()
            while (m2 := LEGAL_SUFFIX.match(part, end + (len(part[end:]) - len(part[end:].lstrip(" ,"))))) :
                end = m2.end()  # chains such as "Pvt. Ltd." or "GmbH & Co. KG"
            name = part[:end].strip(" ,:;")
            name = re.sub(r"(?i)^.*?\b(sold by|supplier|vendor|seller|service provider|issued by|from)\b\s*:?\s*", "", name)
            # Need a real name besides the legal suffix (rejects e.g. "Sarl au capital").
            if len(re.findall(r"[^\W\d_]", LEGAL_SUFFIX.sub("", name))) < 3:
                continue
            score = 1.0
            ctx = " ".join(lines[max(0, i - 2): i + 1])
            if VENDOR_CUES.search(ln) or VENDOR_CUES.search(lines[i - 1] if i else ""):
                score += 3
            if CUSTOMER_CUES.search(ctx):
                score -= 3
            key = name.lower()
            candidates[key] = candidates.get(key, 0) + score  # repeats (letterhead + footer) add up
            names.setdefault(key, (name, ln))
    if not candidates:
        return None, None
    key = max(candidates, key=candidates.get)
    if candidates[key] <= 0:
        return None, None
    return names[key]


def extract_rule_based(doc: DocumentContent, decimal_sep: str = ".") -> LLMInvoiceExtraction:
    vendor, ev = _guess_vendor(doc)
    items = _parse_rows(doc, decimal_sep)
    return LLMInvoiceExtraction(vendor_name=vendor, vendor_evidence=ev, currency=None, items_total_text=None, items=items)
