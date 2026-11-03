"""LLM-based structured extraction (OpenAI or Gemini Structured Outputs + Pydantic).

The LLM's job is narrow: *read* the layout text of an invoice and *map* whatever
the vendor calls things ("Particulars", "Omschrijving", "Zeilenbetrag", ...) to
our fixed schema. It returns numbers as the verbatim printed strings; parsing,
normalisation and all validation happen afterwards in deterministic Python.

Long invoices are split into page chunks. Every chunk gets the first page's
header as read-only context (so the vendor is known), and items are only taken
from the chunk's own pages, so line items are never mixed between pages.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field, replace

from pydantic import ValidationError

from models.invoice_schema import DocumentContent, LLMInvoiceExtraction, PageContent
from utils.helpers import get_logger

log = get_logger(__name__)

MAX_CHARS_PER_CALL = int(os.getenv("LLM_MAX_CHARS_PER_CALL", "24000"))
# Extra, slower retries for per-minute quota (429) and "model overloaded" (503)
# responses, on top of the SDK's own quick retries. Free tiers hit these often.
BUSY_RETRIES = int(os.getenv("LLM_BUSY_RETRIES", "2"))
BUSY_WAIT_SECONDS = float(os.getenv("LLM_BUSY_WAIT_SECONDS", "20"))
MAX_BUSY_WAIT_SECONDS = 90.0  # longer suggested waits mean a daily quota: give up instead
HEADER_CONTEXT_CHARS = 1500

SYSTEM_PROMPT = """You are a meticulous accounts-payable assistant in the Finance department of a manufacturing company.
You read supplier invoices of ANY layout and language and extract line items into a fixed schema.

Input: the invoice text, page by page, reconstructed from the PDF layout. Each visual row of the page is one text line;
runs of 2+ spaces separate table columns. Some pages come from OCR and may contain recognition errors. Optional
"TABLE HINT" blocks are machine-detected tables and may be incomplete.

Extract:
- vendor_name: the company that ISSUED the invoice (seller/supplier/service provider). Look at letterheads, logos
  (text recognised inside images), "Sold by", "From", footers with registration/bank/VAT details. NEVER return the
  customer / bill-to / ship-to party. If you cannot tell which party is the issuer, return null.
- items: EVERY individual billable line item (material, product, goods or service) in the order they appear,
  across all pages. Column names differ per vendor and language - map them by meaning, e.g.
  material_name <- Description / Item / Particulars / Product details / Goods / Artikel / Omschrijving / Beschreibung / Désignation;
  quantity <- Qty / Quantity / Units / Menge / Aantal / Hoeveelheid / Nombre / Quantité;
  value <- Amount / Line total / Net amount / Extended price / Item total / Bedrag / Zeilenbetrag / Montant.
  * value_text is the line's total amount as printed, NOT the unit price. If the line has both a net and a gross
    amount, use the column that the invoice's line items total refers to (normally the right-most line amount).
  * Copy numbers EXACTLY as printed (keep separators, signs and currency symbols); do not calculate anything.
  * If a quantity or amount is not printed for a line, use null - never compute or guess it.
  * Put the unit of measure (pcs, kg, L, m, roll, hrs, ...) in "unit" when printed, otherwise null.
  * Keep material_name as printed (you may join a description that wraps onto the next line); include an article
    code only if it is part of the description column.
- Do NOT output as items: subtotals, totals, "carried forward"/"brought forward" amounts, taxes/VAT lines, payment
  records, balances, or informational sub-lines that only break down or annotate a parent item already priced
  (e.g. "incl. levy", serial numbers, bundle components without their own amount). Shipping, freight, discounts,
  surcharges and fees that are separate billed lines ARE items.
- If the same charge appears both in a summary section and in a detail section, extract it only once (from the
  most detailed section) and mention this in ambiguity_note.
- source_page: page number where the item row appears. source_line: copy the row text verbatim.
- items_total_text: the printed total that the extracted item values add up to (e.g. net subtotal, or the grand
  total when line values include tax), copied exactly; null if none is printed.
- ambiguity_note: explain briefly whenever a value is unclear, unreadable, or you were unsure about a mapping.

Never invent vendors, materials, quantities or values that are not in the text."""


class LLMExtractionError(Exception):
    """User-presentable LLM failure (no secrets, no stack traces)."""


class LLMKeyError(LLMExtractionError):
    """The API key itself is unusable (rejected / not allowed): rotate to the next key."""


GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass
class LLMConfig:
    api_key: str | None = field(repr=False)
    model: str
    timeout: float
    max_retries: int
    base_url: str | None = None
    provider: str = "openai"  # "openai" | "gemini"
    # All configured keys as (env variable name, key), in rotation order.
    keys: list[tuple[str, str]] = field(default_factory=list, repr=False)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Pick the provider from LLM_PROVIDER, or (auto) from whichever API key is set.

        Gemini is called through Google's OpenAI-compatible endpoint, so the same
        client, prompt, schema and Structured Outputs call serve both providers.
        """
        provider = os.getenv("LLM_PROVIDER", "auto").lower()
        if provider == "auto":
            provider = "gemini" if _env_keys("GEMINI_API_KEY") and not _env_keys("OPENAI_API_KEY") else "openai"
        timeout = float(os.getenv("LLM_TIMEOUT", os.getenv("OPENAI_TIMEOUT", "90")))
        retries = int(os.getenv("LLM_MAX_RETRIES", os.getenv("OPENAI_MAX_RETRIES", "3")))
        if provider == "gemini":
            keys = _env_keys("GEMINI_API_KEY")
            return cls(
                api_key=keys[0][1] if keys else None,
                keys=keys,
                model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
                timeout=timeout,
                max_retries=retries,
                base_url=os.getenv("GEMINI_BASE_URL") or GEMINI_OPENAI_BASE_URL,
                provider="gemini",
            )
        keys = _env_keys("OPENAI_API_KEY")
        return cls(
            api_key=keys[0][1] if keys else None,
            keys=keys,
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            timeout=timeout,
            max_retries=retries,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.key_pool)

    @property
    def key_pool(self) -> list[tuple[str, str]]:
        return self.keys or ([(self.key_var, self.api_key)] if self.api_key else [])

    @property
    def label(self) -> str:
        return "Gemini" if self.provider == "gemini" else "OpenAI"

    @property
    def key_var(self) -> str:
        return "GEMINI_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY"


def _env_keys(var: str) -> list[tuple[str, str]]:
    """Keys from VAR, VAR_2 ... VAR_9 and the comma-separated VARS, de-duplicated, in that order."""
    found = [(n, os.getenv(n, "")) for n in [var] + [f"{var}_{i}" for i in range(2, 10)]]
    found += [(f"{var}S #{i}", k) for i, k in enumerate(os.getenv(f"{var}S", "").split(","), 1)]
    keys, seen = [], set()
    for name, key in found:
        key = key.strip()
        if key and key not in seen:
            seen.add(key)
            keys.append((name, key))
    return keys


# Keys that failed, by fingerprint -> (blocked until epoch seconds, reason). Module level,
# so later invoices of the batch skip a key whose quota is spent instead of re-trying it.
_KEY_BLOCKS: dict[str, tuple[float, str]] = {}


def reset_key_state() -> None:
    _KEY_BLOCKS.clear()


class _KeyPool:
    """Rotates through the configured API keys, skipping keys that recently failed."""

    def __init__(self, cfg: LLMConfig, client=None, client_factory=None):
        self.cfg = cfg
        self.keys = cfg.key_pool or [(cfg.key_var, "")]
        self._fixed = client
        self._factory = client_factory or (lambda key: _client(replace(cfg, api_key=key)))
        self._clients: dict[str, object] = {}

    @staticmethod
    def _id(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def usable(self) -> list[tuple[str, str]]:
        now = time.time()
        return [(n, k) for n, k in self.keys if _KEY_BLOCKS.get(self._id(k), (0.0, ""))[0] <= now]

    def client(self, key: str):
        if self._fixed is not None:
            return self._fixed
        if key not in self._clients:
            self._clients[key] = self._factory(key)
        return self._clients[key]

    def block(self, name: str, key: str, seconds: float, reason: str) -> None:
        _KEY_BLOCKS[self._id(key)] = (time.time() + seconds, reason)
        left = len(self.usable())
        if len(self.keys) > 1:
            log.warning("%s: %s; %s", name, reason,
                        f"switching to the next key ({left} left)" if left else "no other key available")

    def wait_for_next_key(self) -> bool:
        """Sleep until the soonest blocked key is free again, if that is soon. False = give up."""
        until = min(_KEY_BLOCKS.get(self._id(k), (0.0, ""))[0] for _, k in self.keys)
        wait = max(until - time.time(), 0.0)
        if wait > MAX_BUSY_WAIT_SECONDS:
            return False
        log.warning("%s: every API key is rate limited; waiting %.0fs", self.cfg.label, wait)
        time.sleep(wait)
        for _, k in self.keys:
            if _KEY_BLOCKS.get(self._id(k), (0.0, ""))[0] <= until:
                _KEY_BLOCKS.pop(self._id(k), None)
        return True

    def failure_message(self) -> str:
        reasons = [(n, _KEY_BLOCKS.get(self._id(k), (0.0, "it failed"))[1]) for n, k in self.keys]
        if len(reasons) == 1:
            text = reasons[0][1]
            return text[0].upper() + text[1:] + "."
        detail = "; ".join(f"{n}: {r}" for n, r in reasons)
        return f"All {len(reasons)} {self.cfg.label} API keys failed – {detail}."


def _page_block(p: PageContent) -> str:
    head = f"===== PAGE {p.page_number} ({'OCR' if p.method == 'ocr' else 'digital text'}) ====="
    parts = [head, p.text.strip() or "[no readable text on this page]"]
    for i, t in enumerate(p.tables_markdown, 1):
        parts.append(f"--- TABLE HINT {i} (page {p.page_number}) ---\n{t}")
    return "\n".join(parts)


def chunk_pages(doc: DocumentContent, max_chars: int | None = None) -> list[list[PageContent]]:
    """Group consecutive pages into chunks under ``max_chars`` (a page is never split)."""
    max_chars = max_chars or MAX_CHARS_PER_CALL
    chunks: list[list[PageContent]] = []
    current: list[PageContent] = []
    size = 0
    for p in doc.pages:
        n = len(_page_block(p))
        if current and size + n > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(p)
        size += n
    if current:
        chunks.append(current)
    return chunks


def build_user_prompt(doc: DocumentContent, pages: list[PageContent], total_chunks: int, chunk_no: int) -> str:
    intro = f"Invoice file: {doc.filename} ({doc.page_count} page(s))."
    if total_chunks > 1:
        first = doc.pages[0].text[:HEADER_CONTEXT_CHARS]
        page_nums = ", ".join(str(p.page_number) for p in pages)
        intro += (
            f"\nThis invoice is long and is processed in parts. This is part {chunk_no} of {total_chunks}, "
            f"containing pages {page_nums}. Extract items ONLY from these pages."
            f"\n----- HEADER OF PAGE 1 (context for the vendor only; do not extract items from it"
            f"{' unless page 1 is part of this chunk' if chunk_no == 1 else ''}) -----\n{first}\n-----"
        )
    body = "\n\n".join(_page_block(p) for p in pages)
    return f"{intro}\n\n{body}"


def _client(cfg: LLMConfig):
    from openai import OpenAI

    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout, max_retries=cfg.max_retries)


def _messages(user_prompt: str, schema_in_prompt: bool = False) -> list[dict]:
    system = SYSTEM_PROMPT
    if schema_in_prompt:
        system += ("\n\nRespond with ONE JSON object only, matching this JSON schema exactly:\n"
                   + json.dumps(LLMInvoiceExtraction.model_json_schema()))
    return [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]


def _call_json_mode(client, cfg: LLMConfig, user_prompt: str) -> LLMInvoiceExtraction:
    """Fallback for OpenAI-compatible endpoints that reject strict JSON-schema output:
    plain JSON mode with the schema in the prompt, validated by Pydantic."""
    completion = client.chat.completions.create(
        model=cfg.model,
        temperature=0,
        messages=_messages(user_prompt, schema_in_prompt=True),
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content or ""
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return LLMInvoiceExtraction.model_validate_json(content)


def _quota_details(exc) -> tuple[float | None, str]:
    """(suggested retry delay in seconds, quota id) from a provider error body, if present."""
    body = getattr(exc, "body", None)
    delay, quota_id = None, ""
    for item in body if isinstance(body, list) else [body]:
        err = item.get("error", item) if isinstance(item, dict) else {}
        for detail in err.get("details", []) if isinstance(err, dict) else []:
            for v in detail.get("violations", []) or []:
                quota_id = quota_id or str(v.get("quotaId", ""))
            if "retryDelay" in detail:
                try:
                    delay = float(str(detail["retryDelay"]).rstrip("s"))
                except ValueError:
                    pass
    return delay, quota_id


def _is_invalid_key(exc) -> bool:
    """Gemini reports a bad key as HTTP 400 ("Please pass a valid API key"), not 401."""
    text = f"{getattr(exc, 'message', '')} {getattr(exc, 'body', '')}".lower()
    return any(t in text for t in ("valid api key", "api key not valid", "api_key_invalid", "api key expired"))


def _retry_delay(exc) -> float:
    """Seconds to wait before retrying: the provider's suggested delay if given."""
    delay, _ = _quota_details(exc)
    return delay + 1 if delay is not None else BUSY_WAIT_SECONDS


def _call(pool: _KeyPool, cfg: LLMConfig, user_prompt: str) -> LLMInvoiceExtraction:
    """One extraction call. A key that is rejected or out of quota is set aside and the
    next configured key is used; short per-minute limits and overloads are waited for."""
    import openai

    waits = 0
    while True:
        overloaded = None
        for name, key in pool.usable():
            try:
                return _call_once(pool.client(key), cfg, user_prompt)
            except LLMKeyError as exc:
                pool.block(name, key, math.inf, f"{exc} (check {name} in your .env file)")
            except openai.RateLimitError as exc:
                wait = _retry_delay(exc)
                _, quota_id = _quota_details(exc)
                if "PerDay" in quota_id or wait > MAX_BUSY_WAIT_SECONDS:
                    # A daily quota will not reset within a reasonable wait: do not wait for this key.
                    hours = f" It resets in about {wait / 3600:.0f} hours" if wait > MAX_BUSY_WAIT_SECONDS else ""
                    pool.block(name, key, max(wait, 3600.0),
                               f"the daily {cfg.label} quota for model '{cfg.model}' is used up "
                               f"(free tiers allow only a few requests per day).{hours}")
                else:
                    pool.block(name, key, wait, f"the {cfg.label} API rate limit or quota was exceeded")
            except openai.InternalServerError as exc:
                overloaded = exc  # model-wide, not the key's fault: try the next key, then wait
                if len(pool.keys) > 1:
                    log.warning("%s: %s is overloaded (%s); trying the next key",
                                name, cfg.model, getattr(exc, "status_code", ""))

        if overloaded is not None:
            if waits >= BUSY_RETRIES:
                raise LLMExtractionError(
                    f"The {cfg.label} model '{cfg.model}' is overloaded right now (server error "
                    f"{getattr(overloaded, 'status_code', '')}). Please retry later or choose another model.")
            log.warning("%s busy; waiting %.0fs before retry %d/%d",
                        cfg.label, BUSY_WAIT_SECONDS, waits + 1, BUSY_RETRIES)
            time.sleep(BUSY_WAIT_SECONDS)
        elif waits >= BUSY_RETRIES or not pool.wait_for_next_key():
            raise LLMExtractionError(pool.failure_message() + _advice(pool))
        waits += 1


def _advice(pool: _KeyPool) -> str:
    reasons = " ".join(_KEY_BLOCKS.get(pool._id(k), (0.0, ""))[1] for _, k in pool.keys)
    if "daily" in reasons:
        return " Add another API key, enable billing, or switch to another model in .env."
    if "rate limit" in reasons:
        return " Please retry in a minute."
    return ""


def _call_once(client, cfg: LLMConfig, user_prompt: str) -> LLMInvoiceExtraction:
    import openai

    name = cfg.label
    try:
        try:
            completion = client.chat.completions.parse(
                model=cfg.model,
                temperature=0,
                messages=_messages(user_prompt),
                response_format=LLMInvoiceExtraction,  # JSON-schema enforced Structured Outputs
            )
        except openai.BadRequestError as exc:
            if _is_invalid_key(exc) or (cfg.provider == "openai" and not cfg.base_url):
                raise
            log.warning("%s rejected strict schema output (%s); retrying in JSON mode",
                        name, getattr(exc, "message", exc))
            return _call_json_mode(client, cfg, user_prompt)
    except openai.AuthenticationError:
        raise LLMKeyError(f"the {name} API key was rejected")
    except openai.PermissionDeniedError:
        raise LLMKeyError(f"the {name} API key is not allowed to use model '{cfg.model}'")
    except (openai.RateLimitError, openai.InternalServerError):
        raise  # handled by _call: next key, or wait and retry
    except openai.APITimeoutError:
        raise LLMExtractionError(
            f"The {name} API timed out while reading this invoice (model '{cfg.model}' may be overloaded; "
            "try again or choose another model).")
    except openai.APIConnectionError:
        raise LLMExtractionError(f"Could not connect to the {name} API. Check your internet connection.")
    except openai.LengthFinishReasonError:
        raise LLMExtractionError("The invoice is too long for a single AI response.")
    except openai.NotFoundError:
        raise LLMExtractionError(f"The {name} model '{cfg.model}' was not found (check the model name in .env).")
    except openai.BadRequestError as exc:
        if _is_invalid_key(exc):
            raise LLMKeyError(f"the {name} API key was rejected")
        log.error("%s bad request: %s", name, getattr(exc, "message", exc))
        raise LLMExtractionError(f"The AI model '{cfg.model}' rejected the request (check the model name in .env).")
    except openai.APIError as exc:
        log.error("%s API error: %s", name, type(exc).__name__)
        raise LLMExtractionError(f"The {name} API returned an error.")
    except ValidationError:
        raise LLMExtractionError("The AI response did not match the expected invoice schema.")

    msg = completion.choices[0].message
    if getattr(msg, "refusal", None):
        raise LLMExtractionError("The AI model refused to process this document.")
    if msg.parsed is None:
        raise LLMExtractionError("The AI returned an empty or invalid response.")
    # Re-validate explicitly: we never trust the SDK/LLM output blindly.
    return LLMInvoiceExtraction.model_validate(msg.parsed.model_dump())


def extract_with_llm(
    doc: DocumentContent, cfg: LLMConfig | None = None, client=None, client_factory=None
) -> LLMInvoiceExtraction:
    """Run structured extraction over the document, chunking long invoices by page.

    ``client`` is used for every key (tests); ``client_factory(key)`` builds one per key.
    """
    cfg = cfg or LLMConfig.from_env()
    if not cfg.enabled and client is None and client_factory is None:
        raise LLMExtractionError(f"No {cfg.label} API key configured ({cfg.key_var}).")
    pool = _KeyPool(cfg, client=client, client_factory=client_factory)
    chunks = chunk_pages(doc)
    merged: LLMInvoiceExtraction | None = None
    for i, pages in enumerate(chunks, 1):
        prompt = build_user_prompt(doc, pages, len(chunks), i)
        log.info("%s: LLM call %d/%d (%d chars, model %s)", doc.filename, i, len(chunks), len(prompt), cfg.model)
        part = _call(pool, cfg, prompt)
        allowed = {p.page_number for p in pages}
        for it in part.items:
            # Guard against items attributed to pages outside this chunk.
            if it.source_page not in allowed:
                it.source_page = pages[0].page_number if len(pages) == 1 else it.source_page
        if merged is None:
            merged = part
        else:
            merged.items.extend(part.items)
            merged.vendor_name = merged.vendor_name or part.vendor_name
            merged.vendor_evidence = merged.vendor_evidence or part.vendor_evidence
            merged.currency = merged.currency or part.currency
            # The total that closes the item table is normally on the last page.
            merged.items_total_text = part.items_total_text or merged.items_total_text
    assert merged is not None
    return merged
