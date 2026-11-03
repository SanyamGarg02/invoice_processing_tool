"""Evaluate the pipeline against hand-verified ground truth for the sample invoices.

Usage:
    python scripts/evaluate.py                    # uses LLM if OPENAI_API_KEY is set
    python scripts/evaluate.py --rule-based       # force the rule-based fallback
    python scripts/evaluate.py --dir my_invoices  # any folder with a ground_truth.json

The most important number for Finance is "silent errors": records that were
auto-validated (would be exported without review) but are wrong. Wrong records
that were flagged for review are acceptable – a human catches them.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from models.invoice_schema import RecordStatus  # noqa: E402
from utils.helpers import similarity  # noqa: E402


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) < 0.005


def match_items(expected: list[dict], records) -> list[tuple[dict, object | None]]:
    """Greedy one-to-one matching on value + name similarity (order-insensitive)."""
    pairs, used = [], set()
    for exp in expected:
        values = [exp["value"]] + exp.get("value_alternatives", [])
        best, best_score = None, 0.0
        for i, r in enumerate(records):
            if i in used:
                continue
            score = similarity(exp["material"], r.material_name or "")
            if any(_close(v, r.material_value) for v in values):
                score += 1.0
            if score > best_score:
                best, best_score = i, score
        if best is not None and best_score >= 0.6:
            used.add(best)
            pairs.append((exp, records[best]))
        else:
            pairs.append((exp, None))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "sample_invoices"))
    ap.add_argument("--rule-based", action="store_true")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    if args.rule_based:
        os.environ["EXTRACTION_MODE"] = "rule-based"
    logging.disable(logging.INFO)

    from services.pipeline import process_invoice

    folder = Path(args.dir)
    truth = json.loads((folder / "ground_truth.json").read_text())["invoices"]
    totals = dict(expected=0, found=0, qty_ok=0, value_ok=0, extra=0, vendor_ok=0, invoices=0,
                  auto_validated=0, silent_errors=0, flagged=0)
    for fname, gt in truth.items():
        path = folder / fname
        if not path.exists():
            print(f"skip {fname} (missing)")
            continue
        inv = process_invoice(fname, path.read_bytes())
        records = [r for r in inv.records if r.status != RecordStatus.REJECTED]
        pairs = match_items(gt["items"], records)
        matched_ids = {id(r) for _, r in pairs if r is not None}
        vendor_ok = bool(inv.vendor_name) and any(
            similarity(a, inv.vendor_name) > 0.75 or a.lower() in inv.vendor_name.lower() for a in gt["vendor_aliases"])
        found = sum(1 for _, r in pairs if r is not None)
        qty_ok = sum(1 for e, r in pairs if r is not None and _close(e["quantity"], r.material_quantity))
        val_ok = sum(1 for e, r in pairs if r is not None and any(
            _close(v, r.material_value) for v in [e["value"]] + e.get("value_alternatives", [])))
        extra = len(records) - len(matched_ids)

        # A record is "correct" when matched and both numbers are right and vendor is right.
        correct = {id(r) for e, r in pairs if r is not None and vendor_ok and _close(e["quantity"], r.material_quantity)
                   and any(_close(v, r.material_value) for v in [e["value"]] + e.get("value_alternatives", []))}
        auto = [r for r in records if r.status == RecordStatus.VALIDATED]
        silent = [r for r in auto if id(r) not in correct]

        totals["invoices"] += 1
        totals["expected"] += len(gt["items"])
        totals["found"] += found
        totals["qty_ok"] += qty_ok
        totals["value_ok"] += val_ok
        totals["extra"] += extra
        totals["vendor_ok"] += vendor_ok
        totals["auto_validated"] += len(auto)
        totals["silent_errors"] += len(silent)
        totals["flagged"] += sum(1 for r in records if r.status == RecordStatus.NEEDS_REVIEW)
        print(f"{fname:28s} {inv.status.value:12s} engine={inv.extraction_engine or '-':22s} "
              f"vendor={'OK ' if vendor_ok else 'BAD'} items {found}/{len(gt['items'])} extra={extra} "
              f"qty_ok={qty_ok} value_ok={val_ok} auto_validated={len(auto)} silent_errors={len(silent)}")
        for r in silent:
            print(f"    SILENT ERROR: {r.material_name!r} q={r.material_quantity} v={r.material_value}")

    t = totals
    print("\n=== Summary ===")
    print(f"Invoices: {t['invoices']} | vendor correct: {t['vendor_ok']}/{t['invoices']}")
    print(f"Line items found: {t['found']}/{t['expected']} | extra (not in ground truth): {t['extra']}")
    print(f"Quantity correct: {t['qty_ok']}/{t['expected']} | Value correct: {t['value_ok']}/{t['expected']}")
    print(f"Auto-validated records: {t['auto_validated']} | flagged for review: {t['flagged']}")
    print(f"Silent errors (auto-validated but wrong): {t['silent_errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
