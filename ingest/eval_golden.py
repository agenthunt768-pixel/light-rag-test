"""Run the golden-set questions against the LightRAG server and score the answers.

Each question is sent to ``/query`` together with its client profile. The
golden "Expected Answer" is a list of product ideas written from the bank's
real emails; this pack only partly covers them, so treat scores as relative
(between runs or settings), not as absolute accuracy.

Scoring is coarse on purpose: each expected line is reduced to the product
families it names (TARF, DCL, CLI, DNT, participation note, ...), and the line
counts as recalled when the answer mentions any of those families. Lines that
name no known family (commentary, reviewer notes) are not scored.

Usage:
    python ingest/eval_golden.py [--mode mix] [--ids PR002,PR004] [--out eval_results]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "mistral-synthetic-2026-09-29" / "golden-set(1).xlsx"
SERVER = "http://127.0.0.1:9621"

# Product family -> regex over expected lines and answers (case-insensitive).
FAMILIES: dict[str, str] = {
    "TARF": r"\bTARFs?\b|target redemption",
    "Accumulator": r"\bAQ\b|accumulator",
    "Decumulator": r"\bDQ\b|decumulator",
    "Dual Currency Loan": r"\bDCLs?\b|dual currency loan",
    "Currency Linked Investment": r"\bCLIs?\b|currency linked investment",
    "Double No Touch": r"\bDNT\b|double no touch",
    "Digital": r"\bdigital\b",
    "Participation": r"participation|participaton|\bBB ?P? ?SIP\b|bull ?bear",
    "Put Spread": r"put spread",
    "Range Accrual": r"range accrual|\bCRAN\b|\bDRAN\b|range strategy|range-bound \w+ trade expressions",
    "Synthetic Loan": r"synthetic loan",
    "Liability Management": r"liability management",
    "Lookback": r"lookback",
    "Commodity Linked Note": r"(gold|bullion|brent)[- ]linked",
    "Autocallable": r"auto ?-?callable",
    "Triple Currency Note": r"triple currency|\bTCN\b",
    "Leveraged Note": r"leveraged (capped )?(participation )?note",
    "Warrant": r"warrant",
    "Vanilla Option": r"\b(buy|sell)\s+(an?\s+)?(\w+\s+){0,2}(call|put)\b(?!\s+spread)|covered call|(put|call) option",
    "Strangle/Straddle": r"strangle|straddle",
    "Risk Reversal": r"risk reversal",
    "KIKO": r"\bE?KIKO\b",
    "Structured Deposit": r"structured deposit|\bSDs?\b",
}
COMPILED = {name: re.compile(rx, re.IGNORECASE) for name, rx in FAMILIES.items()}


def families_in(text: str) -> set[str]:
    return {name for name, rx in COMPILED.items() if rx.search(text)}


def load_golden(ids: set[str] | None) -> list[dict]:
    ws = openpyxl.load_workbook(GOLDEN, read_only=True, data_only=True)["Golden set"]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r or len(r) < 2 or not r[0] or not r[1]:
            continue
        if ids and r[0] not in ids:
            continue
        # PRX rows carry only an id and a question: asked, but not scored.
        r = tuple(r) + (None,) * (4 - len(r))
        rows.append({"id": r[0], "question": r[1].strip(), "profile": r[2] or "", "expected": r[3] or ""})
    return rows


def query(question: str, profile: str, mode: str) -> tuple[str, list[str], float]:
    text = f"{question}\n\nClient profile: {profile}" if profile else question
    body = json.dumps({"query": text, "mode": mode, "enable_rerank": False, "include_references": True}).encode()
    req = urllib.request.Request(f"{SERVER}/query", data=body, headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    refs = [ref.get("file_path", "") for ref in data.get("references") or []]
    return data.get("response", ""), refs, time.time() - start


def score(expected: str, answer: str) -> tuple[int, int, list[str]]:
    answer_families = families_in(answer)
    scored = hit = 0
    missed = []
    for line in expected.splitlines():
        fams = families_in(line)
        if not fams:
            continue
        scored += 1
        if fams & answer_families:
            hit += 1
        else:
            missed.append(line.strip())
    return hit, scored, missed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="mix", choices=["naive", "local", "global", "hybrid", "mix"])
    parser.add_argument("--ids", help="comma-separated question ids, e.g. PR002,PR004")
    parser.add_argument("--out", type=Path, default=ROOT / "eval_results")
    args = parser.parse_args()

    rows = load_golden(set(args.ids.split(",")) if args.ids else None)
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"golden-{args.mode}-{datetime.now():%Y%m%d-%H%M}.csv"

    total_hit = total_scored = 0
    with out_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "id", "question", "profile", "recall", "hit", "scored", "missed_expected",
            "expected", "answer", "references", "seconds"])
        writer.writeheader()
        for row in rows:
            try:
                answer, refs, secs = query(row["question"], row["profile"], args.mode)
            except Exception as exc:  # keep going; record the failure
                answer, refs, secs = f"ERROR: {exc}", [], 0.0
            hit, scored, missed = score(row["expected"], answer)
            total_hit += hit
            total_scored += scored
            recall = f"{hit / scored:.2f}" if scored else ""
            print(f"{row['id']:7} recall={recall or 'n/a':4} ({hit}/{scored})  {secs:5.1f}s  {row['question'][:60]}")
            writer.writerow({**row, "recall": recall, "hit": hit, "scored": scored,
                             "missed_expected": " | ".join(missed), "answer": answer,
                             "references": " | ".join(refs), "seconds": f"{secs:.1f}"})
    overall = total_hit / total_scored if total_scored else 0
    print(f"\nOverall family recall: {total_hit}/{total_scored} = {overall:.2f}  ({len(rows)} questions, mode={args.mode})")
    print(f"Results: {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
