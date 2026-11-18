"""Convert the mistral-synthetic data pack into files the LightRAG server can ingest.

The server cannot read ``.eml`` files, so each email becomes one Markdown file:
headers on top, the body with its boilerplate (``IMPORTANT REMINDERS`` and
``DISCLAIMERS``) removed, and the text of any PDF attachments appended. Deck
attachments are not inlined; the current deck versions from the library are
copied as ``.pptx`` instead, which the server parses itself.

The demo estate's product grid becomes one Markdown file per product family
and region, and its research notes one file per underlying with each note as a
dated section (2,386 notes of ~240 characters would otherwise cost one LLM
extraction call each).

Usage:
    python ingest/prepare_mistral.py emails   [--ids em-0010,em-0042] [--out DIR]
    python ingest/prepare_mistral.py decks    [--names "Deck A.pptx,Deck B.pptx"] [--out DIR]
    python ingest/prepare_mistral.py products [--out DIR]
    python ingest/prepare_mistral.py research [--out DIR]
    python ingest/prepare_mistral.py all      [--out DIR]

Without ``--ids`` / ``--names`` every email / current deck is converted.
"""

from __future__ import annotations

import argparse
import ast
import email
import html
import io
import json
import re
import shutil
from collections import defaultdict
from email import policy
from email.message import EmailMessage
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "mistral-synthetic-2026-09-29"
EMAIL_DIR = PACK / "01-email-lookalike" / "emails"
DECK_DIR = PACK / "01-email-lookalike" / "library" / "sites" / "GFM" / "Pitchbooks"
RECORDS_DIR = PACK / "05-test-data" / "records"
DEFAULT_OUT = ROOT / "ingest_staging"
REGIONS = {"SG": "Singapore", "HK": "Hong Kong"}

# Boilerplate sections: identical legal text in most mails, ~29% of body words.
BOILERPLATE_SECTIONS = {"IMPORTANT REMINDERS", "DISCLAIMERS"}
# Body section headings are lines in capitals, e.g. "TRADE IDEAS",
# "12M USDCNH DOUBLE NO TOUCH STRUCTURED DEPOSIT".
SECTION_HEADING = re.compile(r"^[A-Z0-9][A-Z0-9 &/'().,%:+-]{3,}$")


def strip_boilerplate(body: str) -> str:
    """Drop boilerplate sections, keeping everything up to the next heading."""
    kept, skipping = [], False
    for line in body.splitlines():
        stripped = line.strip()
        if SECTION_HEADING.match(stripped):
            skipping = stripped in BOILERPLATE_SECTIONS
        if not skipping:
            kept.append(line)
    text = "\n".join(kept)
    text = re.sub(r"^\[chart\]\s*$", "", text, flags=re.MULTILINE)  # inline PNG placeholders
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _body_text(msg: EmailMessage) -> str:
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    text = part.get_content()
    if part.get_content_type() == "text/html":
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return text


def _display_names(header: str | None) -> str:
    """'Name <addr>, Other <addr>' -> 'Name, Other' (list names carry segment and region)."""
    if not header:
        return ""
    return ", ".join(name or addr for name, addr in email.utils.getaddresses([header]))


def email_to_markdown(path: Path) -> str:
    msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    # No To/Cc and no email id: in the pilot, distribution-list names became
    # dozens of "client segment" entities and the id became an entity of its
    # own. The file name already identifies the mail in citations.
    lines = [
        f"# {msg['Subject']}",
        "",
        f"- Date: {msg['Date']}",
        f"- From: {msg['From']}",
        "",
        strip_boilerplate(_body_text(msg)),
    ]
    for att in msg.iter_attachments():
        name = att.get_filename() or "attachment"
        if att.get_content_type() == "application/pdf":
            reader = PdfReader(io.BytesIO(att.get_content()))
            text = "\n".join((page.extract_text() or "").strip() for page in reader.pages)
            lines += ["", f"## Attachment: {name}", "", strip_boilerplate(text)]
        elif name.lower().endswith(".pptx"):
            lines += ["", f"_Deck attached: {name} (ingested separately from the library)_"]
    return "\n".join(lines).rstrip() + "\n"


def prepare_emails(out: Path, ids: list[str] | None) -> list[Path]:
    paths = sorted(EMAIL_DIR.glob("*.eml"))
    if ids:
        wanted = set(ids)
        paths = [p for p in paths if p.stem.split("_", 1)[-1] in wanted]
        missing = wanted - {p.stem.split("_", 1)[-1] for p in paths}
        if missing:
            raise SystemExit(f"Unknown email ids: {sorted(missing)}")
    dest = out / "emails"
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for p in paths:
        target = dest / f"{p.stem}.md"
        target.write_text(email_to_markdown(p), encoding="utf-8")
        written.append(target)
    return written


def prepare_decks(out: Path, names: list[str] | None) -> list[Path]:
    # Only the current version of each pitchbook; older ones live under _versions/.
    paths = sorted(DECK_DIR.glob("*.pptx"))
    if names:
        paths = [p for p in paths if p.name in set(names)]
        if len(paths) != len(names):
            raise SystemExit(f"Some decks not found in {DECK_DIR}: {names}")
    dest = out / "decks"
    dest.mkdir(parents=True, exist_ok=True)
    return [Path(shutil.copy2(p, dest / p.name)) for p in paths]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def _product_line(row: dict) -> str:
    parts = [
        f"underlying {row['underlying']}",
        f"tenor {row['tenor']}" if row.get("tenor") else None,
        f"currency {row['currency']}" if row.get("currency") else None,
        f"risk rating {row['risk_rating']}",
        f"client segments {', '.join(row['segments'])}" if row.get("segments") else None,
        f"capital protection: {row['capital_protection']}" if row.get("capital_protection") else None,
    ]
    line = f"- {row['product_name']} on {row['underlying']} ({row['id']}): " + "; ".join(p for p in parts if p)
    if row.get("observed_params"):
        line += f". Terms: {row['observed_params']}"
    return line


def prepare_products(out: Path) -> list[Path]:
    """One file per (product family, region), products as sections, rows as bullets."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in _read_jsonl(RECORDS_DIR / "product_grid.jsonl"):
        for country in row["countries"]:
            groups[(row["family"], country)].append(row)
    dest = out / "products"
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for (family, country), rows in sorted(groups.items()):
        region = REGIONS.get(country, country)
        lines = [
            f"# {family} products offered in {region}",
            "",
            f"Product grid rows for the {family} family booked in {region}, "
            f"{len(rows)} rows. Each row is one product on one underlying with its "
            "indicative terms. Client segments: PB is private banking, TPC and TRE "
            "(Treasures) are other wealth segments.",
        ]
        by_name: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_name[row["product_name"]].append(row)
        for name, items in sorted(by_name.items()):
            first = items[0]
            lines += [
                "",
                f"## {name} ({region})",
                "",
                f"Category: {first['category']}. Asset class: {first['asset_class']}. "
                f"Underlyings: {', '.join(sorted({r['underlying'] for r in items}))}.",
                "",
                *(_product_line(r) for r in sorted(items, key=lambda r: (r["underlying"], r["id"]))),
            ]
        target = dest / f"products-{_slug(family)}-{country.lower()}.md"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(target)
    # The pack's own catalogue: a compact overview of every product by region.
    written.append(Path(shutil.copy2(PACK / "PRODUCTS.md", dest / "PRODUCTS.md")))
    return written


def prepare_research(out: Path) -> list[Path]:
    """One file per underlying (first listed), notes as dated sections, oldest first."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for note in _read_jsonl(RECORDS_DIR / "research_docs.jsonl"):
        und = note["underlyings"]
        und = ast.literal_eval(und) if isinstance(und, str) else und
        note["_underlyings"] = und or ["GENERAL"]
        groups[note["_underlyings"][0]].append(note)
    dest = out / "research"
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for underlying, notes in sorted(groups.items()):
        lines = [
            f"# Research notes on {underlying}",
            "",
            f"{len(notes)} notes, oldest first. Each note gives its publication date "
            "and source; prefer the most recent note when levels disagree.",
        ]
        for n in sorted(notes, key=lambda n: (n["publication_date"], n["id"])):
            # Compact one-line header: per-note metadata is ~40% of the text otherwise.
            also = f", also {', '.join(n['_underlyings'][1:])}" if len(n["_underlyings"]) > 1 else ""
            lines += [
                "",
                f"## {n['publication_date']} {n['title']} "
                f"({n['channel_label']}, {n['subtype']}, {n['sentiment']}{also})",
                n["body"].strip(),
            ]
        target = dest / f"research-{_slug(underlying)}.md"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["emails", "decks", "products", "research", "all"])
    parser.add_argument("--ids", help="comma-separated email ids, e.g. em-0010,em-0042")
    parser.add_argument("--names", help="comma-separated deck file names")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    written: list[Path] = []
    if args.source in ("emails", "all"):
        written += prepare_emails(args.out, args.ids.split(",") if args.ids else None)
    if args.source in ("decks", "all"):
        written += prepare_decks(args.out, args.names.split(",") if args.names else None)
    if args.source in ("products", "all"):
        written += prepare_products(args.out)
    if args.source in ("research", "all"):
        written += prepare_research(args.out)
    if len(written) <= 20:
        for p in written:
            print(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
    print(f"{len(written)} file(s) written to {args.out}")


if __name__ == "__main__":
    main()
