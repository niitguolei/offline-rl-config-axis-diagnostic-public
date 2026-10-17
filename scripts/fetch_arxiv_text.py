"""Fetch an arXiv paper's HTML rendering and write a plain-text extraction.

The repo keeps papers as `paper/<topic>/<file>.pdf` plus a committed
`extracted_text/<file>.txt`, and the PDFs are gitignored. This machine has no
PDF extractor installed (no pdftotext, no pdfplumber/pypdf/fitz), so for papers
that have an HTML rendering this goes straight from HTML to text instead of
installing one. It is stdlib-only on purpose.

arXiv serves HTML for recent submissions; older ones go through ar5iv. Both
were verified reachable for the two papers this was written for.

Usage:
    python scripts/fetch_arxiv_text.py --url https://arxiv.org/html/2502.08021v3 \
        --out paper/model_selection/extracted_text/Liu_2025_ModelSelection_OPE.txt
"""

from __future__ import annotations

import argparse
import html
import pathlib
import re
import sys
import urllib.request


def strip_html(raw: str) -> str:
    """HTML to readable text, keeping block structure and marking math."""
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    # Equations are images/MathML here; mark them so a reader knows something
    # was dropped rather than silently losing a formula.
    raw = re.sub(r"(?is)<math[^>]*>.*?</math>", " [MATH] ", raw)
    raw = re.sub(r"(?is)<(figure|table)[^>]*>.*?</\1>", " [FIGURE/TABLE] ", raw)
    raw = re.sub(r"(?i)</(p|div|section|li|h[1-6]|tr|figcaption|blockquote)>", "\n", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = html.unescape(text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    req = urllib.request.Request(
        args.url, headers={"User-Agent": "Mozilla/5.0 (research reading script)"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    text = strip_html(raw)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"{args.url}\n  -> {out}  ({len(text):,} chars, "
          f"{text.count(chr(10)):,} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
