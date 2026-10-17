"""Extract plain text from the survey PDFs in paper/review into paper_extracted/."""
import pathlib
import re

import pdfplumber

ROOT = pathlib.Path(__file__).parent
SRC = ROOT / "paper" / "review"
DST = ROOT / "paper_extracted"
DST.mkdir(exist_ok=True)


def slug(name: str) -> str:
    s = name.replace(".pdf", "")
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]", "", s)
    return s[:80]


for pdf in sorted(SRC.glob("*.pdf")):
    out = DST / f"{slug(pdf.name)}.txt"
    if out.exists() and out.stat().st_size > 2000:
        print(f"skip (exists): {out.name}")
        continue
    pages = []
    with pdfplumber.open(pdf) as doc:
        for i, page in enumerate(doc.pages, 1):
            pages.append(f"\n=== PAGE {i} ===\n{page.extract_text() or ''}")
    text = "\n".join(pages)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out.name}: {len(text)} chars, {len(pages)} pages")
