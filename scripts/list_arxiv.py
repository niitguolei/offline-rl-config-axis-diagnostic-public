"""List titles + dates from the arXiv API result XMLs in the workspace root."""
import pathlib
import re

ROOT = pathlib.Path(__file__).parent

for f in sorted(ROOT.glob("arxiv_*.xml")):
    raw = f.read_text(encoding="utf-8", errors="replace")
    titles = re.findall(r"<title>(.*?)</title>", raw, re.S)
    dates = re.findall(r"<published>(.*?)</published>", raw, re.S)
    # entry 0 is the feed title; real papers start at index 1
    print(f"\n===== {f.name}  ({max(0, len(titles)-1)} papers) =====")
    for i, t in enumerate(titles[1:], 1):
        t = re.sub(r"\s+", " ", t).strip()
        d = dates[i - 1][:10] if i - 1 < len(dates) else "?"
        print(f"  [{d}] {t}")
