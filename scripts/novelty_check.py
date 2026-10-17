"""Query arXiv API for closest prior work on each of the 5 candidate directions."""
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import time

NS = {"a": "http://www.w3.org/2005/Atom"}

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUERIES = {
    "D1 OPE / model selection / early stopping": (
        'all:"offline reinforcement learning" AND '
        '(all:"model selection" OR all:"hyperparameter selection" OR all:"early stopping")'
    ),
    "D1b OPE-guided algorithm design": (
        'all:"offline reinforcement learning" AND '
        '(all:"off-policy evaluation" OR all:"policy evaluation") AND '
        '(all:"pessimism" OR all:"conservative" OR all:"guidance")'
    ),
    "D2 symmetry / equivariance": (
        'all:"offline reinforcement learning" AND '
        '(all:"equivariant" OR all:"symmetry" OR all:"data augmentation")'
    ),
    "D3 behaviour-model-free constraint": (
        'all:"offline reinforcement learning" AND '
        '(all:"support constraint" OR all:"without a behavior model" OR '
        'all:"behavior policy estimation" OR all:"behaviour policy")'
    ),
    "D4 dataset curation / selection": (
        'all:"offline reinforcement learning" AND '
        '(all:"dataset selection" OR all:"data selection" OR all:"data curation" OR '
        'all:"subset selection" OR all:"data quality")'
    ),
    "D5 offline-to-online fine-tuning safety": (
        'all:"offline-to-online" AND (all:"reinforcement learning" OR all:"fine-tuning")'
    ),
}


def search(query: str, max_results: int = 12):
    url = (
        "http://export.arxiv.org/api/query?"
        + urllib.parse.urlencode(
            {
                "search_query": query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": max_results,
            }
        )
    )
    req = urllib.request.Request(url, headers={"User-Agent": "codex-research-check"})
    with urllib.request.urlopen(req, timeout=60) as r:
        root = ET.fromstring(r.read().decode("utf-8", "replace"))
    out = []
    for e in root.findall("a:entry", NS):
        out.append(
            (
                e.findtext("a:published", "", NS)[:10],
                " ".join(e.findtext("a:title", "", NS).split()),
            )
        )
    return out


for label, q in QUERIES.items():
    print(f"\n{'=' * 78}\n{label}\n  query: {q}\n{'-' * 78}")
    try:
        rows = search(q)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {exc}")
        continue
    if not rows:
        print("  (no results)")
    for date, title in rows:
        print(f"  [{date}] {title}")
    time.sleep(3)
