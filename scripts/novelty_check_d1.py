"""Focused follow-up: is 'OPE confidence as a training-time signal for conservatism
and checkpoint selection' already occupied?"""
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import time

NS = {"a": "http://www.w3.org/2005/Atom"}
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUERIES = {
    "OPE as in-training feedback / abstention": (
        'all:"offline reinforcement learning" AND '
        '(all:"confidence interval" OR all:"uncertainty") AND '
        '(all:"checkpoint selection" OR all:"training dynamics" OR all:"model selection" OR '
        'all:"adaptive conservatism")'
    ),
    "oracle / regret metric for offline model selection": (
        'all:"offline reinforcement learning" AND '
        '(all:"oracle" OR all:"regret") AND '
        '(all:"model selection" OR all:"hyperparameter")'
    ),
    "adaptive pessimism coefficient schedule": (
        'all:"offline reinforcement learning" AND '
        '(all:"adaptive" OR all:"adaptive penalty" OR all:"time-varying") AND '
        '(all:"pessimism" OR all:"conservatism" OR all:"penalty coefficient")'
    ),
    "early stopping / training-length offline RL": (
        'all:"offline reinforcement learning" AND '
        '(all:"early stopping" OR all:"training length" OR all:"when to stop")'
    ),
}


def search(query, max_results=10):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "sortBy": "submittedDate",
         "sortOrder": "descending", "max_results": max_results}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "codex-research-check"})
    with urllib.request.urlopen(req, timeout=60) as r:
        root = ET.fromstring(r.read().decode("utf-8", "replace"))
    return [(e.findtext("a:published", "", NS)[:10],
             " ".join(e.findtext("a:title", "", NS).split()))
            for e in root.findall("a:entry", NS)]


for label, q in QUERIES.items():
    print(f"\n{'=' * 78}\n{label}\n{'-' * 78}")
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

