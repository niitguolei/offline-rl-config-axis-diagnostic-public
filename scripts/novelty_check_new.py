"""F1 novelty retrieval for freshly generated offline-RL candidates (round 2).

Discipline: retrieval FIRST. Do not propose a direction until its closest prior
work has been checked against the arXiv API, because two of five directions in
round 1 turned out to be occupied.
"""
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUERIES = {
    "C1 state-density-aware reweighting / balanced sampling": (
        'all:"offline reinforcement learning" AND '
        '(all:"state distribution" OR all:"state density" OR all:"balanced sampling" OR '
        'all:"reweighting" OR all:"importance weighting")'
    ),
    "C2 state-dependent TD(lambda) / adaptive bootstrapping horizon": (
        'all:"reinforcement learning" AND '
        '(all:"partial sweeping" OR all:"learned lambda" OR all:"state-dependent lambda" OR '
        'all:"adaptive bootstrapping" OR all:"variable lambda") AND '
        '(all:"offline" OR all:"batch")'
    ),
    "C3 low-data regime scaling of offline RL": (
        'all:"offline reinforcement learning" AND '
        '(all:"data efficiency" OR all:"limited data" OR all:"sample efficiency" OR '
        'all:"few trajectories" OR all:"scaling")'
    ),
    "C6 reward-free / unlabeled auxiliary objective in offline RL": (
        'all:"offline reinforcement learning" AND '
        '(all:"reward-free" OR all:"unsupervised" OR all:"unlabeled" OR '
        'all:"latent" OR all:"self-supervised")'
    ),
    "C7 evaluation protocol / reproducibility variance": (
        'all:"offline reinforcement learning" AND '
        '(all:"reproducib" OR all:"evaluation protocol" OR all:"variance" OR '
        'all:"seed" OR all:"hyperparameter budget" OR all:"rigorous")'
    ),
    "C6b auxiliary inverse/forward dynamics for offline RL": (
        'all:"offline reinforcement learning" AND '
        '(all:"inverse dynamics" OR all:"forward model auxiliary" OR '
        'all:"self-supervised auxiliary" OR all:"representation learning")'
    ),
}


def search(query, max_results=12):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        }
    )
    req = urllib.request.Request(url, headers={"User-Agent": "codex-research-check"})
    with urllib.request.urlopen(req, timeout=60) as r:
        root = ET.fromstring(r.read().decode("utf-8", "replace"))
    return [
        (
            e.findtext("a:published", "", NS)[:10],
            " ".join(e.findtext("a:title", "", NS).split()),
        )
        for e in root.findall("a:entry", NS)
    ]


for label, q in QUERIES.items():
    print(f"\n{'=' * 78}\n{label}\n  q: {q}\n{'-' * 78}")
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

