"""I4 — append-only experiment records.

Terminal scrollback is not a record. Every training run must leave a row in
results/runs/YYYY-MM.jsonl, including failures: the GATE criteria depend on
knowing what was tried, and "an experiment without a record did not happen".

scripts/log_run.py is a thin CLI over this module, so the CLI and the
training loop cannot drift apart.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

from ..paths import RESULTS_DIR

RUNS_DIR = RESULTS_DIR / "runs"


def env_snapshot() -> dict[str, Any]:
    """Capture the execution environment so results stay interpretable."""
    snap: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch

        snap["torch"] = torch.__version__
        snap["cuda_build"] = torch.version.cuda
        snap["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["device"] = torch.cuda.get_device_name(0)
            snap["capability"] = ".".join(
                str(x) for x in torch.cuda.get_device_capability(0)
            )
    except ImportError:
        snap["torch"] = "missing"
    return snap


def git_rev() -> str | None:
    """Short commit hash of the working tree at record time."""
    exe = (
        r"C:\Users\Gray\.cache\codex-runtimes\codex-primary-runtime"
        r"\dependencies\native\git\cmd\git.exe"
    )
    root = pathlib.Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            [exe, "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def record_path(when: dt.datetime | None = None) -> pathlib.Path:
    when = when or dt.datetime.now()
    return RUNS_DIR / f"{when.strftime('%Y-%m')}.jsonl"


def append(rec: dict[str, Any]) -> pathlib.Path:
    """Append one JSON record. Crash-safe: a kill cannot corrupt earlier rows."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = record_path()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def log_run(
    *,
    run_id: str,
    method: str,
    dataset: str,
    domain: str,
    tier: str,
    seed: int,
    stage: str = "S0",
    status: str = "done",
    score_norm: float | None = None,
    score_raw: float | None = None,
    wall_clock_min: float | None = None,
    peak_vram_gb: float | None = None,
    config: str | None = None,
    notes: str = "",
) -> pathlib.Path:
    """Write one fully-populated record."""
    rec = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "stage": stage,
        "method": method,
        "dataset": dataset,
        "domain": domain,
        "tier": tier,
        "seed": seed,
        "status": status,
        "score_norm": score_norm,
        "score_raw": score_raw,
        "wall_clock_min": wall_clock_min,
        "peak_vram_gb": peak_vram_gb,
        "config": config,
        "notes": notes,
        "git_rev": git_rev(),
        "env": env_snapshot(),
    }
    return append(rec)


class RunGuard:
    """Context manager guaranteeing a record even when the run crashes.

    S0 acceptance criterion: "a failed or crashed run must still leave a
    status='failed' row" -- a crash must not be able to eat the evidence.
    """

    def __init__(self, **fields: Any) -> None:
        self.fields = fields
        self._t0 = dt.datetime.now()
        self._c0 = time.perf_counter()

    def __enter__(self) -> "RunGuard":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        fields = dict(self.fields)
        fields.setdefault("wall_clock_min", (time.perf_counter() - self._c0) / 60.0)
        if exc_type is not None:
            fields["status"] = "failed"
            fields["notes"] = (fields.get("notes", "") + f" | {exc_type.__name__}: {exc}").strip(" |")
        log_run(**fields)
        return False  # never swallow the exception
