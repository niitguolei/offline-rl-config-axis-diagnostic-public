"""T4 -- GATE 1: does critic overestimation grow with the backup length, and is
coverage the moderator?

Run grid (7 trained here, 1 reused):

    n=1   seeds 0,1,2      <- slope endpoints, 3 seeds
    n=10  seeds 0,1,2      <- slope endpoints, 3 seeds
    n=3   seed  0          <- shape of the curve
    n=5   seed  0          <- REUSED from the calibration run, not retrained

Every run records, per coverage bucket, both the positive overestimation and
the SIGNED bias. The signed value matters: a smoke run showed the low-coverage
bucket sitting at exactly 0.000 on the positive part, which is equally
consistent with the critic under-estimating there.

Rebuilt on the vendored reference for the S1 restart. Two things the previous
version got wrong and this one cannot:

  * the reward rescaling. The critic predicts values on the rescaled reward,
    so the MC reference is multiplied by `d.reward_scale` (0.1778 here) before
    the difference is taken. Comparing raw MC returns against a rescaled critic
    manufactures a bias of ~5.6x.
  * the baseline. `SupportAwareIQL` with `coverage=None, fixed_n=n` is a
    subclass of the verified reference whose control path is unit-tested to be
    parameter-identical to it, and `fixed_n=1` is unit-tested to reproduce the
    reference's own Bellman target away from episode boundaries.

Results are written after EVERY run and `--resume` skips runs already present,
because this grid is >10 GPU-hours and the previous round lost 15 hours to a
crash that discarded completed work.

    python experiments/gate1.py --dry-run
    python experiments/gate1.py
    python experiments/gate1.py --resume
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

sys.path.insert(0, str(ROOT))

from src.policy import GaussianPolicy  # noqa: E402
from src.value_functions import TwinQ, ValueFunction  # noqa: E402

from orl.algos.mc_reference import per_bucket_bias, reference_callables  # noqa: E402
from orl.coverage.buckets import buckets_for  # noqa: E402
from orl.data.d4rl_hdf5 import D4RLDataset  # noqa: E402
from orl.data.episode import EpisodeIndexLookup  # noqa: E402
from orl.methods.harness import evaluate, load  # noqa: E402
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR, parse_env_id  # noqa: E402
from orl.record.logger import RunGuard  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

ENV_ID = "halfcheetah_medium_replay-v2"
GAMMA = 0.99
BATCH = 256
STATES_PER_BUCKET = 1000
MC_HORIZON = 1000
EVAL_EPISODES = 20

RUNS = [(1, 0), (1, 1), (1, 2), (3, 0), (10, 0), (10, 1), (10, 2)]
REUSED = (5, 0)          # comes from the calibration run
OUT_NAME = "gate1_runs.json"


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def nstep_kwargs(d, j):
    return {"indices": j, "dataset_rewards": d.rew,
            "dataset_next_obs": d.next_obs, "dataset_terminals": d.terminals}


def steps_from_calibration(cal: dict) -> int:
    """Baseline budget = last checkpoint before the score moved by >= 5%."""
    if cal.get("smoke"):
        raise SystemExit("calibration.json is a SMOKE run; it cannot set GATE 1's budget. Run experiments/calibrate.py to completion first.")
    if max(cal["checkpoints"]) < 10_000:
        raise SystemExit(f"calibration tops out at {max(cal['checkpoints']):,} steps, too small to be a real calibration")
    recs = cal["records"]
    scores = [r["policy_raw"] for r in recs]
    steps = [r["step"] for r in recs]
    for i in range(1, len(scores)):
        prev, cur = scores[i - 1], scores[i]
        if prev != 0 and abs(cur - prev) / abs(prev) >= 0.05:
            return steps[i - 1]
    return steps[-1]


def train_one(d, ep, buckets, raw_obs, n_step: int, seed: int, steps: int,
              args, dev: str) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    agent = SupportAwareIQL(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=args.lr),
        max_steps=steps, tau=args.tau, beta=args.beta, alpha=args.alpha,
        discount=args.discount, episode_lookup=ep,
        coverage=None, fixed_n=n_step, n_min=1, n_max=n_step,
    )
    O, A, R = to_t(d.obs, dev), to_t(d.act, dev), to_t(d.rew, dev)
    NO, TE = to_t(d.next_obs, dev), to_t(d.terminals, dev)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    last = {}
    for i in range(steps):
        j = np.sort(rng.integers(0, len(O), BATCH))
        last = agent.update(O[j], A[j], NO[j], R[j], TE[j], **nstep_kwargs(d, j))
        if (i + 1) % 50_000 == 0:
            print(f"      {i + 1:,}/{steps:,} "
                  f"({(time.perf_counter() - t0) / 60:.0f} min)", flush=True)
    train_s = time.perf_counter() - t0

    ev = evaluate(agent.policy, d.domain, d.normer, n_episodes=args.eval_episodes)
    policy_fn, q_fn = reference_callables(agent, d.normer, device=dev)
    b = per_bucket_bias(
        d.domain, raw_obs, np.digitize(buckets.scores, buckets.edges),
        buckets.edges, policy_fn, q_fn, gamma=GAMMA, horizon=args.mc_horizon,
        states_per_bucket=args.states_per_bucket, device=dev,
        mc_scale=d.reward_scale)
    with torch.no_grad():
        probe = np.sort(np.random.default_rng(7).choice(len(O), 4096,
                                                       replace=False))
        v_mean = float(agent.vf(O[probe]).mean())
    return {"n_step": n_step, "seed": seed, "steps": steps,
            "train_min": train_s / 60.0, "policy_raw": ev["raw_mean"],
            "policy_std": ev["raw_std"], "policy_score": ev["score_mean"],
            "V_mean": v_mean, "q_loss": last["q_loss"],
            "v_loss": last["value_loss"], "buckets": b}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, help="override the calibrated budget")
    ap.add_argument("--only", default=None,
                    help="comma separated n:seed pairs, e.g. 1:0,10:0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--discount", type=float, default=GAMMA)
    ap.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    ap.add_argument("--states-per-bucket", type=int, default=STATES_PER_BUCKET)
    ap.add_argument("--mc-horizon", type=int, default=MC_HORIZON)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.eval_episodes, args.states_per_bucket = 2, 24
        args.mc_horizon = 50
        args.steps = 2000

    cal_path = BENCH_DIR / "calibration.json"
    if args.steps:
        steps, src = args.steps, "override"
    elif cal_path.exists():
        steps = steps_from_calibration(json.loads(
            cal_path.read_text(encoding="utf-8")))
        src = "calibration"
    else:
        print("calibration.json missing -- run experiments/calibrate.py first")
        return 2

    todo = list(RUNS)
    if args.only:
        want = {tuple(int(x) for x in p.split(":")) for p in args.only.split(",")}
        todo = [r for r in todo if r in want]
    # A smoke run must not be able to overwrite the real grid's records.
    out_p = BENCH_DIR / ("gate1_smoke.json" if args.smoke else OUT_NAME)
    records: list[dict] = []
    if args.resume and out_p.exists():
        records = json.loads(out_p.read_text(encoding="utf-8")).get("records", [])
    done = {(r["n_step"], r["seed"]) for r in records}
    todo = [r for r in todo if r not in done]

    print(f"step budget: {steps:,} (from {src})")
    print(f"reusing n={REUSED[0]} seed={REUSED[1]} from the calibration run")
    print(f"training {len(todo)} run(s): {todo}")
    print(f"estimated {len(todo) * steps * 0.013 / 3600:.1f} h at 13 ms/step\n")
    if args.dry_run:
        return 0
    if not todo:
        print("nothing to do")
        return 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    d = load(ENV_ID, with_coverage=False, normalize=True)
    buckets = buckets_for(ENV_ID)
    print(f"reward: raw {d.raw_rew_mean:.4f} -> scaled {d.scaled_rew_mean:.4f} "
          f"(x{d.reward_scale:.4f}); MC returns scaled to match\n")
    with D4RLDataset(ENV_ID, mode="stream") as ds:
        raw_obs = ds.get("observations")
        ep = EpisodeIndexLookup(ds.episodes.starts(), ds.episodes.ends, ds.n)

    domain, tier = parse_env_id(ENV_ID)
    for n_step, seed in todo:
        print(f"=== n={n_step} seed={seed} ===", flush=True)
        with RunGuard(run_id=f"gate1_n{n_step}_s{seed}", method=f"iql+fixed_n{n_step}",
                      dataset=ENV_ID, domain=domain, tier=tier, seed=seed,
                      stage="S1", notes=f"T4 GATE 1 run, {steps:,} steps"):
            rec = train_one(d, ep, buckets, raw_obs, n_step, seed, steps,
                            args, dev)
        records.append(rec)
        line = f"  policy={rec['policy_raw']:.1f} V={rec['V_mean']:.2f}"
        for k in ("0", "1", "2"):
            b = rec["buckets"][k]
            line += (f" | b{k} {b['signed_bias_mean']:+.3f}"
                     f"+-{b['signed_bias_sem']:.3f}")
        print(line, flush=True)

        out = {"env_id": ENV_ID, "steps": steps, "steps_source": src,
               "gamma": GAMMA, "reward_scale": d.reward_scale,
               "reused": list(REUSED), "planned": [list(r) for r in RUNS],
               "baseline": "vendored gwthomas/IQL-PyTorch via SupportAwareIQL",
               "records": records}
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  saved {out_p.name} ({len(records)} records)", flush=True)
    print(f"\ndone. now: python experiments/gate1_verdict.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
