"""E0: is there anything for checkpoint selection to do?

Two modes, per plan v4 sections 3.0 and 11 (review U1).

  --from-jsonl  analyse an existing failure-map file with the shared reporter,
                so `last` / `random` / TRIVIAL appear on every row. Zero GPU.
  --probe       run new probes searching for a NON-TRIVIAL regime, because the
                existing four conditions are all trivial by the gap leg (best
                2.9%). Directions from the plan: 1% subsampling (Cheng's regime,
                where overfitting is most likely), longer training (does the
                policy degrade?), and harder tiers.

The probes record PER-EPISODE returns, which the existing map does not. That is
what makes the monotonicity leg evaluable: review R1 found the ranking truth had
an SEM of 0.34-1.7% against truth spreads of 1-2.4%, and a criterion that any
noise-sized dip can trigger is worse than no criterion (a test caught exactly
that).

    python experiments/e0_triviality.py --from-jsonl
    python experiments/e0_triviality.py --probe --smoke
    python experiments/e0_triviality.py --probe
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

from orl.data.d4rl_hdf5 import D4RLDataset  # noqa: E402
from orl.gaps.construct import subsample_episodes  # noqa: E402
from orl.methods.harness import episode_lookup, load  # noqa: E402
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402
from orl.selection.report import (  # noqa: E402
    format_table,
    nontriviality,
    oracle_gap,
    summarise,
)

sys.stdout.reconfigure(line_buffering=True)

# probe list from the plan: (domain/tier, subsample fraction, max steps)
PROBES = [
    ("halfcheetah_medium_replay-v2", 0.01, 100_000),
    ("halfcheetah_medium_replay-v2", 1.00, 200_000),
    ("halfcheetah_expert-v2", 1.00, 100_000),
    ("walker2d_medium_replay-v2", 0.01, 100_000),
    ("hopper_medium_replay-v2", 1.00, 100_000),
]


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def evaluate_per_episode(policy, env_id, normer, n_episodes: int, seed: int = 0):
    """Like harness.evaluate but returns the per-episode returns (review R1)."""
    import gymnasium as gym

    from orl.envs.obs_layout import ENV_IDS
    from orl.envs.ref_scores import normalized_score

    domain = env_id.split("_")[0]
    dev = next(policy.parameters()).device
    env = gym.make(ENV_IDS[domain])
    rets = []
    try:
        for ep in range(n_episodes):
            o, _ = env.reset(seed=seed + ep)
            done, total = False, 0.0
            while not done:
                ot = torch.as_tensor(normer(o[None, :]), dtype=torch.float32,
                                     device=dev)
                with torch.no_grad():
                    dist = policy(ot)
                    vec = dist if torch.is_tensor(dist) else dist.mean
                    a = vec.detach().cpu().numpy()[0]
                o, r, term, trunc, _ = env.step(np.clip(a, -1.0, 1.0))
                total += float(r)
                done = term or trunc
            rets.append(total)
    finally:
        env.close()
    rets = np.asarray(rets, dtype=np.float64)
    return {"raw_returns": rets.tolist(), "raw_mean": float(rets.mean()),
            "raw_sem": float(rets.std(ddof=1) / np.sqrt(len(rets))),
            "scores": [normalized_score(domain, r) for r in rets]}


def run_probe(env_id: str, fraction: float, steps: int, dev: str,
              checkpoints: int, eval_episodes: int, seed: int) -> dict:
    d = load(env_id, with_coverage=False, normalize=True)
    ep_base = episode_lookup(d.env_id)
    base = {"obs": d.obs, "act": d.act, "rew": d.rew, "next_obs": d.next_obs,
            "terminals": d.terminals, "timeouts": d.timeouts,
            "starts": np.concatenate(([0], ep_base.ends[:-1] + 1)),
            "ends": ep_base.ends}
    g = None if fraction >= 1.0 else subsample_episodes("sub", base, fraction,
                                                        seed=seed)
    obs = base["obs"] if g is None else g.obs
    act = base["act"] if g is None else g.act
    rew = base["rew"] if g is None else g.rew
    nobs = base["next_obs"] if g is None else g.next_obs
    term = base["terminals"] if g is None else g.terminals
    ep = ep_base if g is None else g.episode_lookup()
    O, A = to_t(obs, dev), to_t(act, dev)
    R, NO, TE = to_t(rew, dev), to_t(nobs, dev), to_t(term, dev)
    torch.manual_seed(seed)
    agent = SupportAwareIQL(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=3e-4),
        max_steps=steps, tau=0.7, beta=3.0, alpha=0.005, discount=0.99,
        episode_lookup=ep, coverage=None, fixed_n=1, n_min=1, n_max=1,
    )
    rng = np.random.default_rng(seed)
    cps = np.linspace(steps // checkpoints, steps, checkpoints, dtype=int).tolist()
    rows, done = [], 0
    for cp in cps:
        while done < cp:
            j = np.sort(rng.integers(0, len(O), 256))
            agent.update(O[j], A[j], NO[j], R[j], TE[j])
            done += 1
        ev = evaluate_per_episode(agent.policy, d.env_id, d.normer,
                                  eval_episodes, seed=0)
        rows.append({"step": int(cp), "raw_mean": ev["raw_mean"],
                     "raw_sem": ev["raw_sem"],
                     "score_mean": float(np.mean(ev["scores"])),
                     "score_sem": float(np.std(ev["scores"], ddof=1)
                                        / np.sqrt(len(ev["scores"]))),
                     "raw_returns": ev["raw_returns"]})
        print(f"    {env_id} frac={fraction} {cp:>7,}  raw {ev['raw_mean']:8.1f}"
              f" +- {ev['raw_sem']:5.1f}", flush=True)
    truth = np.array([r["score_mean"] for r in rows])
    sem = float(np.mean([r["score_sem"] for r in rows]))
    nt = nontriviality(truth, [r["step"] for r in rows], sem=sem)
    return {"env": d.env_id, "fraction": fraction, "seed": seed,
            "n_data": int(obs.shape[0]),
            "steps": steps, "checkpoints": cps, "rows": rows,
            "score_sem_mean": sem, "triviality": nt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-jsonl", action="store_true")
    ap.add_argument("--from-probes", action="store_true",
                    help="re-apply the criterion to saved E0c probes (0 GPU)")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--source", default="failure_map_map2.jsonl")
    ap.add_argument("--checkpoints", type=int, default=6)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--envs", default=None,
                    help="probe mode: comma separated env ids (default PROBES)")
    ap.add_argument("--fractions", default=None,
                    help="probe mode: comma separated subsample fractions; "
                         "crossed with --envs and --seeds")
    ap.add_argument("--seeds", default="0",
                    help="probe mode: comma separated seeds, so the SEED SPREAD "
                         "of the truth level is measurable -- E1 showed the 1% "
                         "regime varies 3-4x across seeds")
    ap.add_argument("--probe-steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.checkpoints, args.eval_episodes = 3, 3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_jsonl or not (args.probe or args.from_probes):
        p = BENCH_DIR / args.source
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        s = summarise(rows, {"critic": "critic_mean", "fqe": "fqe",
                             "pdis": "pdis_clip2.0", "wis": "wis_clip0.5"})
        print("E0a: existing map with the trivial baselines on every row")
        print(format_table(s))
        print("\nverdicts:")
        for cond, e in s.items():
            t = e["triviality"]
            print(f"  {cond:<7} last_gap {t['last_gap']:6.1%}  "
                  f"gap_leg {str(t['gap_leg']):<5}  "
                  f"peak {t['peak_size']:6.2f} ({t['monotonicity_leg']})  "
                  f"-> {'TRIVIAL' if t['trivial'] else 'NON-TRIVIAL'}")
        (BENCH_DIR / "e0a_baselines.json").write_text(
            json.dumps(s, indent=2), encoding="utf-8")
        print(f"\nwrote e0a_baselines.json")
        return 0

    if args.from_probes:
        p = BENCH_DIR / ("e0c_probes_smoke.json" if args.smoke else "e0c_probes.json")
        probes = json.loads(p.read_text(encoding="utf-8"))["probes"]
        print("E0c verdicts under the current criterion (practical floor included)")
        print(f"{'probe':<44}{'last gap':>10}{'peak':>9}{'rel':>8}"
              f"{'sem':>7}  verdict")
        for r in probes:
            truth = np.array([x["score_mean"] for x in r["rows"]])
            steps = [x["step"] for x in r["rows"]]
            nt = nontriviality(truth, steps, sem=r["score_sem_mean"])
            label = (f"{r['env'].split('-v2')[0]} "
                     f"frac={r['fraction']} steps={r['steps']//1000}k")
            print(f"{label:<44}{nt['last_gap']:>10.1%}{nt['peak_size']:>9.2f}"
                  f"{nt['peak_relative']:>8.2%}{r['score_sem_mean']:>7.2f}  "
                  f"{'TRIVIAL' if nt['trivial'] else 'NON-TRIVIAL'}"
                  f"  ({nt['monotonicity_leg']})")
        n_non = sum(1 for r in probes
                    if not nontriviality([x["score_mean"] for x in r["rows"]],
                                         [x["step"] for x in r["rows"]],
                                         sem=r["score_sem_mean"])["trivial"])
        print(f"\n{n_non}/{len(probes)} probes are NON-TRIVIAL -> "
              f"GATE 4b {'PASSES' if n_non else 'FAILS'}")
        return 0

    if args.envs or args.fractions:
        envs = (args.envs or "walker2d_medium_replay-v2").split(",")
        fracs = [float(x) for x in (args.fractions or "0.01").split(",")]
        steps = args.probe_steps or (2000 if args.smoke else 66_000)
        probes = [(e, f, steps) for e in envs for f in fracs]
    else:
        probes = PROBES
    seeds = [int(x) for x in args.seeds.split(",")]
    if args.smoke:
        probes = probes[:1]
    out_p = BENCH_DIR / ("e0c_probes_smoke.json" if args.smoke else "e0c_probes.json")
    results = []
    if out_p.exists():
        results = json.loads(out_p.read_text(encoding="utf-8")).get("probes", [])
    done = {(r["env"], r["fraction"], r["steps"], r.get("seed", 0))
            for r in results}
    t0 = time.perf_counter()
    for env_id, frac, steps in probes:
        for seed in seeds:
            if (env_id, frac, steps, seed) in done:
                print(f"  {env_id} frac={frac} seed={seed} cached, skipping")
                continue
            print(f"E0c probe: {env_id}  subsample={frac}  steps={steps:,}  "
                  f"seed={seed}", flush=True)
            r = run_probe(env_id, frac, steps, dev, args.checkpoints,
                          args.eval_episodes, seed)
            results.append(r)
            out_p.write_text(json.dumps({"probes": results}, indent=2),
                             encoding="utf-8")
            t = r["triviality"]
            print(f"  -> last_gap {t['last_gap']:6.1%}  peak {t['peak_size']:6.2f} "
                  f"(sem {r['score_sem_mean']:.2f})  "
                  f"{'TRIVIAL' if t['trivial'] else 'NON-TRIVIAL'}")
    print(f"\nwrote {out_p.name}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
