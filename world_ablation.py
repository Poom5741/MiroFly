"""
v0.4b: WORLD-LEVEL ablation — real per-neuron cores inside the full
opinion-dynamics world (follower graph, posting cascade).

Question (same as v0.1, now with v0.3 dual-channel spiking cores):
does the REAL wiring change what the *society* does — not just what the
brain does? Learning is OFF everywhere, to isolate topology.

Conditions (3 seeds each, identical story/initialization):
  intact   — real FlyWire signed wiring
  rewired  — configuration-model (degrees + weights preserved, pairing destroyed)
  no-core  — drive ignored, noise matched

Metrics: society-level (posts, outrage peak/halflife, alarm) plus the
channel-specificity metric that matters since v0.3: approach-channel
contamination during the scandal window (does the scandal spill into the
sweet channel? real wiring should contain it better).

Run: python world_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mirofly import build_dual_core, SpikeCore, World  # noqa: E402
from ablation_v02 import rewire  # noqa: E402
from demo_learning import STORY, N_LEARN, N_NAIVE  # noqa: E402

TICKS = 24
SCANDAL_WINDOW = [4, 5, 6, 7, 8]           # first scandal wave (+1 lag covered)
SCANDAL_TOPICS = {"miro_soup"}


def run_world(circuit, citizen_seed, muted=False, n_citizens=120):
    core = SpikeCore(circuit, n_citizens=n_citizens, seed=citizen_seed, gain=128.0)
    if muted:
        core.drive_gain_p[:] = 1e-6
        core.noise_sigma[:] = 0.35
    world = World(core=core, meta=circuit["meta"], pref=None,
                  n_citizens=n_citizens, seed=99, readout_gain=6.0)
    avoid_peak = 0.0
    peak_t = 0
    scandal_approach, scandal_avoid, scandal_n = [], [], []
    responders = set()
    total_posts = 0
    posts_per_tick = []
    for t in range(TICKS):
        for _, topic, text in [(tt, tp, tx) for tt, tp, tx in STORY if tt == t]:
            world.inject_event(t, text, topic=topic)
        ro = world.step(t)
        posts = world.maybe_post(t, ro, ro["drives"])
        total_posts += len(posts)
        posts_per_tick.append(len(posts))
        av = float(ro["avoid"].mean())
        ap = float(ro["approach"].mean())
        if av > avoid_peak:
            avoid_peak, peak_t = av, t
        if t in SCANDAL_WINDOW or t - 1 in SCANDAL_WINDOW:
            scandal_approach.append(ap)
            scandal_avoid.append(av)
        if t in SCANDAL_WINDOW:
            hot = np.where(ro["avoid"] > ro["avoid"].mean() + 0.03)[0]
            responders.update(hot.tolist())
    # half-life of outrage after peak
    avoid_series = []
    return {
        "total_posts": total_posts,
        "peak_outrage": round(avoid_peak, 4),
        "peak_tick": peak_t,
        "scandal_approach_spill": round(float(np.mean(scandal_approach)), 4),
        "scandal_avoid": round(float(np.mean(scandal_avoid)), 4),
        "containment": round(float(np.mean(scandal_avoid))
                             - float(np.mean(scandal_approach)), 4),
        "scandal_responders": len(responders),
        "max_posts_tick": max(posts_per_tick),
    }


def main(n_seeds=3):
    circuit = build_dual_core()
    rewired = rewire(circuit)
    conds = {"intact": circuit, "rewired": rewired, "no-core": None}
    results = {k: [] for k in conds}
    for name, circ in conds.items():
        for s in range(n_seeds):
            m = run_world(circ if circ is not None else circuit,
                          citizen_seed=200 + s, muted=(name == "no-core"))
            results[name].append(m)
            print(f"  [{name} seed {s}] {m}")

    keys = list(next(iter(results.values()))[0].keys())
    print("\n" + "=" * 86)
    print(f"{'metric':<24}", end="")
    for name in conds:
        print(f"{name + ' (mean±std)':>21}", end="")
    print()
    print("-" * 86)
    for k in keys:
        print(f"{k:<24}", end="")
        for name in conds:
            vals = np.array([r[k] for r in results[name]], dtype=float)
            print(f"{vals.mean():>14.3f}±{vals.std():.2f}", end="")
        print()
    print("=" * 86)
    ci = np.mean([r["containment"] for r in results["intact"]])
    cr = np.mean([r["containment"] for r in results["rewired"]])
    print(f"\ncontainment (avoid-spill gap during scandal): intact {ci:.3f} "
          f"vs rewired {cr:.3f}")
    print("positive = the society's outrage stays in the threat channel;")
    print("low/negative = scandal leaks into the sweet channel (confusion).")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path("results"); out.mkdir(exist_ok=True)
    ks = ["total_posts", "peak_outrage", "scandal_avoid",
          "scandal_approach_spill", "containment", "scandal_responders"]
    xs = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    w = 0.26
    cols = {"intact": "seagreen", "rewired": "orange", "no-core": "gray"}
    for i, name in enumerate(["intact", "rewired", "no-core"]):
        vals = np.array([[r[k] for r in results[name]] for k in ks], dtype=float)
        mx = max(np.abs(vals).max(), 1e-9)
        ax.bar(xs + (i - 1) * w, vals.mean(1) / mx, w, yerr=vals.std(1) / mx,
               label=name, color=cols[name], capsize=3)
    ax.set_xticks(xs); ax.set_xticklabels(ks, rotation=15, ha="right")
    ax.set_title("v0.4b world-level ablation — v0.3 spiking cores, 3 seeds")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "world_ablation.png", dpi=130)
    print(f"saved {out/'world_ablation.png'}")


if __name__ == "__main__":
    main()
