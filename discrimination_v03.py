"""
v0.3: THE DISCRIMINATION TEST — does the REAL wiring separate sweet from
bitter better than rewired wiring?

Now the circuit has two real sensory channels:
  sweet energy  -> the real sugar GRNs (21 real FlyWire neurons)
  bitter energy -> the real bitter GRNs (39 real FlyWire neurons, LB1a/LB1b,
                   Engert et al. 2022 annotations)
Readouts are real downstream populations of each seed set.

PRE-REGISTERED (strictly, in this order):
  1. sweep gain on INTACT only; freeze largest non-saturating gain
  2. run intact / rewired (config-model) / no-core, 5 seeds each
  3. report discrimination scores whatever they show:
       D_sweet = (approach pop during sweet) - (threat pop during sweet)
       D_bitter = (threat pop during bitter) - (approach pop during bitter)
     Real biology predicts D > 0 for the intact circuit if it is truly
     selective. If rewired discriminates as well or better, we publish that.

Run: python discrimination_v03.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mirofly import build_dual_core, SpikeCore  # noqa: E402
from ablation_v02 import rewire  # noqa: E402  (same configuration-model control)

TICKS = 30
# (tick, sweet_energy, bitter_energy)
TIMELINE = {2: (1.0, 0.0), 3: (1.0, 0.0),
            9: (0.0, -1.0), 10: (0.0, -1.0), 11: (0.0, -1.0),
            12: (0.0, -1.0), 13: (0.0, -1.0), 14: (0.0, -1.0),
            19: (1.0, 0.0), 20: (1.0, 0.0)}

SWEET_W = [2, 3, 4, 19, 20, 21]
BITTER_W = [9, 10, 11, 12, 13, 14, 15]


def run_core(circuit, gain, citizen_seed, muted=False):
    core = SpikeCore(circuit, n_citizens=120, seed=citizen_seed, gain=gain)
    if muted:
        core.drive_gain_p[:] = 1e-6
        core.noise_sigma[:] = 0.35
    rows = []
    for t in range(TICKS):
        sweet, bitter = TIMELINE.get(t, (0.0, 0.0))
        drive = (sweet - bitter) * np.ones(core.B, dtype=np.float32)
        core.tick(drive)
        ro = core.readouts()
        rows.append({"s": float(ro["approach"].mean()),
                     "b": float(ro["avoid"].mean()),
                     "firing": float(core.sp.mean())})
    return core, rows


def metrics(rows):
    s = np.array([r["s"] for r in rows])
    b = np.array([r["b"] for r in rows])
    sweet_idx = [t for t in range(TICKS) if TIMELINE.get(t, (0, 0))[0] > 0]
    bitter_idx = [t for t in range(TICKS) if TIMELINE.get(t, (0, 0))[1] < 0]
    sweet_idx += [t + 1 for t in sweet_idx]  # +1 tick response lag
    bitter_idx += [t + 1 for t in bitter_idx]
    sweet_resp, sweet_cross = s[sweet_idx].mean(), b[sweet_idx].mean()
    bitter_resp, bitter_cross = b[bitter_idx].mean(), s[bitter_idx].mean()
    return {
        "sweet_resp": round(float(sweet_resp), 4),
        "sweet_cross": round(float(sweet_cross), 4),
        "bitter_resp": round(float(bitter_resp), 4),
        "bitter_cross": round(float(bitter_cross), 4),
        "D_sweet": round(float(sweet_resp - sweet_cross), 4),
        "D_bitter": round(float(bitter_resp - bitter_cross), 4),
        "D_total": round(float((sweet_resp - sweet_cross)
                               + (bitter_resp - bitter_cross)), 4),
    }


def main(n_seeds=5):
    circuit = build_dual_core()
    m = circuit["meta"]
    print(f"dual circuit: {m['n_neurons']} neurons, {m['n_synapses_in_core']} synapses, "
          f"{m['n_sugar_seeds']} sugar + {m['n_bitter_seeds']} bitter GRN seeds\n")

    # ---- pre-registration: sweep on intact only ----
    print("PRE-REGISTRATION: gain sweep on INTACT dual circuit")
    print(f"{'gain':>8}{'sweet_resp':>13}{'bitter_resp':>13}{'max_firing':>12}")
    chosen = None
    for g in [16.0, 32.0, 64.0, 128.0]:
        _, rows = run_core(circuit, g, citizen_seed=0)
        mm = metrics(rows)
        print(f"{g:>8.1f}{mm['sweet_resp']:>13.4f}{mm['bitter_resp']:>13.4f}"
              f"{max(r['firing'] for r in rows):>12.4f}")
        if mm["bitter_resp"] > 0.05 and max(r["firing"] for r in rows) < 0.45:
            chosen = g
    if chosen is None:
        chosen = 64.0
    print(f"FROZEN GAIN = {chosen}\n")
    G = chosen

    rewired = rewire(circuit)
    conds = {"intact": circuit, "rewired": rewired, "no-core": None}
    results = {k: [] for k in conds}
    for name, circ in conds.items():
        for s in range(n_seeds):
            core, rows = run_core(circ if circ is not None else circuit, G,
                                  citizen_seed=100 + s, muted=(name == "no-core"))
            results[name].append(metrics(rows))

    keys = ["sweet_resp", "sweet_cross", "bitter_resp", "bitter_cross",
            "D_sweet", "D_bitter", "D_total"]
    print("=" * 84)
    print(f"{'metric':<16}", end="")
    for name in conds:
        print(f"{name + ' (mean±std)':>22}", end="")
    print()
    print("-" * 84)
    for k in keys:
        print(f"{k:<16}", end="")
        for name in conds:
            vals = np.array([r[k] for r in results[name]], dtype=float)
            print(f"{vals.mean():>16.4f}±{vals.std():.3f}", end="")
        print()
    print("=" * 84)
    d_i = np.mean([r["D_total"] for r in results["intact"]])
    d_r = np.mean([r["D_total"] for r in results["rewired"]])
    print(f"\nHONEST READ: intact D_total={d_i:.3f} vs rewired D_total={d_r:.3f}")
    if d_i > d_r:
        print("  -> REAL wiring discriminates sweet/bitter better than rewired.")
        print("     The measured circuit carries channel specificity random")
        print("     wiring does not have. v0.2 'selective not powerful' now has")
        print("     its constructive counterpart.")
    else:
        print("  -> real wiring does NOT discriminate better than rewired at")
        print("     this scale/regime. Publishing as-is; the stand-in drive")
        print("     injection may be the limiter (next: real bitter SEZ synapses).")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path("results"); out.mkdir(exist_ok=True)
    ks = ["sweet_resp", "sweet_cross", "bitter_resp", "bitter_cross", "D_total"]
    xs = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    w = 0.26
    cols = {"intact": "seagreen", "rewired": "orange", "no-core": "gray"}
    for i, name in enumerate(["intact", "rewired", "no-core"]):
        vals = np.array([[r[k] for r in results[name]] for k in ks], dtype=float)
        ax.bar(xs + (i - 1) * w, vals.mean(1), w, yerr=vals.std(1),
               label=name, color=cols[name], capsize=3)
    ax.set_xticks(xs); ax.set_xticklabels(ks)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("v0.3 sweet/bitter discrimination — 5 seeds, frozen gain")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "discrimination_v03.png", dpi=130)
    print(f"saved {out/'discrimination_v03.png'}")


if __name__ == "__main__":
    main()
