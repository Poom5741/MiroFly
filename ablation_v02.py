"""
v0.2 ablation: per-neuron spiking cores — does REAL topology beat rewired?

PRE-REGISTERED PROTOCOL (followed strictly, see commit history):
  1. Build the real circuit.
  2. Tune the global gain G on the INTACT circuit only, using a scripted drive
     timeline, choosing the largest G whose intact response is strong but
     non-saturating. Record the sweep in output.
  3. FREEZE G. Run all three conditions with identical G, identical drive
     timeline, identical per-citizen seeds (5 seeds each):
       intact   — real signed edges
       rewired  — same edges/weights/degrees, postsynaptic targets permuted
                  (configuration-model style: pairing destroyed, degree
                  sequences preserved)
       no-core  — drive ignored, noise matched to intact std
  4. Report mean ± std. Publish whatever it shows.

Drives-only (no social world): the sharpest test of topology, since the
follower-graph feedback loop can mask structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mirofly import build_neuron_core, SpikeCore  # noqa: E402
from mirofly.encoder import encode  # noqa: E402

# same story beats as demo.py: (tick, text)
STORY = [
    (0, "monday meeting about quarterly roadmap"),
    (1, "coffee machine fixed today, nice"),
    (3, "annual health check signup open"),
    (5, "team lunch at the canteen, food was fine"),
    (9, "BREAKING: Miro the office cat was taken to make soup!! โกรธมาก ตาย!!"),
    (10, "re: Miro soup scandal โกงใจมาก"),
    (12, "again: Miro soup ตาย!!"),
    (14, "Miro soup repost โกรธ"),
    (16, "Miro soup repost again"),
    (19, "TWIST: Miro alive!! found in the server room eating banana กล้วยหวาน win"),
    (21, "Miro photos released, อร่อย banana ดีใจ"),
    (23, "office returns to normal, safe ปลอดภัย"),
]
TICKS = 30


def drive_timeline():
    drives = np.zeros(TICKS)
    for t, text in STORY:
        drives[t] += encode(text)
    return drives


def run_brain(circuit, gain, citizen_seed):
    core = SpikeCore(circuit, n_citizens=120, seed=citizen_seed, gain=gain)
    drives = drive_timeline()
    log = []
    for t in range(TICKS):
        out = core.tick(np.full(core.B, drives[t], dtype=np.float32))
        ro = core.readouts(None, None)
        log.append({**{k: float(v.mean()) for k, v in ro.items()},
                    "novelty": float(out["novelty"].mean()),
                    "firing": float(core.sp.mean()),
                    "echo": float(core.rate.mean())})
    return core, log


def metrics(log):
    avoid = np.array([r["avoid"] for r in log])
    app = np.array([r["approach"] for r in log])
    alarm = np.array([r["alarm"] for r in log])
    echo = np.array([r["echo"] for r in log])
    firing = np.array([r["firing"] for r in log])
    return {
        "threat_response": round(float(avoid[9:15].mean()), 4),
        "habituation_drop": round(float(avoid[10:13].mean() - avoid[15:18].mean()), 4),
        "twist_approach": round(float(app[19:22].mean()), 4),
        "alarm_auc": round(float(alarm.sum()), 3),
        "echo_after_storm": round(float(echo[28:].mean()), 4),
        "max_firing_frac": round(float(firing.max()), 4),
    }


def sweep_gain(circuit):
    print("PRE-REGISTRATION: gain sweep on INTACT circuit only")
    print(f"{'gain':>8}{'threat_resp':>14}{'approach_resp':>15}{'max_firing':>12}")
    chosen = None
    for g in [8.0, 16.0, 32.0, 64.0]:
        _, log = run_brain(circuit, g, citizen_seed=0)
        m = metrics(log)
        print(f"{g:>8.1f}{m['threat_response']:>14.4f}{m['twist_approach']:>15.4f}"
              f"{m['max_firing_frac']:>12.4f}")
        # rule: strong but non-saturating (pre-registered)
        if m["threat_response"] > 0.05 and m["max_firing_frac"] < 0.45:
            chosen = g
    if chosen is None:
        chosen = 2.0
    print(f"FROZEN GAIN = {chosen}\n")
    return chosen


def rewire(circuit, seed=2024):
    """Configuration-model control: permute postsynaptic targets.
    Preserves every presynaptic neuron's out-degree, the in-degree multiset,
    and every weight; destroys the real pairing."""
    rng = np.random.default_rng(seed)
    post = circuit["post"].copy()
    rng.shuffle(post)
    c2 = dict(circuit)
    c2["post"] = post
    c2["in_scale"] = np.maximum(
        np.bincount(post, weights=np.abs(circuit["w"]), minlength=circuit["in_scale"].shape[0]),
        1e-9)
    # nz_targets/starts are positions in the sorted-by-post array; after
    # permuting targets we must re-sort and re-derive them
    e = np.argsort(c2["post"], kind="stable")
    c2["post"] = c2["post"][e]
    c2["pre"] = c2["pre"][e]
    c2["w"] = c2["w"][e]
    nz, starts = np.unique(c2["post"], return_index=True)
    c2["nz_targets"], c2["starts"] = nz, starts
    return c2


def main(n_seeds=5):
    circuit = build_neuron_core()
    m = circuit["meta"]
    print(f"circuit: {m['n_neurons']} neurons, {m['n_synapses_in_core']} synapses, "
          f"{m['n_approach']} approach / {m['n_threat']} threat readouts\n")

    G = sweep_gain(circuit)

    rewired = rewire(circuit)

    conds = {
        "intact": circuit,
        "rewired": rewired,
        "no-core": None,  # handled specially
    }
    results = {k: [] for k in conds}
    for name, circ in conds.items():
        for s in range(n_seeds):
            drives = drive_timeline()
            if name == "no-core":
                core = SpikeCore(circuit, n_citizens=120, seed=100 + s, gain=G)
                core.drive_gain_p[:] = 1e-6          # brain ignored
                core.noise_sigma[:] = 0.35           # matched to intact alarm scale
            else:
                core = SpikeCore(circ, n_citizens=120, seed=100 + s, gain=G)
            log = []
            for t in range(TICKS):
                out = core.tick(np.full(core.B, drives[t], dtype=np.float32))
                ro = core.readouts(None, None)
                log.append({**{k: float(v.mean()) for k, v in ro.items()},
                            "echo": float(core.rate.mean()),
                            "firing": float(core.sp.mean()),
                            "novelty": float(out["novelty"].mean())})
            results[name].append(metrics(log))

    keys = list(next(iter(results.values()))[0].keys())
    print("=" * 78)
    print(f"{'metric':<22}", end="")
    for name in conds:
        print(f"{name + ' (mean±std)':>20}", end="")
    print()
    print("-" * 78)
    for k in keys:
        print(f"{k:<22}", end="")
        for name in conds:
            vals = np.array([r[k] for r in results[name]], dtype=float)
            print(f"{vals.mean():>11.4f}±{vals.std():.3f}", end="")
        print()
    print("=" * 78)
    print("\nHonesty note: whatever this table shows goes into the README as-is.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path("results"); out.mkdir(exist_ok=True)
    keys_short = ["threat_resp", "twist_app", "alarm_auc", "echo"]
    keys_map = {"threat_resp": "threat_response", "twist_app": "twist_approach",
                "alarm_auc": "alarm_auc", "echo": "echo_after_storm"}
    xs = np.arange(len(keys_short))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    w = 0.26
    cols = {"intact": "seagreen", "rewired": "orange", "no-core": "gray"}
    for i, name in enumerate(["intact", "rewired", "no-core"]):
        vals = np.array([[r[keys_map[k]] for r in results[name]] for k in keys_short],
                        dtype=float)
        ax.bar(xs + (i - 1) * w, vals.mean(1), w, yerr=vals.std(1),
               label=name, color=cols[name], capsize=3)
    ax.set_xticks(xs); ax.set_xticklabels(keys_short)
    ax.set_title("v0.2 per-neuron spiking cores — 5 seeds each, frozen gain")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "ablation_v02.png", dpi=130)
    print(f"saved {out/'ablation_v02.png'}")


if __name__ == "__main__":
    main()
