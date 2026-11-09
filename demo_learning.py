"""
v0.4 demo: citizens LEARN from events (dopamine-gated MB rule).

Setup: 120 citizens share the same real v0.3 dual-channel circuit.
  90 learners  (per-citizen learning rate 1.0)
  30 naive     (learning rate 0 — matched controls in the SAME world)

Storyline: mundane chatter -> scandal wave (miro_soup x4) -> banana twist.
Predictions if the MB rule works:
  1. learners' valence["miro_soup"] goes negative -> they DISCOUNT repeats
     (cry-wolf) relative to naive citizens
  2. valence["banana_win"] goes positive -> approach amplification
  3. mundane topics stay ~0 under the dopamine gate (gating = selectivity)
  4. followers' trust in the loudest scandal-poster declines

Also runs the gating ablation: gated vs ungated vs off.

Run: python demo_learning.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mirofly import build_dual_core, SpikeCore, PreferenceLayer, World

N_LEARN, N_NAIVE = 90, 30
GAIN = 128.0  # frozen from v0.3 pre-registration

STORY = [  # (tick, topic, text)
    (1, "office", "quarterly roadmap meeting notes"),
    (2, "office", "printer fixed, carry on"),
    (4, "miro_soup", "BREAKING: Miro the office cat was taken to make soup!! โกรธมาก ตาย!!"),
    (5, "miro_soup", "Miro soup scandal spreading โกงใจ"),
    (6, "miro_soup", "Miro soup repost โกรธ"),
    (7, "miro_soup", "Miro soup still trending ตาย"),
    (10, "office", "budget review scheduled"),
    (13, "banana_win", "TWIST: Miro alive!! eating banana in the server room กล้วยหวาน win อร่อย"),
    (14, "banana_win", "Miro banana photos released ดีใจฟิน"),
    (16, "office", "facilities survey email"),
    (19, "miro_soup", "Miro soup meme revival ตาย โกรธ"),
    (20, "miro_soup", "someone reposts Miro soup again"),
    (23, "banana_win", "office adopts banana corner for Miro หวานขอบคุณ"),
]


def build_world(gate: float = 0.05, learn: bool = True, citizen_seed: int = 100):
    circuit = build_dual_core()
    core = SpikeCore(circuit, n_citizens=N_LEARN + N_NAIVE, seed=7, gain=GAIN)
    lrc = np.concatenate([np.ones(N_LEARN), np.zeros(N_NAIVE)])
    pref = PreferenceLayer(
        N_LEARN + N_NAIVE, topics=["office", "miro_soup", "banana_win"],
        seed=5, per_citizen_lr=lrc, dopamine_gate=gate, kappa=2.5)
    if not learn:
        pref.lrc[:] = 0.0
    world = World(core=core, meta=circuit["meta"], pref=pref,
                  n_citizens=N_LEARN + N_NAIVE, seed=99, readout_gain=6.0)
    return world, pref


def run_story(world, ticks=24):
    log = []
    for t in range(ticks):
        for _, topic, text in [(tt, tp, tx) for tt, tp, tx in STORY if tt == t]:
            world.inject_event(t, text, topic=topic)
        ro = world.step(t)
        posts = world.maybe_post(t, ro, ro["drives"])
        learners = slice(0, N_LEARN)
        naive = slice(N_LEARN, N_LEARN + N_NAIVE)
        log.append({
            "tick": t, "posts": len(posts),
            "approach_L": float(ro["approach"][learners].mean()),
            "approach_N": float(ro["approach"][naive].mean()),
            "avoid_L": float(ro["avoid"][learners].mean()),
            "avoid_N": float(ro["avoid"][naive].mean()),
        })
    return world, log, posts


def reaction_to(log, ticks, which="avoid"):
    key = f"{which}_L"
    keyn = f"{which}_N"
    idx = [t for t, _ in enumerate(log) if t in ticks]
    idx += [t + 1 for t in idx if t + 1 < len(log)]
    return (np.mean([log[i][key] for i in idx]),
            np.mean([log[i][keyn] for i in idx]))


def main():
    world, pref, posts = None, None, None
    world, pref = build_world()
    world2, log, posts = run_story(world)
    pref = world2.pref

    print("=" * 76)
    print("LEARNED VALENCE (learners mean; naive structurally 0)")
    print("-" * 76)
    for i, topic in enumerate(pref.topics):
        v = pref.valence[:N_LEARN, i]
        print(f"  {topic:<14} {v.mean():+.3f} ± {v.std():.3f}")
    print()

    # reaction contrasts
    a1L, a1N = reaction_to(log, [5, 6, 7], which="avoid")   # first scandal wave
    a2L, a2N = reaction_to(log, [20, 21], which="avoid")    # revival scandal
    apL, apN = reaction_to(log, [14, 15, 24], which="approach")  # banana
    print("=" * 76)
    print("REACTION CONTRASTS (learners vs naive, same world, same wiring)")
    print("-" * 76)
    print(f"  scandal wave 1 avoid : learners {a1L:+.3f} | naive {a1N:+.3f}")
    print(f"  scandal revival avoid: learners {a2L:+.3f} | naive {a2N:+.3f}"
          f"   (learned discounting = cry-wolf if L < N here)")
    print(f"  banana approach      : learners {apL:+.3f} | naive {apN:+.3f}"
          f"   (learned amplification if L > N)")

    # trust: loudest scandal poster
    scandal_posts = [e for e in world2.feed
                     if e.source == "citizen" and e.topic == "miro_soup"]
    if scandal_posts:
        authors = np.array([e.author for e in scandal_posts])
        loudest = int(np.bincount(authors).argmax())
        followers = [i for i in range(world2.n_citizens)
                     if loudest in world2.follows[i]]
        tr = pref.trust[followers, loudest].mean()
        print(f"  loudest scandal poster: citizen {loudest} "
              f"({len(scandal_posts)} posts), followers' trust = {tr:+.3f}")

    # ---- gating ablation ----
    print("\n" + "=" * 76)
    print("GATING ABLATION (end valence of scandal vs mundane topic)")
    print("=" * 76)
    print(f"{'condition':<12}{'val(miro_soup)':>16}{'val(office)':>14}"
          f"{'sum|val| mundane':>20}")
    for name, kwargs in [("gated", dict(gate=0.05)),
                         ("ungated", dict(gate=-1.0)),
                         ("off", dict(learn=False))]:
        w, p = build_world(**kwargs)
        w2, _, _ = run_story(w)
        p = w2.pref  # layer always exists; "off" has lrc=0
        vi = p.valence[:N_LEARN, p.t_idx["miro_soup"]].mean()
        vo = p.valence[:N_LEARN, p.t_idx["office"]].mean()
        mundane = np.abs(p.valence[:N_LEARN, p.t_idx["office"]]).sum()
        print(f"{name:<12}{vi:>16.4f}{vo:>14.4f}{mundane:>20.4f}")
    print("(gating prediction: scandal goes negative everywhere, office stays")
    print(" ~0 only when gated; ungated drifts, off never learns)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path("results"); out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # valence trajectories: re-run with tracking
    w3, p3 = build_world()
    val_track = {t: [] for t in p3.topics}
    for t in range(24):
        for _, topic, text in [(tt, tp, tx) for tt, tp, tx in STORY if tt == t]:
            w3.inject_event(t, text, topic=topic)
        ro3 = w3.step(t)
        w3.maybe_post(t, ro3, ro3["drives"])
        for topic in val_track:
            val_track[topic].append(
                float(p3.valence[:N_LEARN, p3.t_idx[topic]].mean()))
    for topic, color in [("miro_soup", "crimson"), ("banana_win", "seagreen"),
                         ("office", "gray")]:
        axes[0].plot(val_track[topic], label=topic, color=color, lw=2)
    axes[0].set_title("learned valence (dopamine-gated)")
    axes[0].set_xlabel("tick"); axes[0].legend(); axes[0].axhline(0, color="k", lw=0.5)
    labels = ["scandal1\navoid", "revival\navoid", "banana\napproach"]
    vals = [[a1L, a1N], [a2L, a2N], [apL, apN]]
    xs = np.arange(3); w = 0.35
    axes[1].bar(xs - w / 2, [v[0] for v in vals], w, label="learners", color="seagreen")
    axes[1].bar(xs + w / 2, [v[1] for v in vals], w, label="naive", color="gray")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(labels)
    axes[1].set_title("reaction contrast (same world, same wiring)")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "learning_v04.png", dpi=130)
    print(f"\nsaved {out/'learning_v04.png'}")


if __name__ == "__main__":
    main()
