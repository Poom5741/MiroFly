"""
MiroFly demo: an office-cat scandal propagates through 120 citizens whose
affect cores run on REAL FlyWire wiring. Watch: alarm cascade -> outrage
habituation -> novelty revival on the twist.

Run: python demo.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from mirofly import build_core, AffectCore, World, encode

STORY = [
    (0, "monday meeting about quarterly roadmap"),
    (1, "coffee machine fixed today, nice"),
    (3, "annual health check signup open"),
    (5, "team lunch at the canteen, food was fine"),
    (9,  "BREAKING: Miro the office cat was taken to make soup!! โกรธมาก ตาย!!"),
    (10, "re: Miro soup scandal โกงใจมาก"),
    (12, "again: Miro soup ตาย!!"),
    (14, "Miro soup repost โกรธ"),
    (16, "Miro soup repost again"),
    (19, "TWIST: Miro alive!! found in the server room eating banana กล้วยหวาน win"),
    (21, "Miro photos released, อร่อย banana ดีใจ"),
    (23, "office returns to normal, safe ปลอดภัย"),
]


def run(n_citizens=120, ticks=26, W=None, core=None, seed=99):
    if core is None:
        c = build_core()
        W = c["W"]
        core = AffectCore(W, c["drive_g"], n_citizens, seed=seed,
                          avoid_g=c["meta"]["avoid_groups"])
        meta = c["meta"]
    else:
        meta = core.meta
    world = World(core=core, meta=meta, n_citizens=n_citizens, seed=seed)
    log = []
    print(f"MiroFly demo — {n_citizens} citizens on {meta['n_neurons']}-neuron real cores")
    print(f"(data: {meta['data']})\n")
    for t in range(ticks):
        for txt in [s for tt, s in STORY if tt == t]:
            world.inject_event(t, txt, source="news")
            print(f"  [t{t:02d}] NEWS: {txt}  (drive={encode(txt):+.2f})")
        ro = world.step(t)
        posts = world.maybe_post(t, ro)
        mean_alarm = float(np.mean(ro["alarm"]))
        mean_avoid = float(np.mean(ro["avoid"]))
        mean_app = float(np.mean(ro["approach"]))
        log.append({"tick": t, "posts": len(posts), "alarm": mean_alarm,
                    "avoid": mean_avoid, "approach": mean_app,
                    "novelty": float(np.mean(core.drive_ema[:, 0]))})
        if posts:
            loudest = posts[int(np.argmax([abs(p.drive) for p in posts]))]
            print(f"        {len(posts):3d} posts | alarm={mean_alarm:.3f} "
                  f"avoid={mean_avoid:+.3f} approach={mean_app:+.3f} "
                  f"| loudest([{loudest.author}]): {loudest.text[:52]}")
    return world, log, meta


if __name__ == "__main__":
    world, log, meta = run()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path("results"); out.mkdir(exist_ok=True)
    t = [r["tick"] for r in log]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, [r["alarm"] for r in log], label="alarm", color="crimson", lw=2)
    ax.plot(t, [r["avoid"] for r in log], label="avoid (outrage)", color="orange", lw=2)
    ax.plot(t, [r["approach"] for r in log], label="approach", color="seagreen", lw=2)
    ax.plot(t, [r["posts"] / 20 for r in log], label="posts/20", color="gray", lw=1)
    for tt, s in STORY:
        if "BREAKING" in s or "TWIST" in s:
            ax.axvline(tt, color="k", ls=":", alpha=0.5)
    ax.legend(); ax.set_xlabel("tick"); ax.set_ylabel("population mean")
    ax.set_title("MiroFly: an office-cat scandal on real FlyWire wiring")
    fig.tight_layout(); fig.savefig(out / "demo_timeline.png", dpi=130)
    print(f"\nsaved {out/'demo_timeline.png'}")
