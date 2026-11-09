"""
Ablation: does the REAL wiring matter?

Conditions (same event stream, same seeds):
  1. intact      — real FlyWire signed wiring
  2. shuffled    — same weights, rewired (destroys structure, keeps distribution)
  3. no-core     — affect replaced by matched random noise

This is the project's honesty engine: whatever the answer, we publish it.
Run: python ablation.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from mirofly import build_core, AffectCore
from demo import run as run_story

SCANDAL_TICK = 9
RESPONSE_WINDOW = (9, 14)


def summarize(log):
    alarm = np.array([r["alarm"] for r in log])
    avoid = np.array([r["avoid"] for r in log])
    posts = np.array([r["posts"] for r in log])
    peak_i = int(avoid.argmax())
    peak = float(avoid[peak_i])
    half = None
    if peak > 1e-6:
        after = avoid[peak_i:]
        below = np.where(after < peak / 2)[0]
        half = int(below[0]) if len(below) else None
    return {
        "peak_outrage": round(peak, 3),
        "outrage_halflife_ticks": half,
        "total_posts": int(posts.sum()),
        "mean_alarm": round(float(alarm.mean()), 3),
    }


def main(n_citizens=120):
    base = build_core()
    W = base["W"]
    rng = np.random.default_rng(42)

    print("=" * 72)
    print("CONDITION 1/3: intact real wiring")
    print("=" * 72)
    core = AffectCore(W, base["drive_g"], n_citizens, seed=7,
                      avoid_g=base["meta"]["avoid_groups"])
    core.meta = base["meta"]
    _, log_i, _ = run_story(n_citizens=n_citizens, W=W, core=core)
    m_intact = summarize(log_i)

    print("\n" + "=" * 72)
    print("CONDITION 2/3: shuffled wiring (same weights, rewired)")
    print("=" * 72)
    W_shuf = rng.permutation(W.flatten()).reshape(W.shape)
    core = AffectCore(W_shuf, base["drive_g"], n_citizens, seed=7,
                      avoid_g=base["meta"]["avoid_groups"])
    core.meta = base["meta"]
    _, log_s, _ = run_story(n_citizens=n_citizens, W=W_shuf, core=core)
    m_shuf = summarize(log_s)

    print("\n" + "=" * 72)
    print("CONDITION 3/3: no-core (matched random noise affect)")
    print("=" * 72)
    # matched: readouts are noise with same std as intact run
    std_i = max(float(np.std([r["avoid"] for r in log_i])), 0.15)
    core = AffectCore(W * 0.0, base["drive_g"], n_citizens, seed=7,
                      avoid_g=base["meta"]["avoid_groups"])
    core.meta = base["meta"]
    core.noise_sigma = np.full((n_citizens, 1), std_i)
    core.drive_gain_p = np.full((n_citizens, 1), 1e-6)  # brain ignored
    _, log_n, _ = run_story(n_citizens=n_citizens, W=np.zeros_like(W), core=core)
    m_none = summarize(log_n)

    print("\n" + "=" * 72)
    print(f"{'metric':<26}{'intact':>12}{'shuffled':>12}{'no-core':>12}")
    print("-" * 72)
    for k in m_intact:
        v1, v2, v3 = m_intact[k], m_shuf[k], m_none[k]
        f = lambda v: str(v)
        print(f"{k:<26}{f(v1):>12}{f(v2):>12}{f(v3):>12}")
    print("=" * 72)
    print("\nInterpretation: if intact ≈ shuffled, the real topology isn't")
    print("contributing dynamics yet — that is the honest headline until it does.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path("results"); out.mkdir(exist_ok=True)
    keys = list(m_intact.keys())
    xs = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.27
    def norm(vals):
        v = np.array([float(x) if x is not None else 0.0 for x in vals], dtype=float)
        mx = max(np.abs(v).max(), 1e-9)
        return v / mx
    ax.bar(xs - w, norm([m_intact[k] for k in keys]), w, label="intact", color="seagreen")
    ax.bar(xs, norm([m_shuf[k] for k in keys]), w, label="shuffled", color="orange")
    ax.bar(xs + w, norm([m_none[k] for k in keys]), w, label="no-core", color="gray")
    ax.set_xticks(xs); ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_title("Ablation (normalized per metric) — real wiring vs controls")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=130)
    print(f"saved {out/'ablation.png'}")


if __name__ == "__main__":
    main()
