"""v0.3 regression test: the dual-channel circuit discriminates.

Frozen protocol from ablation_v03 (gain 128, 5-seed mean D_sweet=+0.029,
D_bitter=-0.005, rewired D_total=-0.010). This test asserts the intact
circuit's sweet-channel discrimination with generous tolerance — if a code
change breaks channel specificity, we want to know.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mirofly import build_dual_core, SpikeCore  # noqa: E402

TICKS = 30
TIMELINE = {2: (1.0, 0.0), 3: (1.0, 0.0),
            9: (0.0, -1.0), 10: (0.0, -1.0), 11: (0.0, -1.0),
            12: (0.0, -1.0), 13: (0.0, -1.0), 14: (0.0, -1.0),
            19: (1.0, 0.0), 20: (1.0, 0.0)}
FROZEN_GAIN = 128.0


def test_dual_core_builds_from_real_data():
    c = build_dual_core()
    m = c["meta"]
    assert m["n_sugar_seeds"] >= 20 and m["n_bitter_seeds"] >= 35
    assert m["n_synapses_in_core"] > 50000
    print(f"  dual circuit: {m['n_neurons']} neurons, {m['n_synapses_in_core']} synapses")


def test_sweet_discrimination_positive():
    c = build_dual_core()
    core = SpikeCore(c, n_citizens=60, seed=3, gain=FROZEN_GAIN)
    s_hist, b_hist = [], []
    for t in range(TICKS):
        sweet, bitter = TIMELINE.get(t, (0.0, 0.0))
        core.tick((sweet - bitter) * np.ones(core.B, dtype=np.float32))
        ro = core.readouts()
        s_hist.append(float(ro["approach"].mean()))
        b_hist.append(float(ro["avoid"].mean()))
    s, b = np.array(s_hist), np.array(b_hist)
    sweet_w = [2, 3, 4, 19, 20, 21] + [t + 1 for t in [2, 3, 19, 20]]
    D_sweet = float(s[sweet_w].mean() - b[sweet_w].mean())
    assert D_sweet > 0.0, f"sweet discrimination should be positive, got {D_sweet:.4f}"
    print(f"  D_sweet = {D_sweet:+.4f} (frozen gain {FROZEN_GAIN})")


if __name__ == "__main__":
    test_dual_core_builds_from_real_data()
    test_sweet_discrimination_positive()
    print("\nV0.3 TESTS PASSED")
