"""Sanity tests: the core must behave like a nervous system, not noise."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mirofly import build_core, AffectCore, encode  # noqa: E402


def test_build_core_real_data():
    c = build_core()
    m = c["meta"]
    assert m["n_neurons"] > 100
    assert m["n_synapses_in_core"] > 500
    assert m["drive_group"] >= 0
    print(f"  core: {m['n_neurons']} neurons, {m['n_synapses_in_core']} synapses, "
          f"{m['n_groups']} groups")


def test_threat_causes_alarm_not_sweetness():
    c = build_core()
    core = AffectCore(c["W"], c["drive_g"], n_citizens=40, seed=5, avoid_g=c["meta"]["avoid_groups"])
    # warm-up silence
    for _ in range(5):
        core.tick(np.zeros(40))
    ro0 = core.readouts(c["approach_g"], c["avoid_g"])
    # sustained threat
    for _ in range(8):
        core.tick(np.full(40, -0.9))
    ro1 = core.readouts(c["approach_g"], c["avoid_g"])
    assert ro1["avoid"].mean() > ro0["avoid"].mean(), "threat should raise avoid"
    print(f"  threat: avoid {ro0['avoid'].mean():+.3f} -> {ro1['avoid'].mean():+.3f}")


def test_habituation_repeats_decay():
    c = build_core()
    core = AffectCore(c["W"], c["drive_g"], n_citizens=40, seed=5, avoid_g=c["meta"]["avoid_groups"])
    core.tick(np.zeros(40))
    first = None
    last = None
    for i in range(12):
        ro = core.readouts(c["approach_g"], c["avoid_g"])
        core.tick(np.full(40, -0.9))
        resp = np.abs(ro["avoid"]).mean()
        if i == 1:
            first = resp
        last = resp
    assert last < first, f"repeated identical stimulus must habituate ({first:.3f} -> {last:.3f})"
    print(f"  habituation: response {first:.3f} -> {last:.3f} over repeated stimulus")


def test_novelty_spikes_on_new_stimulus():
    c = build_core()
    core = AffectCore(c["W"], c["drive_g"], n_citizens=10, seed=5, avoid_g=c["meta"]["avoid_groups"])
    core.tick(np.zeros(10))
    for _ in range(6):
        out = core.tick(np.full(10, 0.8))
    late_novelty = out["novelty"].mean()
    out = core.tick(np.zeros(10))
    quiet = out["novelty"].mean()
    out = core.tick(np.full(10, -0.9))
    new_novelty = out["novelty"].mean()
    assert new_novelty > late_novelty, "a new stimulus should surprise more than a repeated one"
    assert quiet < new_novelty
    print(f"  novelty: repeated={late_novelty:.3f} quiet={quiet:.3f} new={new_novelty:.3f}")


def test_encoder():
    assert encode("โกรธมาก ตายหมดแล้ว") < -0.3
    assert encode("อร่อย หวาน ฟิน love it") > 0.3
    assert abs(encode("the weather today")) < 0.05
    print("  encoder: Thai+English lexicon OK")


if __name__ == "__main__":
    for fn in [test_build_core_real_data, test_encoder,
               test_threat_causes_alarm_not_sweetness,
               test_habituation_repeats_decay,
               test_novelty_spikes_on_new_stimulus]:
        print(f"RUN {fn.__name__}")
        fn()
    print("\nALL TESTS PASSED")
