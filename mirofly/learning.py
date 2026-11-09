"""
v0.4: dopamine-gated preference learning — the mushroom-body layer.

Biology being borrowed (honestly labeled as MB-*inspired* abstraction, not
an MB circuit simulation): Drosophila Kenyon-cell -> MBON synapses are
depressed by dopaminergic gating when outcomes differ from prediction
(reward prediction error). Learning happens on SURPRISE only — expected
events do not rewrite valence.

What citizens learn here:
  1. topic valence  — "this kind of thing is bad/good for me"
  2. source trust   — "this citizen's posts predict bad outcomes for me"

How it plugs in: the affect core stays pure neuroscience (v0.3 spikes on
real wiring). The preference layer sits ABOVE it: events modulate drives by
learned valence; outcomes are read from real-circuit readouts; dopamine is
the core's novelty surprise. Learned citizens react differently to the SAME
wiring over time.
"""

from __future__ import annotations

import numpy as np


class PreferenceLayer:
    def __init__(self, n_citizens: int, topics: list[str], seed: int = 5,
                 lr: float = 0.35, trust_lr: float = 0.25,
                 dopamine_gate: float = 0.05, kappa: float = 0.9,
                 kappa_trust: float = 0.9, outcome_gain: float = 3.0,
                 per_citizen_lr: np.ndarray | None = None):
        self.n = n_citizens
        self.topics = list(topics)
        self.t_idx = {t: i for i, t in enumerate(self.topics)}
        self.lr = lr
        self.trust_lr = trust_lr
        self.gate = dopamine_gate
        self.kappa = kappa
        self.kappa_trust = kappa_trust
        self.outcome_gain = outcome_gain  # outcome readout -> valence scale
        # per-citizen learning rate (0 = naive control citizens)
        self.lrc = (np.ones(n_citizens) if per_citizen_lr is None
                    else np.asarray(per_citizen_lr, dtype=float))
        rng = np.random.default_rng(seed)
        # small individual priors so citizens start slightly different
        self.valence = rng.normal(0, 0.02, size=(n_citizens, len(self.topics)))
        self.trust = np.zeros((n_citizens, n_citizens))
        self.history: list[dict] = []

    # ---------- drive modulation ----------
    def modulate(self, base_drive: np.ndarray, topic_idcs: np.ndarray) -> np.ndarray:
        """Scale each citizen's drive by learned valence of the event topic.

        Positive learned valence amplifies the drive (sensitization),
        negative dampens it (learned discounting). |valence| < 1 keeps the
        sign of the original drive; the layer never inverts a stimulus.
        """
        if topic_idcs is None or len(topic_idcs) == 0:
            return base_drive
        factor = 1.0 + self.kappa * np.clip(self.valence[:, topic_idcs], -1, 1).mean(axis=1)
        return base_drive * np.clip(factor, 0.2, 2.0)

    def modulate_social(self, base_drive: np.ndarray, authors: np.ndarray) -> np.ndarray:
        """Scale drive from a citizen post by learned trust in the author."""
        if authors is None:
            return base_drive
        ar = np.arange(self.n)
        factor = 1.0 + self.kappa_trust * np.clip(self.trust[ar, authors], -1, 1)
        return base_drive * np.clip(factor, 0.2, 2.0)

    # ---------- learning ----------
    def update(self, topic_idcs, outcome: np.ndarray, dopamine: np.ndarray,
               author=None):
        """Dopamine-gated delta rule.

        outcome: per-citizen affective outcome (avoid - approach), in [-1, 1]
        dopamine: per-citizen novelty surprise from the affect core
        Gated: updates happen only where dopamine exceeds the gate —
        expected events do not rewrite valence (the MB rule).
        """
        gate_mask = (dopamine > self.gate) * self.lrc
        if topic_idcs is not None and len(topic_idcs) > 0:
            pred = self.valence[:, topic_idcs].mean(axis=1)
            rpe = np.clip(outcome, -1, 1) - pred
            self.valence[:, topic_idcs] += (self.lr * gate_mask * rpe)[:, None] \
                * np.ones((1, len(topic_idcs)))
        if author is not None:
            pred_t = self.trust[np.arange(self.n), author]
            rpe_t = np.clip(outcome, -1, 1) - pred_t
            self.trust[np.arange(self.n), author] += self.trust_lr * gate_mask * rpe_t

    # ---------- bookkeeping ----------
    def snapshot(self, tick: int) -> dict:
        return {"tick": tick,
                "valence_mean_abs": float(np.abs(self.valence).mean()),
                "trust_mean_abs": float(np.abs(self.trust).mean())}
