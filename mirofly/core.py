"""
MiroFly affect core: batched population dynamics on REAL FlyWire wiring.

One batch element = one citizen. The batch dimension of the repo's PyTorch
runner (originally: independent trials) is reinterpreted as citizens sharing
the same real wiring, personalized by gains + their own adaptation state.

This is a rate-model v0.1 on aggregated REAL signed connectivity:
  r_{t+1} = tanh( gain * ( (r_t * adapt) @ W^T  +  drive_gain * drive + noise ) )
  adapt decays where a drive persists (habituation), recovers otherwise.
  dopamine = positive novelty surprise (drive above its EMA) -> memory tag.

Upgrade path (see README): per-neuron spiking microcircuits.
"""

from __future__ import annotations

import numpy as np


class AffectCore:
    def __init__(self, W, drive_g, n_citizens: int, seed: int = 1234,
                 gain: float = 1.0, drive_gain: float = 2.5,
                 habituation_eta: float = 0.045, recovery_eta: float = 0.01,
                 avoid_g: list[int] | None = None):
        self.W = np.asarray(W, dtype=float)
        self.G = self.W.shape[0]
        self.drive_g = drive_g
        self.B = n_citizens
        self.gain = gain
        self.drive_gain = drive_gain
        self.habituation_eta = habituation_eta
        self.recovery_eta = recovery_eta
        rng = np.random.default_rng(seed)
        # personalization: each citizen has its own drive sensitivity & noise level
        self.drive_gain_p = drive_gain * rng.uniform(0.7, 1.5, size=(self.B, 1))
        self.noise_sigma = rng.uniform(0.01, 0.05, size=(self.B, 1))
        self.rng = np.random.default_rng(seed + 1)
        self.avoid_g: list[int] = list(avoid_g or [])   # real taste-inhibited groups
        # persistent state
        self.r = np.zeros((self.B, self.G))          # group rates
        self.adapt = np.ones((self.B, self.G))       # 1 = fresh, ->0.2 = burnt out
        self.drive_ema = np.zeros((self.B, 1))       # novelty reference

    def set_channels(self, avoid_g: list[int]):
        """Register the real taste-inhibited groups as the threat channel."""
        self.avoid_g = list(avoid_g)

    def tick(self, drives: np.ndarray) -> dict:
        """drives: (B,) signed scalar.

        drives > 0: appetitive energy into the REAL taste-input group
        (sugar GRN population).
        drives < 0: aversive energy into the REAL taste-inhibited groups
        (the populations the taste channel genuinely suppresses — v0.1
        stand-in for the bitter channel; roadmap: real bitter GRN IDs).
        """
        d = np.asarray(drives, dtype=float).reshape(self.B, 1)
        pos = np.clip(d, 0.0, None)      # sweet energy
        neg = np.clip(-d, 0.0, None)     # threat energy
        noise = self.rng.normal(0.0, self.noise_sigma, size=(self.B, self.G))
        recurrent = (self.r * self.adapt) @ self.W.T
        stim = noise.copy()
        # input-synapse depression: the drive passes through the channel's own
        # adaptation state (the classic mechanism of habituation)
        stim[:, self.drive_g] += (self.drive_gain_p[:, 0] * pos[:, 0]
                                  * self.adapt[:, self.drive_g])
        if self.avoid_g:
            stim[:, self.avoid_g] += (self.drive_gain_p[:, 0, None] * neg[:, 0, None]
                                      * self.adapt[:, self.avoid_g])
        v = recurrent + stim
        self.r = np.tanh(v)

        # habituation on whichever channel was driven; recovery when quiet
        self.adapt[:, self.drive_g] = np.clip(
            self.adapt[:, self.drive_g]
            - self.habituation_eta * pos[:, 0] + self.recovery_eta * (pos[:, 0] < 0.05),
            0.2, 1.0)
        if self.avoid_g:
            self.adapt[:, self.avoid_g] = np.clip(
                self.adapt[:, self.avoid_g]
                - self.habituation_eta * neg[:, 0, None]
                + self.recovery_eta * (neg[:, 0] < 0.05)[:, None],
                0.2, 1.0)

        # dopamine = positive surprise on total drive energy (RPE proxy)
        energy = np.abs(d[:, 0])
        novelty = np.clip(energy - self.drive_ema[:, 0], 0.0, None)
        self.drive_ema[:, 0] = 0.85 * self.drive_ema[:, 0] + 0.15 * energy

        # novelty readout returned here; approach/avoid/alarm need wiring-aware
        # groups, so callers use readouts() after tick().
        return {"novelty": novelty, "energy": energy}

    def readouts(self, approach_g: list[int], avoid_g: list[int]) -> dict:
        """Population rates of REAL wiring-selected groups.

        approach_g: groups the taste channel excites (real positive coupling)
        avoid_g: groups the taste channel inhibits — they fire when the
        aversive channel drives them (disinhibition-flavored, real neurons).
        """
        approach = self.r[:, approach_g].mean(axis=1)
        avoid = self.r[:, avoid_g].mean(axis=1) if self.avoid_g else \
            np.zeros(self.B)
        a = np.clip(avoid - approach * 0.5, 0.0, None)
        alarm_ema = getattr(self, "_alarm", None)
        self._alarm = a if alarm_ema is None else 0.7 * alarm_ema + 0.3 * a
        return {
            "approach": approach,
            "avoid": avoid,
            "alarm": self._alarm.copy(),
        }
