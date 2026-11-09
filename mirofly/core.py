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


class SpikeCore:
    """v0.2: batched per-neuron dynamics on the real circuit (no aggregation).

    One batch element = one citizen with its own adaptation + memory state.
    Recurrent propagation runs through the REAL per-neuron signed synapses;
    citizens share wiring, differ in drive gains, noise, and history.
    Batched sparse matvec: sort edges by target once, then
      current[:, nz_targets] = reduceat(spikes[:, pre] * w, starts, axis=1)
    """

    def __init__(self, circuit, n_citizens: int, seed: int = 1234,
                 gain: float = 2.0, tau: float = 6.0, threshold: float = 1.0,
                 refractory: int = 2, habituation_eta: float = 0.05,
                 recovery_eta: float = 0.015, noise_sigma: float = 0.02,
                 channels: dict | None = None):
        self.circ = circuit
        self.B = n_citizens
        self.N = circuit["meta"]["n_neurons"]
        self.gain = gain
        self.decay = float(np.exp(-1.0 / tau))
        self.thr = threshold
        self.refr = refractory
        self.habituation_eta = habituation_eta
        self.recovery_eta = recovery_eta
        rng = np.random.default_rng(seed)
        self.drive_gain_p = rng.uniform(0.7, 1.5, size=(self.B, 1)).astype(np.float32)
        self.noise_sigma = rng.uniform(0.005, 0.02, size=(self.B, 1)).astype(np.float32)
        self.rng = np.random.default_rng(seed + 1)

        # channel definition: v0.3 dual-core passes explicit channels;
        # fall back to whichever circuit layout we were handed
        if channels is not None:
            ch = channels
        elif "seed_local" in circuit:
            ch = {"sweet_seeds": circuit["seed_local"],
                  "bitter_seeds": np.asarray(circuit["threat_idx"], dtype=int),
                  "approach_pop": circuit["approach_idx"],
                  "threat_pop": circuit["threat_idx"]}
        elif "approach_seeds" in circuit:
            ch = {"sweet_seeds": circuit["approach_seeds"],
                  "bitter_seeds": circuit["threat_seeds"],
                  "approach_pop": circuit["approach_pop"],
                  "threat_pop": circuit["threat_pop"]}
        else:
            raise KeyError("circuit has neither v0.2 nor v0.3 channel keys")
        self.sweet_seeds = np.asarray(ch["sweet_seeds"], dtype=int)
        self.bitter_seeds = np.asarray(ch["bitter_seeds"], dtype=int)
        self.approach_pop = np.asarray(ch["approach_pop"], dtype=int)
        self.threat_pop = np.asarray(ch["threat_pop"], dtype=int)

        # per-target normalized signed weights (input conductance normalization)
        w = circuit["w"] / np.maximum(circuit["in_scale"][circuit["post"]], 1e-9)
        self.w = (w * self.gain).astype(np.float32)
        self.pre = circuit["pre"]
        self.starts = circuit["starts"]
        self.nz_targets = circuit["nz_targets"]

        # state
        self.v = np.zeros((self.B, self.N), dtype=np.float32)
        self.sp = np.zeros((self.B, self.N), dtype=np.float32)
        self.ref = np.zeros((self.B, self.N), dtype=np.int16)
        self.rate = np.zeros((self.B, self.N), dtype=np.float32)
        self.adapt = np.ones((self.B, 2), dtype=np.float32)  # [sweet, bitter]
        self.drive_ema = np.zeros((self.B, 1), dtype=np.float32)
        self._alarm = None

    def tick(self, drives: np.ndarray) -> dict:
        d = np.asarray(drives, dtype=np.float32).reshape(self.B, 1)
        pos = np.clip(d, 0.0, None)
        neg = np.clip(-d, 0.0, None)

        # recurrent input through REAL synapses (last tick's spikes)
        vals = self.sp[:, self.pre] * self.w[None, :]
        I = np.zeros((self.B, self.N), dtype=np.float32)
        I[:, self.nz_targets] = np.add.reduceat(vals, self.starts, axis=1)

        # channel drives gated by their own depression (habituation)
        I[:, self.sweet_seeds] += pos * self.adapt[:, 0:1] * self.drive_gain_p
        I[:, self.bitter_seeds] += neg * self.adapt[:, 1:2] * self.drive_gain_p
        I += self.rng.normal(0.0, 1.0, size=(self.B, self.N)).astype(np.float32) \
            * self.noise_sigma

        # LIF step
        self.v = self.v * self.decay + I
        sp = (self.v > self.thr) & (self.ref == 0)
        self.v[sp] = 0.0
        self.ref = np.maximum(self.ref - 1, 0)
        self.ref[sp] = np.int16(self.refr)
        self.sp = sp.astype(np.float32)
        self.rate = 0.85 * self.rate + 0.15 * self.sp

        # channel depression / recovery
        self.adapt[:, 0] = np.clip(
            self.adapt[:, 0] - self.habituation_eta * pos[:, 0]
            + self.recovery_eta * (pos[:, 0] < 0.05), 0.15, 1.0)
        self.adapt[:, 1] = np.clip(
            self.adapt[:, 1] - self.habituation_eta * neg[:, 0]
            + self.recovery_eta * (neg[:, 0] < 0.05), 0.15, 1.0)

        # dopamine = positive surprise on total drive energy (RPE proxy)
        energy = np.abs(d[:, 0]).astype(np.float32)
        novelty = np.clip(energy - self.drive_ema[:, 0], 0.0, None)
        self.drive_ema[:, 0] = 0.85 * self.drive_ema[:, 0] + 0.15 * energy
        return {"novelty": novelty, "energy": energy}

    def readouts(self, approach_g=None, avoid_g=None) -> dict:
        """Baseline-relative population rates (ΔF/F style).

        The real circuit reverberates: after a storm, absolute rates stay
        elevated for many ticks (near-marginal spectral radius). Learning
        must score DEVIATIONS from each citizen's own slow baseline,
        otherwise every post-storm event reads as bad. Baseline tau ~ 20.
        """
        a_now = self.rate[:, self.approach_pop].mean(axis=1)
        b_now = self.rate[:, self.threat_pop].mean(axis=1)
        if not hasattr(self, "_base_a"):
            self._base_a = a_now.copy()
            self._base_b = b_now.copy()
        self._base_a = 0.95 * self._base_a + 0.05 * a_now
        self._base_b = 0.95 * self._base_b + 0.05 * b_now
        approach = a_now - self._base_a
        avoid = b_now - self._base_b
        a = np.clip(avoid - approach * 0.5, 0.0, None)
        self._alarm = a if self._alarm is None else 0.7 * self._alarm + 0.3 * a
        return {"approach": approach, "avoid": avoid, "alarm": self._alarm.copy()}


class AffectCore:
    """v0.1: batched group-level rate core (kept for comparison; see README)."""

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
