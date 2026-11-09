"""
MiroFly connectome builder.

Loads the REAL FlyWire v783 signed connectivity (from the eonsystemspbc/fly-brain
repo data), takes the real downstream neighborhood of the real sugar-sensing
GRNs, and aggregates it into neuron groups that become the shared "affect core"
wiring for every simulated citizen.

HONESTY NOTES (keep these — they are the project's spine):
- We use REAL neuron IDs, REAL synapse counts, and REAL neurotransmitter-inferred
  signs (Eckstein et al. 2024 excitatory flags).
- Aggregating neurons into groups is a v0.1 approximation; the upgrade path is
  per-neuron spiking microcircuits (see README roadmap).
- The fly brain has NO comprehension of news. It contributes mechanistic
  dynamics (persistence, habituation, alarm propagation), not opinions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# Real FlyWire root IDs of the sugar gustatory receptor neurons, copied from
# eonsystemspbc/fly-brain code/benchmark.py EXPERIMENTS['sugar']['neu_exc'].
SUGAR_GRN_IDS = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940639259967, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
]


def load_connectivity(fly_brain_dir: str | Path):
    """Load the signed connectivity parquet (only the columns we need)."""
    p = Path(fly_brain_dir) / "data" / "2025_Connectivity_783.parquet"
    df = pd.read_parquet(p, columns=[
        "Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory",
    ])
    # signed weight: +conn for excitatory, -conn for inhibitory synapses
    signed = np.where(df["Excitatory"].to_numpy() > 0,
                      df["Connectivity"].to_numpy(), -df["Connectivity"].to_numpy())
    pre = df["Presynaptic_Index"].to_numpy()
    post = df["Postsynaptic_Index"].to_numpy()
    return pre, post, signed


def load_neuron_ids(fly_brain_dir: str | Path) -> np.ndarray:
    """Row i of the completeness CSV is neuron index i in the parquet."""
    c = pd.read_csv(Path(fly_brain_dir) / "data" / "2025_Completeness_783.csv",
                    index_col=0)
    return c.index.to_numpy()


def _kmeans(features: np.ndarray, k: int, iters: int = 12, seed: int = 7):
    """Tiny numpy k-means (enough for grouping neurons into core groups)."""
    rng = np.random.default_rng(seed)
    # k-means++-ish init: random distinct points
    idx = rng.choice(len(features), size=k, replace=False)
    centers = features[idx].astype(float)
    for _ in range(iters):
        d = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = features[m].mean(0)
    d = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def build_core(fly_brain_dir: str | Path = Path.home() / "fly-brain",
               n_groups: int = 16, max_neurons: int = 2400, seed: int = 7):
    """
    Build the affect-core wiring from real connectivity.

    Returns a dict with:
      W        (G,G)  signed group-to-group weight matrix (row = from)
      drive_g  int    the group containing the real sugar GRNs (input channel)
      approach_g list of groups positively coupled to the drive group
      avoid_g    list of groups negatively coupled to the drive group
      meta     dict with sizes/sources for honest reporting
    """
    ids = load_neuron_ids(fly_brain_dir)
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    seed_idx = np.array([id_to_idx[n] for n in SUGAR_GRN_IDS], dtype=int)

    pre, post, signed = load_connectivity(fly_brain_dir)

    # 2-hop downstream neighborhood of the real seeds, capped by weight rank
    agg = pd.Series(np.abs(signed[np.isin(pre, seed_idx)])).groupby(
        post[np.isin(pre, seed_idx)]).sum().sort_values(ascending=False)
    hop1 = agg.index[:2000]
    # second hop: strongest downstream partners of hop1
    agg2 = pd.Series(np.abs(signed[np.isin(pre, hop1)])).groupby(
        post[np.isin(pre, hop1)]).sum().sort_values(ascending=False)
    hop2 = agg2.index[: max_neurons - len(seed_idx) - len(hop1)]
    members = np.concatenate([seed_idx, hop1.to_numpy(), hop2.to_numpy()])
    member_set = np.zeros(len(ids), dtype=bool)
    member_set[members] = True

    # keep only synapses inside the neighborhood (include seed in-neighbors)
    sm = member_set[pre] & member_set[post]
    pre_s, post_s, w_s = pre[sm], post[sm], signed[sm]

    # remap to local indices
    local = -np.ones(len(ids), dtype=int)
    local[members] = np.arange(len(members))
    pre_l, post_l = local[pre_s], local[post_s]

    # features for grouping: signed in-weight from seeds, |out| degree, exc ratio
    n = len(members)
    in_from_seed = np.zeros(n)
    seed_local = local[seed_idx]
    mseed = np.isin(pre_l, seed_local)
    np.add.at(in_from_seed, post_l[mseed], w_s[mseed])
    out_abs = np.zeros(n); np.add.at(out_abs, pre_l, np.abs(w_s))
    exc = np.where(w_s > 0, w_s, 0.0)
    out_exc = np.zeros(n); np.add.at(out_exc, pre_l, exc)
    exc_ratio = np.divide(out_exc, out_abs, out=np.zeros(n), where=out_abs > 0)
    feats = np.stack([
        np.tanh(in_from_seed / 50.0),
        np.tanh(out_abs / 200.0),
        exc_ratio,
    ], axis=1)

    groups = _kmeans(feats, n_groups, seed=seed)

    # aggregate signed group-to-group weights, normalized by group size
    W = np.zeros((n_groups, n_groups))
    gs = np.bincount(groups, minlength=n_groups).astype(float)
    np.add.at(W, (groups[pre_l], groups[post_l]), w_s)
    W = W / np.maximum(gs[:, None], 1.0)

    # Spectral normalization: real synaptic weights span huge magnitudes, which
    # saturates any rate network. Scale to spectral radius ~0.9 (standard
    # practice for rate models; keeps sign structure intact).
    eig = np.abs(np.linalg.eigvals(W)).max()
    if eig > 1e-9:
        W = W * (0.9 / eig)

    # the drive group = group holding the most seed neurons
    drive_g = int(np.bincount(groups[seed_local], minlength=n_groups).argmax())

    # readout groups from REAL signed coupling of the drive group
    # (exclude the drive group itself)
    row = W[drive_g].copy()
    row[drive_g] = -np.inf  # sentinel so drive_g sorts to the negative tail
    order = np.argsort(row)
    avoid_g = [g for g in order[:6] if g != drive_g][:3]       # most inhibited
    approach_g = [g for g in order[::-1] if g != drive_g][:3]  # most excited

    meta = {
        "n_neurons": int(n),
        "n_synapses_in_core": int(sm.sum()),
        "n_seed_neurons": int(len(seed_idx)),
        "n_groups": n_groups,
        "drive_group": drive_g,
        "approach_groups": approach_g,
        "avoid_groups": avoid_g,
        "spectral_radius": 0.9,
        "seed_source": "EXPERIMENTS['sugar']['neu_exc'] of eonsystemspbc/fly-brain",
        "data": "FlyWire v783 signed connectivity (Eckstein et al. 2024 signs)",
    }
    return {"W": W, "drive_g": drive_g, "approach_g": approach_g,
            "avoid_g": avoid_g, "meta": meta}
