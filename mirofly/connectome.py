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


def build_neuron_core(fly_brain_dir: str | Path = Path.home() / "fly-brain",
                      max_neurons: int = 2400, seed: int = 7,
                      readout_k: int = 30):
    """
    v0.2: per-neuron circuit (no group aggregation).

    Returns edges sorted by postsynaptic target, ready for batched sparse
    matvec via np.add.reduceat, plus real seed/approach/threat neuron sets:
      - seeds:    the real sugar GRNs (drive channel, approach energy)
      - approach: top neurons by real positive signed input from seeds
      - threat:   top neurons by real negative (inhibitory) input from seeds
                  — the real taste-inhibited population (v0.1 stand-in note
                  for bitter GRNs still applies)
    """
    ids = load_neuron_ids(fly_brain_dir)
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    seed_idx = np.array([id_to_idx[n] for n in SUGAR_GRN_IDS], dtype=int)

    pre, post, signed = load_connectivity(fly_brain_dir)

    # 2-hop downstream neighborhood of the real seeds, capped by weight rank
    agg = pd.Series(np.abs(signed[np.isin(pre, seed_idx)])).groupby(
        post[np.isin(pre, seed_idx)]).sum().sort_values(ascending=False)
    hop1 = agg.index[:2000]
    agg2 = pd.Series(np.abs(signed[np.isin(pre, hop1)])).groupby(
        post[np.isin(pre, hop1)]).sum().sort_values(ascending=False)
    hop2 = agg2.index[: max_neurons - len(seed_idx) - len(hop1)]
    members = np.concatenate([seed_idx, hop1.to_numpy(), hop2.to_numpy()])
    member_set = np.zeros(len(ids), dtype=bool)
    member_set[members] = True

    sm = member_set[pre] & member_set[post]
    pre_s, post_s, w_s = pre[sm], post[sm], signed[sm]

    local = -np.ones(len(ids), dtype=int)
    local[members] = np.arange(len(members))
    pre_l, post_l = local[pre_s], local[post_s]

    # real readout sets by coupling from the seeds
    n = len(members)
    from_seed = np.zeros(n)
    seed_local = local[seed_idx]
    mseed = np.isin(pre_l, seed_local)
    np.add.at(from_seed, post_l[mseed], w_s[mseed])
    from_seed[seed_local] = 0.0  # exclude seeds from their own readouts
    order = np.argsort(from_seed)
    approach_idx = order[-readout_k:].tolist()          # strongest excited
    threat_idx = order[:readout_k].tolist()             # strongest inhibited

    # sort edges by postsynaptic target for reduceat-style sparse matvec
    e = np.argsort(post_l, kind="stable")
    pre_sorted = pre_l[e]
    post_sorted = post_l[e]
    w_sorted = w_s[e]
    nz_targets, starts, counts = np.unique(post_sorted, return_index=True,
                                           return_counts=True)

    # per-target input normalization scale (sum of |w| into each target)
    in_scale = np.zeros(n)
    np.add.at(in_scale, post_l, np.abs(w_s))

    meta = {
        "n_neurons": int(n),
        "n_synapses_in_core": int(sm.sum()),
        "n_seed_neurons": int(len(seed_idx)),
        "n_approach": len(approach_idx),
        "n_threat": len(threat_idx),
        "seed_source": "EXPERIMENTS['sugar']['neu_exc'] of eonsystemspbc/fly-brain",
        "data": "FlyWire v783 signed connectivity (Eckstein et al. 2024 signs)",
    }
    return {
        "pre": pre_sorted, "post": post_sorted, "w": w_sorted,
        "nz_targets": nz_targets, "starts": starts,
        "in_scale": in_scale,
        "seed_local": seed_local, "approach_idx": approach_idx,
        "threat_idx": threat_idx, "meta": meta,
    }


def build_dual_core(fly_brain_dir: str | Path = Path.home() / "fly-brain",
                    max_neurons: int = 3200, seed: int = 7, readout_k: int = 30):
    """
    v0.3: TWO real sensory channels — sugar (approach) AND bitter (threat).

    Bitter GRN root IDs come from FlyWire community annotations (Engert et
    al. 2022 labels), bundled in data/bitter_grns_flywire499.csv (compiled
    by sydneywalker1/flywire499; validated: all 39 IDs present in v783,
    zero overlap with sugar set, 19/21 repo sugar IDs reproduced).

    Drive: sweet energy -> real sugar GRNs; bitter energy -> real bitter GRNs.
    Readouts: top neurons really excited by each seed set.
    """
    ids = load_neuron_ids(fly_brain_dir)
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    sugar_ids = set(SUGAR_GRN_IDS)
    bdf = pd.read_csv(Path(__file__).resolve().parent.parent / "data"
                      / "bitter_grns_flywire499.csv")
    bitter_ids = set(bdf.root_id.astype("int64")) & set(ids.tolist())
    sugar_seed_idx = np.array([id_to_idx[i] for i in sugar_ids], dtype=int)
    bitter_seed_idx = np.array([id_to_idx[i] for i in bitter_ids], dtype=int)

    pre, post, signed = load_connectivity(fly_brain_dir)

    def two_hop(seeds):
        agg = pd.Series(np.abs(signed[np.isin(pre, seeds)])).groupby(
            post[np.isin(pre, seeds)]).sum().sort_values(ascending=False)
        h1 = agg.index[:2000]
        agg2 = pd.Series(np.abs(signed[np.isin(pre, h1)])).groupby(
            post[np.isin(pre, h1)]).sum().sort_values(ascending=False)
        h2 = agg2.index[: 800]
        return np.concatenate([h1.to_numpy(), h2.to_numpy()])

    extra = np.concatenate([two_hop(sugar_seed_idx), two_hop(bitter_seed_idx)])
    members = np.unique(np.concatenate([sugar_seed_idx, bitter_seed_idx, extra]))
    members = members[:max_neurons]
    member_set = np.zeros(len(ids), dtype=bool)
    member_set[members] = True

    sm = member_set[pre] & member_set[post]
    pre_s, post_s, w_s = pre[sm], post[sm], signed[sm]
    local = -np.ones(len(ids), dtype=int)
    local[members] = np.arange(len(members))
    pre_l, post_l = local[pre_s], local[post_s]

    n = len(members)
    all_seeds = np.concatenate([sugar_seed_idx, bitter_seed_idx])
    seed_local_all = local[all_seeds]
    from_sugar = np.zeros(n); from_bitter = np.zeros(n)
    sugar_l = local[sugar_seed_idx]; bitter_l = local[bitter_seed_idx]
    msug = np.isin(pre_l, sugar_l)
    np.add.at(from_sugar, post_l[msug], w_s[msug])
    mbit = np.isin(pre_l, bitter_l)
    np.add.at(from_bitter, post_l[mbit], w_s[mbit])
    exclude = set(seed_local_all.tolist())
    approach_pop = [int(i) for i in np.argsort(from_sugar)[-readout_k:]
                    if int(i) not in exclude]
    threat_pop = [int(i) for i in np.argsort(from_bitter)[-readout_k:]
                  if int(i) not in exclude]

    e = np.argsort(post_l, kind="stable")
    pre_sorted, post_sorted, w_sorted = pre_l[e], post_l[e], w_s[e]
    nz_targets, starts = np.unique(post_sorted, return_index=True)
    in_scale = np.zeros(n)
    np.add.at(in_scale, post_l, np.abs(w_s))

    meta = {
        "n_neurons": int(n),
        "n_synapses_in_core": int(sm.sum()),
        "n_sugar_seeds": int(len(sugar_seed_idx)),
        "n_bitter_seeds": int(len(bitter_seed_idx)),
        "n_approach": len(approach_pop),
        "n_threat": len(threat_pop),
        "seed_source": "sugar: fly-brain repo; bitter: FlyWire community "
                       "annotations (Engert et al. 2022) via flywire499",
        "data": "FlyWire v783 signed connectivity (Eckstein et al. 2024 signs)",
    }
    return {
        "pre": pre_sorted, "post": post_sorted, "w": w_sorted,
        "nz_targets": nz_targets, "starts": starts, "in_scale": in_scale,
        "approach_seeds": sugar_l, "threat_seeds": bitter_l,
        "approach_pop": approach_pop, "threat_pop": threat_pop,
        "meta": meta,
    }


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
