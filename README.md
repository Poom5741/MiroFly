# MiroFly 🪰🧠

**Give every AI agent in an opinion-dynamics world (like [MiroFish](https://github.com/666ghj/MiroFish)) a real nervous system as its affect core — built from the actual FlyWire connectome of an adult fruit fly.**

MiroFly agents don't roleplay emotions with prompts. Their internal states
(approach, avoidance, alarm, novelty, outrage persistence) are computed by
population dynamics running on **2,400 real neurons and 187k real signed
synapses** extracted from FlyWire v783, personalized per citizen, persistent
across ticks.

> ⚠️ **Honesty first** — read before quoting this project:
> The fly brain does NOT understand news. The LLM (in production) does all
> language work. The fly wiring contributes *mechanistic affect dynamics*
> (habituation, alarm cascades, novelty surprise) — not comprehension, and
> (not yet) proven predictive power. See [Ablation](#-ablation-results) below.

## Architecture

```
           MiroFish world (LLM agents, memory, social graph)
                              │ event/post text
                              ▼
                   ┌─────────────────────┐
   text ──encode──▶│  drive vector       │   sweet → real taste-input group
                   │  (approach/threat)  │   threat → real taste-inhibited groups
                   └─────────┬───────────┘
                             ▼
     ┌──────────────────────────────────────────────┐
     │  AffectCore (one per citizen, batched)       │
     │  r' = tanh( (r·adapt)@Wᵀ + drive + noise )   │
     │  · W = REAL FlyWire signed group wiring      │
     │  · adapt = synaptic depression (habituation) │
     │  · dopamine = novelty surprise (RPE proxy)   │
     └──────────────────┬───────────────────────────┘
                        ▼ readouts: approach / avoid / alarm / novelty
              prompt conditioning + posting bias
```

The trick that makes it fit on a laptop: the fly-brain repo's PyTorch runner
batches independent *trials* — **we reinterpret the batch dimension as
citizens.** 120 citizens share the real wiring, differing in gains, noise, and
their own adaptation/memory state.

## What's real (and what isn't)

| Component | Status |
|---|---|
| Neuron neighborhood | ✅ REAL — 2-hop downstream of the actual sugar GRNs (IDs from [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)) |
| Synapse weights + signs | ✅ REAL — FlyWire v783, Eckstein et al. 2024 neurotransmitter-inferred signs |
| Dynamics (v0.2) | ✅ per-neuron LIF on real synapses, batched per citizen |
| Threat channel | ⚠️ v0.1 stand-in — the real taste-*inhibited* population (roadmap: real bitter GRN IDs) |
| Spectral normalization | ⚙️ standard rate-model practice (radius 0.9), sign structure preserved |
| Comprehension of events | ❌ none — encoder/LLM maps text to drives; the brain only computes dynamics |

## Ablation results (the honesty engine)

### v0.3 — the discrimination test: real bitter GRNs

The threat channel is no longer a stand-in: **39 real bitter GRNs** (LB1a/LB1b,
FlyWire community annotations of Engert et al. 2022) drive the circuit's
second channel, alongside the 21 real sugar GRNs — 1,313 neurons, 108k signed
synapses, both channels real end to end.

Same pre-registered protocol (gain frozen at 128 from intact-only sweep,
5 seeds/condition, configuration-model rewired control). Numbers below are
the **v0.4 re-run under baseline-relative readouts** (ΔF/F style — see v0.4;
the conclusion held and got cleaner):

| metric (mean±std) | intact real wiring | rewired control | no-core (noise) |
|---|---|---|---|
| sweet response (approach pop) | 0.048±0.002 | 0.068±0.001 | 0.044±0.002 |
| sweet cross (threat pop) | 0.024±0.000 | 0.064±0.001 | 0.057±0.001 |
| bitter response (threat pop) | 0.082±0.001 | 0.115±0.000 | 0.053±0.002 |
| **D_sweet** (sweet − cross) | **+0.024±0.001** | +0.004±0.000 | **−0.013±0.002** |
| **D_bitter** (bitter − cross) | **+0.006±0.000** | −0.011±0.001 | +0.014±0.002 |
| **D_total** | **+0.029±0.001** | **−0.008±0.000** | +0.000±0.001 |

**Finding: the real circuit discriminates sweet from bitter on BOTH channels;
rewired wiring discriminates on neither.** Intact D_total = +0.029 vs rewired
−0.008, variance ~0.001. Random connectomes don't just lack specificity —
they actively cross-contaminate channels (rewired sweet_cross ≈ sweet_resp).

The threat channel is no longer a stand-in: **39 real bitter GRNs** (LB1a/LB1b,
FlyWire community annotations of Engert et al. 2022) drive the circuit's
second channel, alongside the 21 real sugar GRNs — 1,313 neurons, 108k signed
synapses, both channels real end to end.

Same pre-registered protocol (gain frozen at 128 from intact-only sweep,
5 seeds/condition, configuration-model rewired control):

| metric (mean±std) | intact real wiring | rewired control | no-core (noise) |
|---|---|---|---|
| sweet response (approach pop) | 0.086±0.002 | 0.122±0.001 | 0.072±0.003 |
| sweet cross (threat pop) | 0.057±0.001 | 0.113±0.001 | 0.092±0.002 |
| bitter response (threat pop) | 0.128±0.001 | 0.166±0.000 | 0.098±0.003 |
| **D_sweet** (sweet − cross) | **+0.029±0.002** | +0.009±0.000 | **−0.020±0.004** |
| D_bitter (bitter − cross) | −0.005±0.001 | −0.019±0.001 | +0.021±0.004 |
| **D_total** | **+0.024±0.001** | **−0.010±0.001** | +0.000±0.001 |

**Finding: the real circuit discriminates sweet from bitter; rewired wiring
cannot.** Intact D_total = +0.024 vs rewired −0.010, variance ~0.001 — the gap
is an order of magnitude above the noise. Sharper still: rewired wiring is
*worse than no brain at all* on the bitter channel (D_bitter −0.019 vs
no-core +0.021). Random connectomes don't just lack specificity — they
actively confuse channels.

Combined with v0.2, the story is now:

> v0.2 — real wiring is selective, not loud (rewired echoes more).
> v0.3 — what real wiring is selective for is *telling sweet from bitter*,
> which rewired wiring cannot do. Specificity is the measured topology's
> contribution — not gain, not capacity.

Honest caveats: effect sizes are small (hundredths in rate units), one gain
regime, short timeline, 2-hop neighborhoods. Magnitudes may move; the sign of
D_total is the claim. Run: `python discrimination_v03.py` →
`results/discrimination_v03.png`

### v0.2 — per-neuron spiking cores

First per-neuron protocol (absolute readouts, gain frozen at 64):
intact vs rewired differed far beyond noise, but rewired graphs were LOUDER
(more echo, more spillover): **"real wiring is selective, not powerful."**
v0.3's discrimination result is the constructive counterpart — the real
circuit is selective *for its channels*.

### v0.4 — dopamine-gated learning (MB-inspired)

Citizens now LEARN: a PreferenceLayer implements the mushroom-body rule
(delta-rule valence updates, gated by the core's dopamine novelty signal —
expected events do not rewrite valence). Two real citizen types in the SAME
world: 90 learners vs 30 naive (lr = 0), same real wiring, same events.

| result | value | verdict |
|---|---|---|
| learned aversion, scandal topic | valence **−0.181±0.034** | ✅ strong, correct sign |
| mundane topic (gated) | −0.043 | ✅ near zero — gate gives selectivity |
| mundane topic (ungated) | −0.093 (sum\|val\| 8.4 vs 4.5) | ✅ ungated drifts 2× |
| learned distrust of alarm-spreader | followers' trust **−0.108** | ✅ credibility dynamics |
| learned discounting (cry-wolf) at revival | learners +0.079 vs naive +0.077 | ⚠️ barely visible — drive modulation is sub-threshold at frozen gain |
| banana valence | −0.089 (expected +) | ⚠️ honest wart: storm-tail contamination at short ISI |

Mechanism notes: lag-corrected scoring (events scored by the NEXT tick's
outcome — same-tick scoring teaches pre-response state), and baseline-relative
readouts (the real circuit reverberates after storms; ΔF/F-style scoring is
required or every post-storm event reads as bad).

Run: `python demo_learning.py` → `results/learning_v04.png`

### v0.1 — grouped rate cores (superseded, kept for the record)

Per-neuron LIF dynamics on 2,400 real neurons / 187k signed synapses.
Pre-registered protocol: gain swept on intact only (frozen at 64), then 5
seeds per condition, identical drive timeline, configuration-model rewiring
(degrees + weights preserved, pairing destroyed).

| metric (mean±std) | intact real wiring | rewired control | no-core (noise) |
|---|---|---|---|
| threat response | 0.127±0.001 | 0.148±0.001 | 0.098±0.002 |
| twist approach response | 0.048±0.000 | 0.192±0.000 | 0.045±0.002 |
| alarm AUC | 1.833±0.008 | 1.816±0.008 | 1.626±0.029 |
| echo after storm | 0.067±0.000 | 0.173±0.000 | 0.074±0.001 |
| max firing fraction | 0.216±0.007 | 0.263±0.002 | 0.200±0.005 |

**Finding: topology matters — but not the way we guessed.** The real circuit
responds *differently* from rewired wiring with tiny variance, yet the rewired
graphs are LOUDER: they echo more, spill more across channels, habituate more.
The measured connectome is **selective, not powerful**: a feeding circuit
doesn't amplify arbitrary injected signals, while random pairing manufactures
parasitic feedback loops that boost everything. This is the v0.2 headline:

> Real wiring ≠ generic compute. It constrains propagation to its own paths.
> A random connectome is a better reservoir — and a worse brain.

Consistent with the community's honest negatives (FlyDoom's motion selectivity,
ENSOphilia's forecast null): measured wiring buys *specificity*, not horsepower.

Run: `python ablation_v02.py` → `results/ablation_v02.png`

### v0.1 — grouped rate cores (superseded, kept for the record)

Same scandal storyline, 120 citizens, 3 conditions:
| metric | intact real wiring | shuffled wiring | no-core (noise) |
|---|---|---|---|
| peak outrage | 0.991 | 0.991 | 0.043 |
| outrage half-life (ticks) | 9 | 7 | 2 |
| total posts | 1200 | 1230 | 810 |
| mean alarm | 0.399 | 0.397 | 0.098 |

**Two honest findings (world-level, v0.1 cores):**
1. The affect machinery *does* matter: removing the core collapses posting
   (−33%) and alarm (−75%). Citizens with nervous systems panic; noise doesn't.
2. **Intact ≈ shuffled at v0.1.** The real topology does not yet beat a
   rewired control through 16-group aggregation. That is the headline until
   per-neuron spiking microcircuits change it. If it never changes, we
   publish that too (see ENSOphilia's precedent of honest negatives).

Run it yourself: `python ablation.py` → `results/ablation.png`

## Quick start

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tests/test_core.py   # sanity: threat→avoid, habituation, novelty
.venv/bin/python demo.py              # office-cat scandal cascade (v0.1 group cores)
.venv/bin/python ablation_v02.py      # v0.2 per-neuron: real vs rewired vs no-core
.venv/bin/python ablation.py          # v0.1 grouped: real vs shuffled vs no-core
.venv/bin/python -m mirofly.mirofish_adapter   # MiroFish integration API demo
```

Requires `~/fly-brain` (clone of eonsystemspbc/fly-brain) for the data files.

## MiroFish integration plan

See `mirofly/mirofish_adapter.py` — the contract is:

```python
mf = MiroFly.from_fly_brain("~/fly-brain", n_agents=N)
state = mf.affect(agent_id, event_text)       # affective tick (no tokens)
prompt += MiroFly.state_clause(state)         # "you feel alarm rising; this feels new"
mf.learn(agent_id, text, reward=feedback)     # dopamine-tagged memory
```

Affect ticks cost **zero LLM tokens** — emotions between posts are free,
which directly attacks MiroFish's stated high-consumption problem.

## Roadmap

- [x] v0.2: per-neuron spiking microcircuits + pre-registered gain protocol
  → **done; real topology is selective, not amplifying (see above)**
- [x] v0.3: real bitter GRNs as the threat channel
  → **done; intact discriminates sweet/bitter, rewired cannot (see above)**
- [x] v0.4: mushroom-body dopamine rule (MB-inspired PreferenceLayer)
  → **done; learned aversion −0.18, gating = selectivity, trust dynamics;
     cry-wolf contrast sub-threshold (honest wart, see v0.4 table)**
- [ ] v0.4b: world-level ablation (follower-graph cascade on v0.3 cores)
- [ ] v0.5: wire into a real MiroFish fork; ablate against a real opinion dataset
- [ ] v0.6: calibration report — publish whatever it shows

## Credits & licensing

- Connectome data: FlyWire v783 (Dorkenwald et al. 2024), signs from
  Eckstein et al. 2024 — obtained via [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)'s
  bundled parquet (that repo is GPL-2.0-or-later; MiroFly **reads its data
  files but imports none of its code**).
- Bitter GRN IDs (v0.3): FlyWire community cell-type annotations of labellar
  bitter GRNs (Engert et al. 2022, eLife 78110), compiled in
  [sydneywalker1/flywire499](https://github.com/sydneywalker1/flywire499)
  (`bitter_GRNs.csv`; bundled in `data/` with attribution — facts and IDs,
  upstream repo has no license claim). Cross-validated: 39/39 IDs present in
  v783; zero overlap with sugar set; 19/21 fly-brain sugar IDs reproduced.
- Model inspiration: Shiu et al., *Nature* 2024 — the leaky integrate-and-fire
  whole-fly-brain model.
- MiroFish: 666ghj/MiroFish — the swarm-prediction engine this plugs into.
- MiroFly code: MIT. Data retains its upstream terms.
