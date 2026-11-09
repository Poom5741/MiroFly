"""
MiroFly world: opinion dynamics on real-brain affect cores + (optional)
dopamine-gated preference learning (v0.4).

Citizens: persona + AffectCore/SpikeCore batch element + optional
PreferenceLayer (MB-inspired valence & trust learning). Events carry topics;
citizen posts inherit the topic that provoked them and their author, so
followers can learn topic valence AND source trust.

No LLM in v0 — posting is template-based (MiroFish adapter swaps in LLMs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .encoder import encode

POST_TEMPLATES = {
    "approach": [
        "อ่านแล้วอบอุ่นดีนะ", "nice, this one is sweet",
        "รู้สึกดีขึ้นเยอะ", "ชอบไอเดียนี้",
    ],
    "avoid": [
        "โกรธมาก!!", "this is not okay", "คนแบบนี้ควรโดนด่า",
        "แย่มาก อ่านแล้วขนลุก",
    ],
    "alarm": [
        "ตื่นตระหนก!!", " PANIC", "วิ่ง!!!", "เตือนทุกคน",
    ],
    "meh": [
        "เห็นแล้วก็เฉยๆ", "อีกแล้วเหรอ", "...", "นอนดีกว่า",
    ],
}


@dataclass
class Event:
    tick: int
    text: str
    drive: float
    source: str = "news"
    topic: str = ""
    author: int = -1
    category: str = ""


@dataclass
class World:
    core: object                 # AffectCore/SpikeCore (batch = citizens)
    meta: dict
    pref: object | None = None   # PreferenceLayer (v0.4)
    n_citizens: int = 120
    follower_prob: float = 0.06
    seed: int = 99
    post_bias: dict = field(default_factory=lambda: dict(
        b0=-1.6, wA=1.2, wV=1.5, wL=2.0))
    readout_gain: float = 1.0   # spike-core readouts are baseline-relative
                                # deviations (~0.1 scale); group cores use 1.0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        self.feed: list[Event] = []
        self.history: list[dict] = []
        self.follows = [set() for _ in range(self.n_citizens)]
        for i in range(self.n_citizens):
            k = int(np.clip(rng.poisson(self.n_citizens * self.follower_prob), 1, 30))
            self.follows[i] = set(rng.choice(self.n_citizens, size=k,
                                             replace=False).tolist()) - {i}
        self.last_post = np.full(self.n_citizens, -10.0)
        self.last_topics = np.full(self.n_citizens, -1, dtype=int)
        self._pending_topics: list[int] = []
        self._pending_authors: set[int] = set()

    # ---------------- event handling ----------------
    def inject_event(self, tick: int, text: str, source: str = "news",
                     topic: str = ""):
        topic = topic or text[:24]
        self.feed.append(Event(tick, text, encode(text), source, topic=topic))

    def _topic_idx(self, topic: str) -> int:
        if self.pref is None:
            return -1
        if topic not in self.pref.t_idx:
            self.pref.topics.append(topic)
            self.pref.t_idx[topic] = len(self.pref.topics) - 1
            self.pref.valence = np.hstack([self.pref.valence,
                                           np.zeros((self.pref.n, 1))])
        return self.pref.t_idx[topic]

    # ---------------- one simulation tick ----------------
    def step(self, tick: int) -> dict:
        news = [e for e in self.feed if e.tick == tick]
        drives = np.zeros(self.n_citizens)
        topics_this_tick = []
        for e in news:
            d = np.full(self.n_citizens, e.drive)
            ti = self._topic_idx(e.topic)
            if self.pref is not None and ti >= 0:
                d = self.pref.modulate(d, np.array([ti]))
            drives += d
            if ti >= 0:
                topics_this_tick.append(ti)
        # posts from the previous tick reach followers with attenuation + trust
        social_authors = []
        for e in self.feed:
            if e.tick == tick - 1 and e.source == "citizen":
                d = np.full(self.n_citizens, 0.35 * e.drive)
                if self.pref is not None and e.author >= 0:
                    d = self.pref.modulate_social(d, np.full(self.n_citizens, e.author))
                for i in range(self.n_citizens):
                    if e.author in self.follows[i]:
                        drives[i] += d[i]
                social_authors.append(e.author)
        drives = np.clip(drives, -1.2, 1.2)

        self.last_topics[:] = topics_this_tick[0] if topics_this_tick else -1
        out = self.core.tick(drives)
        ro = self.core.readouts()

        # ---- v0.4 learning: dopamine-gated valence & trust updates ----
        # LAG-CORRECTED: an event at tick t is scored by the outcome at t+1
        # (the core's readouts need a tick to respond). Scoring the same tick
        # would teach every topic the pre-response state.
        if self.pref is not None:
            outcome = np.clip(self.pref.outcome_gain
                              * (ro["approach"] - ro["avoid"]), -1, 1)
            dopamine = np.asarray(out["novelty"], dtype=float)
            for ti in self._pending_topics:
                self.pref.update(np.array([ti]), outcome, dopamine)
            for a in self._pending_authors:
                self.pref.update(None, outcome, dopamine, author=a)
            self._pending_topics = list(topics_this_tick)
            self._pending_authors = set(social_authors)

        return {"drives": drives, **ro, "news": [e.text for e in news]}

    # ---------------- citizen posting ----------------
    def maybe_post(self, tick: int, readout: dict, drives: np.ndarray) -> list[Event]:
        g = self.readout_gain
        b = self.post_bias
        z = (b["b0"] + b["wA"] * g * readout["approach"]
             + b["wV"] * g * readout["avoid"] + b["wL"] * g * readout["alarm"])
        p_post = 1.0 / (1.0 + np.exp(-z))
        p_post *= np.clip(0.25 + 0.25 * (tick - self.last_post), 0, 1)
        posts = []
        r = self.rng.random(self.n_citizens)
        for i in np.where(r < p_post)[0]:
            a, v, l = (readout["approach"][i] * g, readout["avoid"][i] * g,
                       readout["alarm"][i] * g)
            if l > 0.55:
                cat = "alarm"
            elif v > 0.45:
                cat = "avoid"
            elif a > 0.35:
                cat = "approach"
            else:
                cat = "meh"
            intensity = float(np.clip(max(abs(a), v, l) * 3, 0, 1))
            text = self.rng.choice(POST_TEMPLATES[cat]) + f" [x{intensity:.1f}]"
            ti = int(self.last_topics[i])
            topic = (self.pref.topics[ti] if (self.pref is not None and ti >= 0)
                     else "")
            ev = Event(tick, text, drives[i] * 0.8, source="citizen",
                       topic=topic, author=int(i), category=cat)
            posts.append(ev)
            self.last_post[i] = tick
        self.feed.extend(posts)
        return posts
