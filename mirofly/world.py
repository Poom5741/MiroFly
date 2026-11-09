"""
MiroFly world: a mini opinion-dynamics sandbox with real-brain affect cores.

Each citizen = a persona + an AffectCore batch element (real FlyWire wiring,
personalized gains, persistent adaptation state). Events hit the feed; citizens
develop internal affective states; they post; posts propagate to followers.

No LLM in v0 — posting is template-based. The MiroFish integration swaps
templates for LLM agent calls (see mirofish_adapter.py).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

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


@dataclass
class World:
    core: object                 # AffectCore instance (shared, batch = citizens)
    meta: dict
    n_citizens: int = 120
    follower_prob: float = 0.06
    seed: int = 99
    post_bias: dict = field(default_factory=lambda: dict(
        b0=-1.1, wA=1.2, wV=1.5, wL=2.0))

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        self.feed: list[Event] = []
        self.history: list[dict] = []
        # random follower graph: each citizen follows ~ n*p others
        self.follows = [
            set() for _ in range(self.n_citizens)
        ]
        for i in range(self.n_citizens):
            k = rng.poisson(self.n_citizens * self.follower_prob)
            k = int(np.clip(k, 1, 30))
            self.follows[i] = set(rng.choice(self.n_citizens, size=k,
                                             replace=False).tolist()) - {i}
        # last time each citizen posted (latency realism)
        self.last_post = np.full(self.n_citizens, -10.0)

    # ---------------- event handling ----------------
    def inject_event(self, tick: int, text: str, source: str = "news"):
        self.feed.append(Event(tick, text, encode(text), source))

    # ---------------- one simulation tick ----------------
    def step(self, tick: int) -> dict:
        # every citizen receives the news drives of this tick
        news = [e for e in self.feed if e.tick == tick]
        drives = np.zeros(self.n_citizens)
        seen_texts = []
        for e in news:
            drives += e.drive
            seen_texts.append(e.text)
        # posts from the previous tick reach followers with attenuation
        for e in self.feed:
            if e.tick == tick - 1 and e.source == "citizen":
                for i in range(self.n_citizens):
                    if e.author in self.follows[i]:
                        drives[i] += 0.35 * e.drive

        drives = np.clip(drives, -1.2, 1.2)
        self.core.tick(drives)
        ro = self.core.readouts(self.meta["approach_groups"],
                                self.meta["avoid_groups"])
        return {"drives": drives, **ro, "news": seen_texts}

    # ---------------- citizen posting ----------------
    def maybe_post(self, tick: int, readout: dict, drives: np.ndarray,
                   new_events: list[Event]) -> list[Event]:
        b = self.post_bias
        z = (b["b0"] + b["wA"] * readout["approach"]
             + b["wV"] * readout["avoid"] + b["wL"] * readout["alarm"])
        p_post = 1.0 / (1.0 + np.exp(-z))
        # posting latency: louder states post sooner
        p_post *= np.clip(0.25 + 0.25 * (tick - self.last_post), 0, 1)
        posts = []
        r = self.rng.random(self.n_citizens)
        for i in np.where(r < p_post)[0]:
            a, v, l = readout["approach"][i], readout["avoid"][i], readout["alarm"][i]
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
            ev = Event(tick, text, drives[i] * 0.8, source="citizen")
            ev.author = int(i)
            ev.category = cat
            posts.append(ev)
            self.last_post[i] = tick
        self.feed.extend(posts)
        return posts
