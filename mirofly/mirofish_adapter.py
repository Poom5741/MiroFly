"""
MiroFish adapter — the integration contract between MiroFish agents and
MiroFly affect cores. v0.1 documents the API; the demo below shows the loop
with a stub LLM so it runs without any API key.

Integration sketch (when wiring into MiroFish's Python backend):

    from mirofly import MiroFly

    mf = MiroFly.from_fly_brain(Path("~/fly-brain"))   # builds cores once

    # per simulation tick, for each agent persona:
    state = mf.affect(agent_id, event_text)   # -> {approach, avoid, alarm, novelty}
    prompt = base_prompt + mf.state_clause(state)  # inject feeling language
    action = llm(prompt)                       # MiroFish agent acts as usual
    mf.learn(agent_id, event_text, reward=action_feedback)  # dopamine tag

The fly brain never sees text; the LLM stays the language layer, the core
stays the mechanistic affect layer.
"""

from __future__ import annotations

from pathlib import Path

from .connectome import build_core
from .core import AffectCore
from .encoder import encode


class MiroFly:
    def __init__(self, W, drive_g, meta, n_agents: int):
        self.core = AffectCore(W, drive_g, n_agents,
                               avoid_g=meta["avoid_groups"])
        self.meta = meta
        self.n_agents = n_agents
        self.memory_tags: list[tuple[int, str, float]] = []

    @classmethod
    def from_fly_brain(cls, fly_brain_dir: str | Path = Path.home() / "fly-brain",
                       n_agents: int = 200):
        c = build_core(fly_brain_dir)
        return cls(c["W"], c["drive_g"], c["meta"], n_agents)

    def affect(self, agent_id: int, text: str, side_drive: float | None = None):
        drive = encode(text) if side_drive is None else side_drive
        drives = np.zeros(self.n_agents)
        drives[agent_id] = drive
        self.core.tick(drives)
        ro = self.core.readouts(self.meta["approach_groups"],
                                self.meta["avoid_groups"])
        return {k: float(v[agent_id]) for k, v in ro.items()}

    def learn(self, agent_id: int, text: str, reward: float):
        """Dopamine-tag a memory: |reward| above threshold is consolidated."""
        if abs(reward) > 0.5:
            self.memory_tags.append((agent_id, text, reward))

    @staticmethod
    def state_clause(state: dict) -> str:
        """Translate readouts into feeling language for the agent's prompt."""
        parts = []
        if state["alarm"] > 0.55:
            parts.append("you feel a spike of alarm")
        elif state["avoid"] > 0.45:
            parts.append("you feel irritated and disapproving")
        elif state["approach"] > 0.35:
            parts.append("you feel drawn to this, it feels good")
        else:
            parts.append("you feel flat, nothing grabs you")
        if state["novelty"] > 0.1:
            parts.append("this feels surprisingly new")
        return "Internal state: " + "; ".join(parts) + "."


if __name__ == "__main__":
    mf = MiroFly.from_fly_brain(n_agents=3)
    for t in ["แมวโดนเอาไปทำซุป โกรธมาก!!", "ซุปแมว ซุปแมว ซุปแมว",
              "ข่าวดี! แมวยังมีชีวิต อร่อยกล้วยหวาน win"]:
        s = mf.affect(0, t)
        print(f"{t[:38]:<40} approach={s['approach']:+.2f} "
              f"avoid={s['avoid']:+.2f} alarm={s['alarm']:+.2f} "
              f"novelty={s['novelty']:.2f}")
        print("   ->", MiroFly.state_clause(s))
