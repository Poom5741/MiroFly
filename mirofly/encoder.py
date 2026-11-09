"""
MiroFly encoder v0: lexicon-based text -> drive scalar.

In the MiroFish integration this is where an LLM (or a cheap classifier)
maps an event/post into sensory drives. v0 uses a small Thai+English lexicon
so the prototype runs with zero API keys.
"""

from __future__ import annotations

import numpy as np

LEXICON = {
    # --- approach / sweet / positive (drive > 0) ---
    "sugar": 0.8, "หวาน": 0.8, "sweet": 0.7, "love": 0.7, "รัก": 0.7,
    "joy": 0.7, "ดีใจ": 0.7, "ฟิน": 0.7, "ขอบคุณ": 0.5, "thank": 0.5,
    "win": 0.6, "ชนะ": 0.6, "ปลอดภัย": 0.5, "safe": 0.5, "กิน": 0.5,
    "food": 0.4, "อร่อย": 0.7, "banana": 0.6, "กล้วย": 0.6,
    # --- threat / anger / aversive (drive < 0) ---
    "scandal": -0.9, "อุบัติเหตุ": -0.7, "โกง": -0.9, "corrupt": -0.9,
    "หาย": -0.6, "dead": -0.8, "ตาย": -0.9, "ซุป": -0.85, "soup": -0.8,
    "kill": -0.9, "rage": -0.8, "โกรธ": -0.8, "angry": -0.7,
    "crash": -0.7, "panic": -0.8, "ตื่นตระหนก": -0.8, "ไฟไหม้": -0.8,
    "โจร": -0.7, "threat": -0.7, "ล้ม": -0.6, "ล่ม": -0.7,
}


def encode(text: str) -> float:
    """Return a drive scalar in [-1, 1]."""
    t = text.lower()
    score = 0.0
    for kw, v in LEXICON.items():
        if kw in t:
            score += v
    return float(np.clip(score / 1.5, -1.0, 1.0))
