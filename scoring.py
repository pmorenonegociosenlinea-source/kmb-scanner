from __future__ import annotations

from typing import Dict


def calculate_score(metrics: Dict[str, float]) -> int:
    score = 0
    if metrics["current_price"] > metrics["ema20"]:
        score += 20
    if metrics["ema20"] > metrics["ema50"]:
        score += 20
    if 45 <= metrics["rsi14"] <= 65:
        score += 15
    if metrics["current_price"] > metrics["previous_close"]:
        score += 15
    if metrics["atr14"] < metrics["atr30_avg"]:
        score += 15
    if metrics["avg_volume"] > 1_000_000:
        score += 15
    return score


def score_to_color(score: int) -> str:
    if score >= 80:
        return "green"
    if score >= 60:
        return "yellow"
    return "red"
