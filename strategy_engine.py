from __future__ import annotations

from typing import Dict


def get_strategy(metrics: Dict[str, float]) -> str:
    price = metrics.get("current_price")
    ema20 = metrics.get("ema20")
    ema50 = metrics.get("ema50")
    rsi = metrics.get("rsi14")

    if price is None or ema20 is None or ema50 is None or rsi is None:
        return "Wait"

    if price > ema20 and ema20 > ema50 and 45 <= rsi <= 65:
        return "Bull Put Spread"

    if price < ema20 and ema20 < ema50 and rsi < 45:
        return "Bear Call Spread"

    return "Wait"
