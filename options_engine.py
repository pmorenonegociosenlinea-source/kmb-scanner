from __future__ import annotations

from typing import Dict, Optional, List

import pandas as pd
import yfinance as yf


def _days_to_expiration(expiration: pd.Timestamp) -> int:
    return int((expiration.date() - pd.Timestamp.today().date()).days)


def _select_expiration(expirations: List[str]) -> Optional[str]:
    if not expirations:
        return None

    target_dte = 45
    expirations_sorted = sorted(expirations)

    def _distance(expiration: str) -> int:
        try:
            expiry = pd.Timestamp(expiration)
        except Exception:
            return 10**9
        return abs(_days_to_expiration(expiry) - target_dte)

    return min(expirations_sorted, key=_distance)


def _current_price_from_yfinance(ticker: str) -> Optional[float]:
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d", auto_adjust=False, timeout=20)
        if not history.empty:
            close = history["Close"].iloc[-1]
            if pd.notna(close):
                return float(close)
        fast_info = getattr(stock, "fast_info", None)
        if fast_info is not None:
            last_price = getattr(fast_info, "last_price", None)
            if last_price is not None:
                return float(last_price)
    except Exception:
        return None
    return None


def _price_midpoint(option_row: pd.Series) -> float:
    bid = pd.to_numeric(option_row.get("bid"), errors="coerce")
    ask = pd.to_numeric(option_row.get("ask"), errors="coerce")
    if pd.notna(bid) and pd.notna(ask):
        return float((bid + ask) / 2.0)

    last_price = pd.to_numeric(option_row.get("lastPrice"), errors="coerce")
    if pd.notna(last_price):
        return float(last_price)
    return 0.0


def _build_spread_metrics(short_row: pd.Series, long_row: pd.Series, width: float) -> Dict[str, float]:
    short_bid = pd.to_numeric(short_row.get("bid"), errors="coerce")
    long_ask = pd.to_numeric(long_row.get("ask"), errors="coerce")
    credit = float(short_bid - long_ask) if pd.notna(short_bid) and pd.notna(long_ask) else 0.0
    max_risk = max(width - credit, 0.0)
    return_on_risk = credit / max_risk if max_risk > 0 else 0.0

    short_delta = pd.to_numeric(short_row.get("delta"), errors="coerce")
    probability_of_profit = None
    if pd.notna(short_delta):
        probability_of_profit = float(max(0.0, min(1.0, 1.0 - abs(float(short_delta)))))

    return {
        "estimated_credit": round(credit, 2),
        "max_risk": round(max_risk, 2),
        "return_on_risk": float(return_on_risk),
        "probability_of_profit": probability_of_profit,
    }


def _build_bull_put_spread_candidates(puts_df: pd.DataFrame) -> List[Dict[str, float]]:
    if puts_df is None or puts_df.empty:
        return []

    prepared = puts_df.copy()
    prepared["strike"] = pd.to_numeric(prepared.get("strike"), errors="coerce")
    prepared = prepared.dropna(subset=["strike"]).sort_values(by=["strike"], ascending=True)

    if prepared.empty:
        return []

    candidates: List[Dict[str, float]] = []
    for index, short_row in prepared.iterrows():
        short_strike = float(short_row["strike"])
        long_strike = short_strike - 5.0
        long_row = prepared[prepared["strike"] == long_strike]
        if long_row.empty:
            continue

        width = short_strike - long_strike
        metrics = _build_spread_metrics(short_row, long_row.iloc[0], width)
        short_delta = pd.to_numeric(short_row.get("delta"), errors="coerce")
        if pd.isna(short_delta):
            continue
        short_delta_value = float(short_delta)
        if not (0.15 <= abs(short_delta_value) <= 0.20):
            continue

        # Extract IV Rank if present; fall back to implied volatility or 0.0
        iv_rank_raw = (
            short_row.get("ivRank")
            or short_row.get("iv_rank")
            or short_row.get("impliedVolatility")
            or short_row.get("implied_volatility")
            or 0.0
        )
        try:
            iv_rank_value = float(pd.to_numeric(iv_rank_raw, errors="coerce") or 0.0)
        except Exception:
            iv_rank_value = 0.0

        candidates.append(
            {
                "short_strike": round(short_strike, 2),
                "long_strike": round(long_strike, 2),
                "width": round(width, 2),
                "estimated_credit": metrics["estimated_credit"],
                "max_risk": metrics["max_risk"],
                "return_on_risk": metrics["return_on_risk"],
                "short_delta": float(short_delta_value),
                "probability_of_profit": metrics["probability_of_profit"],
                "iv_rank": iv_rank_value,
            }
        )

    # Rank by: 1) Highest IV Rank, 2) Highest estimated credit, 3) Highest probability of profit
    candidates.sort(
        key=lambda item: (
            -float(item.get("iv_rank", 0.0)),
            -float(item["estimated_credit"]),
            -float(item.get("probability_of_profit") or 0.0),
            float(item["short_strike"]),
        )
    )
    return candidates


def _build_bear_call_spread_candidates(calls_df: pd.DataFrame) -> List[Dict[str, float]]:
    if calls_df is None or calls_df.empty:
        return []

    prepared = calls_df.copy()
    prepared["strike"] = pd.to_numeric(prepared.get("strike"), errors="coerce")
    prepared = prepared.dropna(subset=["strike"]).sort_values(by=["strike"], ascending=True)

    if prepared.empty:
        return []

    candidates: List[Dict[str, float]] = []
    for index, short_row in prepared.iterrows():
        short_strike = float(short_row["strike"])
        long_strike = short_strike + 5.0
        long_row = prepared[prepared["strike"] == long_strike]
        if long_row.empty:
            continue

        width = long_strike - short_strike
        metrics = _build_spread_metrics(short_row, long_row.iloc[0], width)

        candidates.append(
            {
                "short_strike": round(short_strike, 2),
                "long_strike": round(long_strike, 2),
                "width": round(width, 2),
                "estimated_credit": metrics["estimated_credit"],
                "max_risk": metrics["max_risk"],
                "return_on_risk": metrics["return_on_risk"],
                "short_delta": float(pd.to_numeric(short_row.get("delta"), errors="coerce") or 0.0),
                "probability_of_profit": metrics["probability_of_profit"],
            }
        )

    candidates.sort(
        key=lambda item: (
            -float(item["return_on_risk"]),
            -float(item["estimated_credit"]),
            float(item["short_strike"]),
        )
    )
    return candidates


def _normalize_option_chain_payload(chain_data: object) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if isinstance(chain_data, tuple):
        if len(chain_data) >= 2:
            return chain_data[0], chain_data[1]
        return None, None

    if hasattr(chain_data, "calls") or hasattr(chain_data, "puts"):
        calls_df = getattr(chain_data, "calls", None)
        puts_df = getattr(chain_data, "puts", None)
        if calls_df is not None or puts_df is not None:
            return calls_df, puts_df

    if isinstance(chain_data, dict):
        calls_df = chain_data.get("calls")
        puts_df = chain_data.get("puts")
        if calls_df is not None or puts_df is not None:
            return calls_df, puts_df

    raise ValueError("Unexpected option-chain payload")


def build_trade(ticker: str, strategy: str) -> Dict[str, object]:
    if not ticker:
        raise ValueError("Ticker is required")

    if strategy != "Bull Put Spread":
        return {
            "ticker": ticker,
            "strategy": strategy,
            "message": "Unsupported strategy. Only Bull Put Spread is supported.",
        }

    try:
        stock = yf.Ticker(ticker)
        expirations = list(getattr(stock, "options", []) or [])
        if not expirations:
            raise ValueError(f"No option expirations available for {ticker}")

        selected_expiration = _select_expiration(expirations)
        if selected_expiration is None:
            raise ValueError(f"No suitable expiration available for {ticker}")

        chain_data = stock.option_chain(selected_expiration)
        calls_df, puts_df = _normalize_option_chain_payload(chain_data)
        if calls_df is None or puts_df is None:
            raise ValueError(f"Option-chain data missing for {ticker}")

        if strategy == "Bull Put Spread":
            candidates = _build_bull_put_spread_candidates(puts_df)
            if not candidates:
                raise ValueError(f"No valid Bull Put Spread candidates available for {ticker}")
            best = candidates[0]
            return {
                "ticker": ticker,
                "strategy": strategy,
                "expiration": selected_expiration,
                "short_strike": best["short_strike"],
                "long_strike": best["long_strike"],
                "width": best["width"],
                "estimated_credit": best["estimated_credit"],
                "max_risk": best["max_risk"],
                "short_delta": best["short_delta"],
                "probability_of_profit": best["probability_of_profit"],
                "return_on_risk": best["return_on_risk"],
            }

        candidates = _build_bear_call_spread_candidates(calls_df)
        if not candidates:
            raise ValueError(f"No 5-point bear call spreads available for {ticker}")
        best = candidates[0]
        return {
            "ticker": ticker,
            "strategy": strategy,
            "expiration": selected_expiration,
            "short_strike": best["short_strike"],
            "long_strike": best["long_strike"],
            "width": best["width"],
            "estimated_credit": best["estimated_credit"],
            "max_risk": best["max_risk"],
            "short_delta": best["short_delta"],
            "probability_of_profit": best["probability_of_profit"],
            "return_on_risk": best["return_on_risk"],
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "strategy": strategy,
            "message": f"Unable to build live options spread: {exc}",
        }
