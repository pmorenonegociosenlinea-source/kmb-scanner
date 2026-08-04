from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf


def _days_to_expiration(expiration: pd.Timestamp) -> int:
    return int((expiration.date() - pd.Timestamp.today().date()).days)


def _select_expiration(option_chain: Dict[str, pd.DataFrame]) -> Optional[pd.Timestamp]:
    expirations = list(option_chain.keys())
    if not expirations:
        return None

    expirations_sorted = sorted(expirations)
    target = 45
    closest = min(expirations_sorted, key=lambda expiry: abs(_days_to_expiration(expiry) - target))
    return closest


def _select_option_row(option_df: pd.DataFrame, *, target_delta: float, strike_offset: int = 0) -> Optional[Dict[str, object]]:
    if option_df is None or option_df.empty:
        return None

    candidate = option_df.copy()
    candidate = candidate.dropna(subset=["strike"])
    if candidate.empty:
        return None

    candidate["delta"] = pd.to_numeric(candidate.get("delta", pd.Series([np.nan] * len(candidate))), errors="coerce")
    candidate["delta"] = candidate["delta"].fillna(np.nan)

    if candidate["delta"].notna().any():
        candidate = candidate.sort_values(by=["delta"], ascending=True)
        best_row = candidate.iloc[0]
        if not np.isnan(best_row["delta"]):
            return {
                "strike": float(best_row["strike"]),
                "delta": float(best_row["delta"]),
                "last_price": float(best_row.get("lastPrice", 0.0)),
            }

    candidate = candidate.sort_values(by=["strike"], ascending=True)
    selected = candidate.iloc[0]
    return {
        "strike": float(selected["strike"] + strike_offset),
        "delta": float(selected.get("delta", target_delta)),
        "last_price": float(selected.get("lastPrice", 0.0)),
    }


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


def _placeholder_trade(ticker: str, strategy: str, current_price: Optional[float]) -> Dict[str, object]:
    price = float(current_price) if current_price is not None else 100.0
    if strategy == "Bull Put Spread":
        short_strike = round(price - max(1.0, price * 0.02), 2)
        long_strike = round(short_strike - 5.0, 2)
        width = round(short_strike - long_strike, 2)
        estimated_credit = round(max(0.35, min(3.5, price * 0.004)), 2)
        max_risk = round(max(width - estimated_credit, 0.01), 2)
        return_on_risk = round(estimated_credit / max_risk, 2) if max_risk > 0 else 0.0
        return {
            "ticker": ticker,
            "strategy": strategy,
            "expiration": "45 DTE (placeholder)",
            "short_strike": short_strike,
            "long_strike": long_strike,
            "width": width,
            "estimated_credit": estimated_credit,
            "max_risk": max_risk,
            "probability_of_profit": 0.70,
            "return_on_risk": return_on_risk,
        }

    short_strike = round(price + max(1.0, price * 0.02), 2)
    long_strike = round(short_strike + 5.0, 2)
    width = round(long_strike - short_strike, 2)
    estimated_credit = round(max(0.35, min(3.5, price * 0.004)), 2)
    max_risk = round(max(width - estimated_credit, 0.01), 2)
    return_on_risk = round(estimated_credit / max_risk, 2) if max_risk > 0 else 0.0
    return {
        "ticker": ticker,
        "strategy": strategy,
        "expiration": "45 DTE (placeholder)",
        "short_strike": short_strike,
        "long_strike": long_strike,
        "width": width,
        "estimated_credit": estimated_credit,
        "max_risk": max_risk,
        "probability_of_profit": 0.70,
        "return_on_risk": return_on_risk,
    }


def build_trade(ticker: str, strategy: str) -> Dict[str, object]:
    if not ticker:
        raise ValueError("Ticker is required")

    current_price = _current_price_from_yfinance(ticker)
    if strategy not in {"Bull Put Spread", "Bear Call Spread"}:
        return {
            "ticker": ticker,
            "strategy": strategy,
            **_placeholder_trade(ticker, strategy, current_price),
        }

    try:
        stock = yf.Ticker(ticker)
        chain = stock.options
        if not chain:
            raise ValueError(f"No option chain available for {ticker}")

        option_chain = stock.option_chain(chain[0])
    except Exception as exc:
        return {
            "ticker": ticker,
            "strategy": strategy,
            **_placeholder_trade(ticker, strategy, current_price),
            "message": f"Using placeholder trade values because live options data is unavailable: {exc}",
        }

    if not isinstance(option_chain, tuple) or len(option_chain) != 2:
        return {
            "ticker": ticker,
            "strategy": strategy,
            **_placeholder_trade(ticker, strategy, current_price),
            "message": "Using placeholder trade values because the options response was unexpected.",
        }

    calls, puts = option_chain
    if puts is None or calls is None:
        return {
            "ticker": ticker,
            "strategy": strategy,
            **_placeholder_trade(ticker, strategy, current_price),
            "message": "Using placeholder trade values because calls or puts were missing.",
        }

    expirations = list(stock.options)
    if not expirations:
        return {
            "ticker": ticker,
            "strategy": strategy,
            **_placeholder_trade(ticker, strategy, current_price),
            "message": "Using placeholder trade values because no expirations were available.",
        }

    selected_expiration = None
    selected_expiration_value = None
    for expiration in expirations:
        try:
            chain_data = stock.option_chain(expiration)
            if chain_data and isinstance(chain_data, tuple) and len(chain_data) == 2:
                if not chain_data[1].empty:
                    selected_expiration = expiration
                    selected_expiration_value = chain_data
                    break
        except Exception:
            continue

    if selected_expiration is None or selected_expiration_value is None:
        return {
            "ticker": ticker,
            "strategy": strategy,
            **_placeholder_trade(ticker, strategy, current_price),
            "message": "Using placeholder trade values because no usable expiration was found.",
        }

    calls_df, puts_df = selected_expiration_value
    if puts_df.empty or calls_df.empty:
        return {
            "ticker": ticker,
            "strategy": strategy,
            **_placeholder_trade(ticker, strategy, current_price),
            "message": "Using placeholder trade values because the selected expiration had no rows.",
        }

    puts_df = puts_df.copy()
    calls_df = calls_df.copy()

    puts_df["strike"] = pd.to_numeric(puts_df.get("strike"), errors="coerce")
    calls_df["strike"] = pd.to_numeric(calls_df.get("strike"), errors="coerce")
    puts_df["lastPrice"] = pd.to_numeric(puts_df.get("lastPrice"), errors="coerce")
    calls_df["lastPrice"] = pd.to_numeric(calls_df.get("lastPrice"), errors="coerce")
    puts_df["delta"] = pd.to_numeric(puts_df.get("delta"), errors="coerce")
    calls_df["delta"] = pd.to_numeric(calls_df.get("delta"), errors="coerce")

    puts_df = puts_df.dropna(subset=["strike", "lastPrice"])
    calls_df = calls_df.dropna(subset=["strike", "lastPrice"])

    if puts_df.empty or calls_df.empty:
        return {
            "ticker": ticker,
            "strategy": strategy,
            "message": "No valid strike rows available.",
        }

    if strategy == "Bull Put Spread":
        short_put = _select_option_row(puts_df, target_delta=0.18)
        if short_put is None:
            return {
                "ticker": ticker,
                "strategy": strategy,
                "message": "Could not identify a suitable short put.",
            }
        long_put = _select_option_row(puts_df, target_delta=0.18, strike_offset=-5)
        if long_put is None:
            return {
                "ticker": ticker,
                "strategy": strategy,
                "message": "Could not identify a suitable long put.",
            }
        short_strike = short_put["strike"]
        long_strike = long_put["strike"]
        width = short_strike - long_strike
        estimated_credit = max(short_put["last_price"] - long_put["last_price"], 0.0)
        max_risk = max(width - estimated_credit, 0.0)
        probability_of_profit = 1 - abs(short_put["delta"])
        return_on_risk = estimated_credit / max_risk if max_risk > 0 else 0.0

        return {
            "ticker": ticker,
            "strategy": strategy,
            "expiration": selected_expiration,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "width": width,
            "estimated_credit": estimated_credit,
            "max_risk": max_risk,
            "probability_of_profit": probability_of_profit,
            "return_on_risk": return_on_risk,
        }

    if strategy == "Bear Call Spread":
        short_call = _select_option_row(calls_df, target_delta=0.18)
        if short_call is None:
            return {
                "ticker": ticker,
                "strategy": strategy,
                "message": "Could not identify a suitable short call.",
            }
        long_call = _select_option_row(calls_df, target_delta=0.18, strike_offset=5)
        if long_call is None:
            return {
                "ticker": ticker,
                "strategy": strategy,
                "message": "Could not identify a suitable long call.",
            }
        short_strike = short_call["strike"]
        long_strike = long_call["strike"]
        width = long_strike - short_strike
        estimated_credit = max(short_call["last_price"] - long_call["last_price"], 0.0)
        max_risk = max(width - estimated_credit, 0.0)
        probability_of_profit = 1 - abs(short_call["delta"])
        return_on_risk = estimated_credit / max_risk if max_risk > 0 else 0.0

        return {
            "ticker": ticker,
            "strategy": strategy,
            "expiration": selected_expiration,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "width": width,
            "estimated_credit": estimated_credit,
            "max_risk": max_risk,
            "probability_of_profit": probability_of_profit,
            "return_on_risk": return_on_risk,
        }

    return {
        "ticker": ticker,
        "strategy": strategy,
        "message": "Unsupported strategy.",
    }
