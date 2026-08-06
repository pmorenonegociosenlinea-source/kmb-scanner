from __future__ import annotations

from typing import Dict, Optional, List
import math

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


def _get_expirations_from_ticker(stock) -> List[str]:
    """Robustly extract expiration dates from a yfinance Ticker-like object.

    Handles properties that may be lists, Index objects, callables, or other iterables.
    Returns a list of string dates (may be empty).
    """
    expirations = []
    try:
        raw = getattr(stock, "options", None)
        if raw is None:
            return []

        # If attribute is callable (rare), call it
        if callable(raw):
            try:
                raw = raw()
            except Exception:
                return []

        # If it's a pandas Index or numpy array-like, iterate
        try:
            # Some yfinance versions return an Index object
            expirations = [str(x) for x in list(raw) if x is not None]
        except Exception:
            # Fallback: single string
            if isinstance(raw, str):
                expirations = [raw]
            else:
                expirations = []
    except Exception:
        return []

    return expirations


def debug_expirations_for_ticker(ticker: str) -> Dict[str, object]:
    """Return raw diagnostics about how expirations are retrieved for a ticker.

    Does NOT change any trading logic. Intended for debugging only.
    """
    out: Dict[str, object] = {"ticker": ticker, "requested_symbol": None, "raw_options": None, "callable": False, "exception": None}
    try:
        stock = yf.Ticker(ticker)
        # record what symbol yfinance Ticker reports if available
        try:
            out["requested_symbol"] = getattr(stock, "ticker", None)
        except Exception:
            out["requested_symbol"] = None

        raw = None
        try:
            raw = getattr(stock, "options", None)
            out["raw_options_repr"] = repr(raw)
        except Exception as e:
            out["exception"] = f"getting attribute options: {e!r}"
            return out

        # If callable, attempt to call (some versions may expose as callable)
        if callable(raw):
            out["callable"] = True
            try:
                called = raw()
                out["raw_options_after_call_repr"] = repr(called)
                # try to coerce to list
                try:
                    out["raw_options_list"] = [str(x) for x in list(called)]
                except Exception:
                    out["raw_options_list"] = None
            except Exception as e:
                out["exception"] = f"calling options callable: {e!r}"
                return out
        else:
            # not callable; try to iterate or coerce
            try:
                out["raw_options_list"] = [str(x) for x in list(raw) if x is not None]
            except Exception as e:
                out["exception"] = f"iterating options: {e!r}"
                out["raw_options_list"] = None

        return out
    except Exception as exc:
        out["exception"] = f"unexpected: {exc!r}"
        return out


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


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _estimate_put_delta(current_price: float, strike: float, implied_vol: Optional[float], days: int) -> float:
    # Guard and defaults
    try:
        S = float(current_price)
        K = float(strike)
    except Exception:
        return 0.0

    if implied_vol is None:
        sigma = 0.30
    else:
        try:
            sigma = float(implied_vol)
        except Exception:
            sigma = 0.30

    # If IV appears expressed in percent (e.g., 25), convert to decimal
    if sigma > 5:
        sigma = sigma / 100.0

    # Small positive time to expiry (in years)
    T = max(float(days) / 365.0, 1.0 / 365.0)

    # Risk-free rate assumed zero for estimation
    try:
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    except Exception:
        return 0.0

    # Put delta = N(d1) - 1 (negative value)
    put_delta = _norm_cdf(d1) - 1.0
    return float(put_delta)


def _get_option_short_delta(
    short_row: pd.Series,
    ticker: Optional[str] = None,
    expiration: Optional[str] = None,
    current_price: Optional[float] = None,
) -> float:
    raw_delta = pd.to_numeric(short_row.get("delta"), errors="coerce")
    if pd.notna(raw_delta):
        return float(raw_delta)

    cp = current_price
    if cp is None and ticker:
        cp = _current_price_from_yfinance(ticker)
    if cp is None:
        cp = float(pd.to_numeric(short_row.get("strike"), errors="coerce") or 0.0)

    if expiration:
        try:
            dte = _days_to_expiration(pd.Timestamp(expiration))
        except Exception:
            dte = 45
    else:
        dte = 45

    iv_raw = (
        short_row.get("impliedVolatility")
        or short_row.get("implied_volatility")
        or short_row.get("iv")
        or short_row.get("ivRank")
        or short_row.get("iv_rank")
        or 0.0
    )
    try:
        iv_val = float(pd.to_numeric(iv_raw, errors="coerce") or 0.0)
    except Exception:
        iv_val = 0.0

    if 0.0 <= iv_val <= 1.0:
        implied_vol = 0.2 + iv_val * 0.4
    else:
        implied_vol = iv_val

    strike = float(pd.to_numeric(short_row.get("strike"), errors="coerce") or 0.0)
    return _estimate_put_delta(cp, strike, implied_vol, dte)


def _evaluate_bull_put_spreads(
    puts_df: pd.DataFrame,
    ticker: Optional[str] = None,
    expiration: Optional[str] = None,
    current_price: Optional[float] = None,
) -> List[Dict[str, object]]:
    if puts_df is None or puts_df.empty:
        return []

    prepared = puts_df.copy()
    prepared["strike"] = pd.to_numeric(prepared.get("strike"), errors="coerce")
    prepared = prepared.dropna(subset=["strike"]).sort_values(by=["strike"], ascending=True)
    if prepared.empty:
        return []

    candidates: List[Dict[str, object]] = []
    for _, short_row in prepared.iterrows():
        short_strike = float(short_row["strike"])
        long_strike = round(short_strike - 5.0, 2)
        long_rows = prepared[prepared["strike"] == long_strike]

        short_bid = pd.to_numeric(short_row.get("bid"), errors="coerce")
        long_ask = pd.to_numeric(long_rows.iloc[0].get("ask"), errors="coerce") if not long_rows.empty else None
        long_delta = None
        if not long_rows.empty:
            try:
                long_delta_val = pd.to_numeric(long_rows.iloc[0].get("delta"), errors="coerce")
                long_delta = float(long_delta_val) if pd.notna(long_delta_val) else None
            except Exception:
                long_delta = None

        iv_rank_raw = (
            short_row.get("ivRank")
            or short_row.get("iv_rank")
            or 0.0
        )
        try:
            iv_rank_val = float(pd.to_numeric(iv_rank_raw, errors="coerce") or 0.0)
        except Exception:
            iv_rank_val = 0.0

        estimated_credit = 0.0
        if pd.notna(short_bid) and pd.notna(long_ask):
            estimated_credit = float(short_bid - long_ask)

        max_risk = 5.0 - estimated_credit
        return_on_risk = float(estimated_credit / max_risk) if max_risk > 0.0 else 0.0

        raw_delta_val = pd.to_numeric(short_row.get("delta"), errors="coerce")
        if pd.notna(raw_delta_val):
            short_delta = float(raw_delta_val)
            estimated_short_delta = None
        else:
            short_delta = _get_option_short_delta(
                short_row,
                ticker=ticker,
                expiration=expiration,
                current_price=current_price,
            )
            estimated_short_delta = float(short_delta)

        candidate: Dict[str, object] = {
            "expiration": expiration,
            "short_strike": round(short_strike, 2),
            "long_strike": long_strike if not long_rows.empty else None,
            "short_bid": float(short_bid) if pd.notna(short_bid) else None,
            "long_ask": float(long_ask) if pd.notna(long_ask) else None,
            "estimated_credit": round(estimated_credit, 2),
            "max_risk": round(max_risk, 2),
            "return_on_risk": return_on_risk,
            "short_delta": float(short_delta),
            "estimated_short_delta": estimated_short_delta,
            "long_delta": long_delta,
            "iv_rank": iv_rank_val,
            "accepted": False,
            "rejection_reasons": [],
        }

        if long_rows.empty:
            candidate["rejection_reasons"].append("no matching 5-width long strike")
        if candidate["short_bid"] is None or candidate["long_ask"] is None:
            candidate["rejection_reasons"].append("missing short bid or long ask")
        if candidate["estimated_credit"] <= 0.0:
            candidate["rejection_reasons"].append("estimated credit not positive")
        if abs(candidate["short_delta"]) < 0.15:
            candidate["rejection_reasons"].append("delta too low")
        elif abs(candidate["short_delta"]) > 0.20:
            candidate["rejection_reasons"].append("delta too high")

        candidate["accepted"] = (
            not candidate["rejection_reasons"]
            and candidate["estimated_credit"] > 0.0
            and 0.15 <= abs(candidate["short_delta"]) <= 0.20
        )

        candidates.append(candidate)

    return candidates


def _build_bull_put_spread_candidates(
    puts_df: pd.DataFrame,
    ticker: Optional[str] = None,
    expiration: Optional[str] = None,
    current_price: Optional[float] = None,
) -> List[Dict[str, object]]:
    return [
        candidate
        for candidate in _evaluate_bull_put_spreads(puts_df, ticker=ticker, expiration=expiration, current_price=current_price)
        if candidate.get("accepted")
    ]


def _rank_bull_put_candidates(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        [c for c in candidates if c.get("accepted")],
        key=lambda item: (
            -float(item.get("return_on_risk", 0.0)),
            -float(item.get("estimated_credit", 0.0)),
        ),
    )


def _closest_failed_candidates(candidates: List[Dict[str, object]], limit: int = 5) -> List[Dict[str, object]]:
    def failure_distance(candidate: Dict[str, object]) -> tuple:
        delta = abs(float(candidate.get("short_delta") or 0.0))
        if delta < 0.15:
            delta_gap = 0.15 - delta
        elif delta > 0.20:
            delta_gap = delta - 0.20
        else:
            delta_gap = 0.0
        credit = float(candidate.get("estimated_credit") or 0.0)
        credit_gap = 0.0 if credit > 0.0 else abs(credit) + 1.0
        return (
            delta_gap,
            credit_gap,
            -credit,
        )

    failed_candidates = [c for c in candidates if not c.get("accepted")]
    return sorted(failed_candidates, key=failure_distance)[:limit]


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
        expirations = _get_expirations_from_ticker(stock)
        if not expirations:
            raise ValueError(f"No option expirations available for {ticker}")

        selected_expiration = _select_expiration(expirations)
        if not selected_expiration:
            raise ValueError(f"No valid option expiration selected for {ticker}")

        current_price = _current_price_from_yfinance(ticker)
        try:
            _, puts_df = _normalize_option_chain_payload(stock.option_chain(selected_expiration))
        except Exception:
            raise ValueError(f"No valid Bull Put Spread candidates available for {ticker}")

        if puts_df is None:
            raise ValueError(f"No valid Bull Put Spread candidates available for {ticker}")

        candidates = _build_bull_put_spread_candidates(
            puts_df, ticker=ticker, expiration=selected_expiration, current_price=current_price
        )
        for c in candidates:
            c["expiration"] = selected_expiration

        ranked = _rank_bull_put_candidates(candidates)
        if ranked:
            best = ranked[0]
            return {
                "ticker": ticker,
                "strategy": strategy,
                "expiration": selected_expiration,
                "short_strike": best["short_strike"],
                "long_strike": best["long_strike"],
                "width": 5.0,
                "estimated_credit": best["estimated_credit"],
                "max_risk": best["max_risk"],
                "short_delta": best["short_delta"],
                "probability_of_profit": best.get("probability_of_profit"),
                "return_on_risk": best["return_on_risk"],
            }

        closest_candidates = _closest_failed_candidates(candidates, limit=5)
        return {
            "ticker": ticker,
            "strategy": strategy,
            "expiration": selected_expiration,
            "message": f"No valid Bull Put Spread candidates available for {ticker}",
            "closest_candidates": closest_candidates,
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "strategy": strategy,
            "message": f"Unable to build live options spread: {exc}",
        }


def diagnose_bull_put_candidates(ticker: str) -> Dict[str, object]:
    """Return diagnostic information for all 5-width bull put candidates for a ticker.

    The returned dict contains the selected expiration and a list of candidate dicts with
    acceptance status and rejection reasons.
    """
    result: Dict[str, object] = {
        "ticker": ticker,
        "expiration": None,
        "candidates": [],
        "available_expirations": [],
        "per_expirations": [],
    }
    if not ticker:
        return result

    try:
        stock = yf.Ticker(ticker)
        expirations = _get_expirations_from_ticker(stock)
        result["available_expirations"] = expirations
        if not expirations:
            return result

        selected_expiration = _select_expiration(expirations)
        result["expiration"] = selected_expiration

        current_price = _current_price_from_yfinance(ticker)
        try:
            _, puts_df = _normalize_option_chain_payload(stock.option_chain(selected_expiration))
        except Exception:
            result["per_expirations"] = [{"expiration": selected_expiration, "error": "unable to retrieve option chain", "candidates": []}]
            return result

        if puts_df is None or puts_df.empty:
            result["per_expirations"] = [{"expiration": selected_expiration, "error": "no puts data for expiration", "candidates": []}]
            return result

        candidates = _build_bull_put_spread_candidates(
            puts_df, ticker=ticker, expiration=selected_expiration, current_price=current_price
        )
        for c in candidates:
            c["expiration"] = selected_expiration

        result["candidates"] = candidates
        result["per_expirations"] = [{"expiration": selected_expiration, "candidates": candidates}]
        return result
    except Exception:
        return result
