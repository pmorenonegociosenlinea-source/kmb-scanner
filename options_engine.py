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


def _build_bull_put_spread_candidates(
    puts_df: pd.DataFrame,
    ticker: Optional[str] = None,
    expiration: Optional[str] = None,
    current_price: Optional[float] = None,
) -> List[Dict[str, float]]:
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

        # Determine current price and days to expiration for delta estimation
        cp = current_price
        if cp is None and ticker:
            cp = _current_price_from_yfinance(ticker)
        if cp is None:
            # Fallback to using short strike as proxy (keeps estimator deterministic)
            cp = short_strike

        if expiration:
            try:
                dte = _days_to_expiration(pd.Timestamp(expiration))
            except Exception:
                dte = 45
        else:
            dte = 45

        # Get implied vol if present; try multiple keys
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

        # If iv_val appears to be IV Rank (0-1) convert roughly to vol by scaling
        if 0.0 <= iv_val <= 1.0:
            implied_vol = 0.2 + iv_val * 0.4  # map ivRank 0-1 to vol 20%-60%
        else:
            implied_vol = iv_val

        estimated_short_delta = _estimate_put_delta(cp, short_strike, implied_vol, dte)
        # store as negative value for puts; selection by absolute value

        # Require estimated short delta in required range
        if not (0.15 <= abs(estimated_short_delta) <= 0.20):
            continue

        # Require minimum estimated credit
        if metrics["estimated_credit"] < 0.90:
            continue

        # Extract IV Rank if present; fall back to 0.0
        iv_rank_raw = (
            short_row.get("ivRank")
            or short_row.get("iv_rank")
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
                "short_delta": float(estimated_short_delta),
                "probability_of_profit": metrics["probability_of_profit"],
                "iv_rank": iv_rank_value,
            }
        )

    # Rank by: 1) Highest return on risk, 2) Highest estimated credit, 3) Highest probability of profit
    candidates.sort(
        key=lambda item: (
            -float(item.get("return_on_risk", 0.0)),
            -float(item.get("estimated_credit", 0.0)),
            -float(item.get("probability_of_profit") or 0.0),
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
        expirations = _get_expirations_from_ticker(stock)
        if not expirations:
            raise ValueError(f"No option expirations available for {ticker}")

        # Evaluate all expirations and collect candidates across them
        all_candidates: List[Dict[str, object]] = []
        current_price = _current_price_from_yfinance(ticker)
        for exp in expirations:
            try:
                chain_data = stock.option_chain(exp)
            except Exception:
                continue
            calls_df, puts_df = _normalize_option_chain_payload(chain_data)
            if puts_df is None or puts_df is None:
                continue

            candidates = _build_bull_put_spread_candidates(
                puts_df, ticker=ticker, expiration=exp, current_price=current_price
            )
            for c in candidates:
                c["expiration"] = exp
            all_candidates.extend(candidates)

        # After evaluating all expirations, apply final ranking and pick best
        if strategy == "Bull Put Spread":
            if not all_candidates:
                raise ValueError(f"No valid Bull Put Spread candidates available for {ticker}")
            # Already sorted within each call; sort final list by required ranking
            all_candidates.sort(
                key=lambda item: (
                    -float(item.get("return_on_risk", 0.0)),
                    -float(item.get("estimated_credit", 0.0)),
                    -float(item.get("probability_of_profit") or 0.0),
                )
            )
            best = all_candidates[0]
            return {
                "ticker": ticker,
                "strategy": strategy,
                "expiration": best.get("expiration"),
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


def diagnose_bull_put_candidates(ticker: str) -> Dict[str, object]:
    """Return diagnostic information for all 5-width bull put candidates for a ticker.

    The returned dict contains the selected expiration and a list of candidate dicts with
    acceptance status and rejection reasons.
    """
    result: Dict[str, object] = {"ticker": ticker, "expiration": None, "candidates": []}
    if not ticker:
        return result

    try:
        stock = yf.Ticker(ticker)
        expirations = _get_expirations_from_ticker(stock)
        result["available_expirations"] = expirations
        if not expirations:
            return result

        # Evaluate each expiration and collect per-expiration diagnostics
        # determine a selected expiration for quick reference and keep per-expiration details
        selected_expiration = _select_expiration(expirations)
        result["expiration"] = selected_expiration

        per_expirations: List[Dict[str, object]] = []
        for exp in expirations:
            exp_entry: Dict[str, object] = {"expiration": exp, "candidates": []}
            try:
                chain_data = stock.option_chain(exp)
            except Exception:
                exp_entry["error"] = "unable to retrieve option chain"
                per_expirations.append(exp_entry)
                continue

            calls_df, puts_df = _normalize_option_chain_payload(chain_data)
            if puts_df is None or puts_df.empty:
                exp_entry["error"] = "no puts data for expiration"
                per_expirations.append(exp_entry)
                continue

            prepared = puts_df.copy()
            prepared["strike"] = pd.to_numeric(prepared.get("strike"), errors="coerce")
            prepared = prepared.dropna(subset=["strike"]).sort_values(by=["strike"], ascending=True)

            for index, short_row in prepared.iterrows():
                short_strike = float(short_row["strike"])
                long_strike = short_strike - 5.0
                long_row = prepared[prepared["strike"] == long_strike]
                width = short_strike - long_strike

                candidate: Dict[str, object] = {
                    "short_strike": round(short_strike, 2),
                    "long_strike": round(long_strike, 2) if not long_row.empty else None,
                    "width": round(width, 2),
                    "estimated_credit": None,
                    "estimated_short_delta": None,
                    "accepted": False,
                    "rejection_reasons": [],
                }

                if long_row.empty:
                    candidate["rejection_reasons"].append("no matching 5-width long strike")
                    exp_entry["candidates"].append(candidate)
                    continue

                metrics = _build_spread_metrics(short_row, long_row.iloc[0], width)
                candidate["estimated_credit"] = metrics.get("estimated_credit")
                candidate["return_on_risk"] = metrics.get("return_on_risk")
                candidate["probability_of_profit"] = metrics.get("probability_of_profit")

                # Estimate short delta using current price, implied vol and DTE
                cp = _current_price_from_yfinance(ticker) or short_strike
                try:
                    dte = _days_to_expiration(pd.Timestamp(exp))
                except Exception:
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

                est_delta = _estimate_put_delta(cp, short_strike, implied_vol, dte)
                candidate["estimated_short_delta"] = est_delta
                # Delta rejection categorization
                if abs(est_delta) < 0.15:
                    candidate["rejection_reasons"].append("delta too low")
                elif abs(est_delta) > 0.20:
                    candidate["rejection_reasons"].append("delta too high")

                # Estimated credit check
                credit_val = candidate.get("estimated_credit") or 0.0
                if credit_val <= 0.0:
                    candidate["rejection_reasons"].append("insufficient estimated credit")
                elif credit_val < 0.9:
                    candidate["rejection_reasons"].append("credit too low")

                # Return-on-risk informational note (not used to reject candidates)
                ror = candidate.get("return_on_risk") or 0.0
                if ror < 0.25:
                    candidate["rejection_reasons"].append("return on risk too low")

                # Acceptance: must have matching long strike and pass delta+credit thresholds
                # We treat 'return on risk too low' as informational only
                hard_reasons = [r for r in candidate["rejection_reasons"] if r not in {"return on risk too low"}]
                if not hard_reasons:
                    candidate["accepted"] = True

                exp_entry["candidates"].append(candidate)

            per_expirations.append(exp_entry)

        result["per_expirations"] = per_expirations
        return result
    except Exception:
        return result
