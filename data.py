from __future__ import annotations

from typing import Dict, List

import pandas as pd
import yfinance as yf

from indicators import compute_indicator_metrics
from scoring import calculate_score
from strategy_engine import get_strategy

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

TICKERS: List[str] = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "GLD",
    "XLK",
    "XLF",
    "XLE",
    "SMH",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
]


def _normalize_ohlcv_frame(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw_df is None or getattr(raw_df, "empty", True):
        raise ValueError("Downloaded data is empty")

    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(1):
            df = df.xs(ticker, axis=1, level=1, drop_level=True)
        else:
            df = df.copy()
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError(f"Unable to flatten columns for {ticker}")

    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={"Adj Close": "Adj Close"})

    if isinstance(df.columns, pd.Index):
        normalized = {}
        for column in REQUIRED_COLUMNS:
            if column in df.columns:
                normalized[column] = df[column]
            elif column.lower() in {name.lower() for name in df.columns}:
                match_name = next(name for name in df.columns if name.lower() == column.lower())
                normalized[column] = df[match_name]

        if set(normalized) != set(REQUIRED_COLUMNS):
            missing = [column for column in REQUIRED_COLUMNS if column not in normalized]
            raise ValueError(f"Missing expected OHLCV columns for {ticker}: {missing}")

        normalized_df = pd.DataFrame(normalized)
        normalized_df = normalized_df.dropna(how="all")
        if normalized_df.empty:
            raise ValueError(f"Normalized data is empty for {ticker}")
        return normalized_df

    raise ValueError(f"Unexpected column format for {ticker}")


def download_ticker_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    raw_df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if raw_df is None or getattr(raw_df, "empty", True):
        raise ValueError(f"No data available for {ticker}")

    df = _normalize_ohlcv_frame(raw_df, ticker)
    if len(df) < 2:
        raise ValueError(f"Not enough rows for {ticker}")
    return df


def download_watchlist_data(tickers: List[str], period: str = "2y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    market_data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            market_data[ticker] = download_ticker_data(ticker, period=period, interval=interval)
        except Exception:
            continue
    return market_data


def prepare_scan_results(market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, df in market_data.items():
        metrics = compute_indicator_metrics(df)
        score = calculate_score(metrics)
        strategy = get_strategy(metrics)
        rows.append(
            {
                "Ticker": ticker,
                "Score": score,
                "Strategy": strategy,
                "Current Price": metrics["current_price"],
                "Daily %": metrics["daily_change"],
                "EMA20": metrics["ema20"],
                "EMA50": metrics["ema50"],
                "RSI14": metrics["rsi14"],
                "ATR14": metrics["atr14"],
                "Average Volume": metrics["avg_volume"],
                "Distance from EMA20": metrics["distance_ema20"],
                "Distance from EMA50": metrics["distance_ema50"],
            }
        )

    if not rows:
        return pd.DataFrame(columns=[
            "Ticker",
            "Score",
            "Strategy",
            "Current Price",
            "Daily %",
            "EMA20",
            "EMA50",
            "RSI14",
            "ATR14",
            "Average Volume",
            "Distance from EMA20",
            "Distance from EMA50",
        ])

    scan_results = pd.DataFrame(rows)
    scan_results = scan_results.sort_values("Score", ascending=False).reset_index(drop=True)
    return scan_results
