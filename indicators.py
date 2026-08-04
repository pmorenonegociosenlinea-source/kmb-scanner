from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _coerce_series(values: object, *, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"Expected a single column for {name}, received {values.shape[1]} columns")
        series = values.iloc[:, 0].copy()
    else:
        raise TypeError(f"Expected a pandas Series or DataFrame for {name}")

    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    if series.empty:
        raise ValueError(f"No numeric values available for {name}")
    return series.astype(float)


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    series = _coerce_series(series, name="EMA input")
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = _coerce_series(series, name="RSI input")
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = _coerce_series(df["High"], name="High")
    low = _coerce_series(df["Low"], name="Low")
    close = _coerce_series(df["Close"], name="Close")
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def calculate_avg_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    volume = _coerce_series(df["Volume"], name="Volume")
    return volume.rolling(window=window).mean()


def compute_indicator_metrics(df: pd.DataFrame) -> dict:
    if df is None or getattr(df, "empty", True):
        raise ValueError("Indicator data is empty")

    close = _coerce_series(df["Close"], name="Close")
    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)
    rsi14 = calculate_rsi(close, 14)
    atr14 = calculate_atr(df, 14)
    avg_volume = calculate_avg_volume(df, 20)

    last_close_value = close.iloc[-1]
    previous_close_value = close.iloc[-2] if len(close) > 1 else last_close_value
    if isinstance(last_close_value, pd.Series):
        last_close_value = last_close_value.iloc[0]
    if isinstance(previous_close_value, pd.Series):
        previous_close_value = previous_close_value.iloc[0]
    last_close = float(last_close_value.item() if hasattr(last_close_value, "item") else last_close_value)
    previous_close = float(previous_close_value.item() if hasattr(previous_close_value, "item") else previous_close_value)

    last_atr = atr14.iloc[-1]
    last_atr_value = last_atr.item() if hasattr(last_atr, "item") else last_atr
    atr30_avg = float(atr14.rolling(window=30).mean().iloc[-1].item() if hasattr(atr14.rolling(window=30).mean().iloc[-1], "item") else atr14.rolling(window=30).mean().iloc[-1])

    return {
        "current_price": last_close,
        "daily_change": ((last_close / previous_close) - 1) * 100 if previous_close else 0.0,
        "ema20": float(ema20.iloc[-1].item() if hasattr(ema20.iloc[-1], "item") else ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1].item() if hasattr(ema50.iloc[-1], "item") else ema50.iloc[-1]),
        "rsi14": float(rsi14.iloc[-1].item() if hasattr(rsi14.iloc[-1], "item") else rsi14.iloc[-1]),
        "atr14": float(last_atr_value),
        "avg_volume": float(avg_volume.iloc[-1].item() if hasattr(avg_volume.iloc[-1], "item") else avg_volume.iloc[-1]),
        "distance_ema20": last_close - float(ema20.iloc[-1].item() if hasattr(ema20.iloc[-1], "item") else ema20.iloc[-1]),
        "distance_ema50": last_close - float(ema50.iloc[-1].item() if hasattr(ema50.iloc[-1], "item") else ema50.iloc[-1]),
        "previous_close": previous_close,
        "atr30_avg": atr30_avg,
    }


def make_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick",
        )
    )
    fig.add_trace(go.Scatter(x=df.index, y=calculate_ema(df["Close"], 20), mode="lines", name="EMA20", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df.index, y=calculate_ema(df["Close"], 50), mode="lines", name="EMA50", line=dict(color="purple")))
    fig.update_layout(title=f"{ticker} Price", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    return fig


def make_rsi_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    rsi = calculate_rsi(df["Close"], 14)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=rsi, mode="lines", name="RSI14", line=dict(color="royalblue")))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title=f"{ticker} RSI14", xaxis_title="Date", yaxis_title="RSI", template="plotly_white")
    return fig


def make_atr_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    atr = calculate_atr(df, 14)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=atr, mode="lines", name="ATR14", line=dict(color="darkgreen")))
    fig.update_layout(title=f"{ticker} ATR14", xaxis_title="Date", yaxis_title="ATR", template="plotly_white")
    return fig


def make_volume_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="steelblue"))
    fig.update_layout(title=f"{ticker} Volume", xaxis_title="Date", yaxis_title="Volume", template="plotly_white")
    return fig
