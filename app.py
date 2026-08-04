import pandas as pd
import streamlit as st

from data import TICKERS, download_watchlist_data, prepare_scan_results
from indicators import compute_indicator_metrics, make_price_chart, make_rsi_chart, make_volume_chart
from options_engine import build_trade

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - optional dependency fallback
    st_autorefresh = None

st.set_page_config(page_title="KMB Scanner", page_icon="📈", layout="wide")

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None


def get_selected_ticker_from_table(table_state, scan_results):
    if table_state is None:
        return None

    selection = getattr(table_state, "selection", None)
    if selection is None:
        return None

    rows = getattr(selection, "rows", None)
    if rows is None:
        return None

    if isinstance(rows, int):
        row_indexes = [rows]
    else:
        row_indexes = list(rows)

    if not row_indexes:
        return None

    row_index = row_indexes[0]
    if 0 <= row_index < len(scan_results):
        return scan_results.iloc[row_index]["Ticker"]
    return None


def slice_last_six_months(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if len(df.index) < 2:
        return df
    return df.iloc[-180:].copy()


def render_analysis_section(market_data, ticker, metrics):
    df = slice_last_six_months(market_data[ticker])
    st.markdown("---")
    st.subheader(f"Analysis for {ticker}")

    col_summary, col_price = st.columns([1, 2])
    with col_summary:
        st.metric("Current Price", f"{metrics['current_price']:.2f}")
        st.metric("Daily %", f"{metrics['daily_change']:.2f}%")
        st.metric("RSI(14)", f"{metrics['rsi14']:.1f}")
        st.metric("EMA20", f"{metrics['ema20']:.2f}")
        st.metric("EMA50", f"{metrics['ema50']:.2f}")
        st.metric("ATR14", f"{metrics['atr14']:.2f}")

    with col_price:
        st.plotly_chart(make_price_chart(df, ticker), use_container_width=True)

    volume_col, rsi_col = st.columns([1.2, 1])
    with volume_col:
        st.plotly_chart(make_volume_chart(df, ticker), use_container_width=True)
    with rsi_col:
        st.plotly_chart(make_rsi_chart(df, ticker), use_container_width=True)


def render_trade_builder(ticker: str, strategy: str):
    st.markdown("---")
    st.subheader("KMB Trade Builder")
    try:
        trade = build_trade(ticker, strategy)
    except Exception as exc:
        st.exception(exc)
        return

    trade_df = pd.DataFrame([trade])
    st.dataframe(trade_df, use_container_width=True, hide_index=True)


@st.cache_data(ttl=300)
def load_dashboard_data():
    market_data = download_watchlist_data(TICKERS)
    scan_results = prepare_scan_results(market_data)
    return market_data, scan_results


if st_autorefresh is not None:
    st_autorefresh(interval=300000, limit=None, key="scanner_refresh")
else:
    st.caption("Auto-refresh package is unavailable; use the refresh button to reload the data.")

col_title, col_refresh = st.columns([4, 1])
with col_title:
    st.title("KMB Scanner")
with col_refresh:
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.caption("Daily technical scanner built with yfinance, pandas, Plotly, and Streamlit.")

market_data, scan_results = load_dashboard_data()

if scan_results.empty:
    st.error("No market data was loaded. Check your internet connection and try again.")
    st.stop()

st.subheader("Watchlist scorecard")

score_display = scan_results.copy()
score_display["Status"] = score_display["Score"].apply(
    lambda value: "🟢 Strong" if value >= 80 else "🟡 Watch" if value >= 60 else "🔴 Weak"
)


def color_score(value: int):
    if value >= 80:
        return "background-color: #d4edda"
    if value >= 60:
        return "background-color: #fff3cd"
    return "background-color: #f8d7da"

table_state = st.dataframe(
    score_display[[
        "Ticker",
        "Score",
        "Strategy",
        "Status",
        "Current Price",
        "Daily %",
        "EMA20",
        "EMA50",
        "RSI14",
        "ATR14",
        "Average Volume",
        "Distance from EMA20",
        "Distance from EMA50",
    ]].style.map(color_score, subset=["Score"]),
    use_container_width=True,
    hide_index=True,
    key="score_table",
    on_select="rerun",
    selection_mode="single-row",
)

selected_ticker = get_selected_ticker_from_table(table_state, score_display)
if selected_ticker:
    st.session_state.selected_ticker = selected_ticker

if st.session_state.selected_ticker:
    ticker = st.session_state.selected_ticker
    if ticker not in market_data:
        st.warning("Ticker data is not available right now.")
        st.stop()

    metrics = compute_indicator_metrics(market_data[ticker])
    render_analysis_section(market_data, ticker, metrics)
    strategy = "Bull Put Spread" if metrics["current_price"] > metrics["ema20"] else "Bear Call Spread"
    render_trade_builder(ticker, strategy)
