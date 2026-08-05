import pandas as pd
import streamlit as st

from data import TICKERS, download_watchlist_data, prepare_scan_results
from indicators import compute_indicator_metrics, make_price_chart, make_rsi_chart, make_volume_chart
from options_engine import build_trade
from options_engine import diagnose_bull_put_candidates

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


def _render_bull_put_diagnostics(scan_results: pd.DataFrame):
    bull_puts = scan_results[scan_results["Strategy"] == "Bull Put Spread"]
    if bull_puts.empty:
        st.markdown("**Diagnostics**\nNo tickers with Bull Put Spread strategy to diagnose.")
        return

    st.markdown("---")
    st.subheader("Diagnostics: Bull Put Candidates")
    for _, row in bull_puts.iterrows():
        ticker = row["Ticker"]
        diag = diagnose_bull_put_candidates(ticker)
        st.markdown(f"**{ticker}**")
        expiration = diag.get("expiration") or "N/A"
        st.markdown(f"Expiration: {expiration}")
        # Show all available expirations returned by yfinance, if any
        avail = diag.get("available_expirations")
        if avail:
            expirations_text = ", ".join(str(x) for x in avail)
            st.markdown(f"Available expirations: {expirations_text}")
        else:
            st.markdown("Available expirations: No expirations returned by yfinance")
        candidates = diag.get("candidates", [])
        if not candidates:
            st.markdown("No candidates found for this ticker.")
            continue

        for c in candidates:
            short = c.get("short_strike")
            delta = c.get("short_delta")
            credit = c.get("estimated_credit")
            width = c.get("width")
            accepted = c.get("accepted")
            reasons = c.get("rejection_reasons") or []

            if accepted:
                st.markdown(f"- {short}: Accepted")
            else:
                if reasons:
                    reason_text = ", ".join(reasons)
                else:
                    reason_text = "Rejected"
                detail = f"- {short}: Rejected: {reason_text}"
                if delta is not None:
                    detail += f" (short delta = {delta:.2f})"
                st.markdown(detail)

        # If there is a selected expiration, show a detailed debug table for every strike evaluated
        sel_exp = diag.get("expiration")
        if sel_exp:
            # find per_expirations entry
            per = diag.get("per_expirations") or []
            sel_entry = next((e for e in per if e.get("expiration") == sel_exp), None)
            if sel_entry:
                rows = []
                for c in sel_entry.get("candidates", []):
                    status = "Accepted" if c.get("accepted") else "Rejected"
                    reasons = c.get("rejection_reasons") or []
                    rows.append(
                        {
                            "Strike": c.get("short_strike"),
                            "Bid": c.get("raw_bid"),
                            "Ask": c.get("raw_ask"),
                            "Mid Price": c.get("mid_price"),
                            "Delta": c.get("estimated_short_delta") if c.get("estimated_short_delta") is not None else c.get("raw_delta"),
                            "Estimated Credit": c.get("estimated_credit"),
                            "Return on Risk": c.get("return_on_risk"),
                            "Status": status,
                            "Rejection Reason": ", ".join(reasons) if reasons else "",
                        }
                    )
                if rows:
                    df_debug = pd.DataFrame(rows)
                    st.markdown("**Debug: evaluated strikes for selected expiration**")
                    st.dataframe(df_debug, use_container_width=True)


def _star_rating_from_score(score: int) -> str:
    if score >= 80:
        stars = 5
    elif score >= 60:
        stars = 4
    elif score >= 40:
        stars = 3
    elif score >= 20:
        stars = 2
    else:
        stars = 1
    return "★" * stars + "☆" * (5 - stars)


def render_trade_of_the_day_card(scan_results: pd.DataFrame):
    bull_puts = scan_results[scan_results["Strategy"] == "Bull Put Spread"]
    if bull_puts.empty:
        st.info("No valid trade today")
        # Show diagnostics for all tickers that have strategy Bull Put Spread in the scan
        _render_bull_put_diagnostics(scan_results)
        return

    top_candidate = bull_puts.iloc[0]
    ticker = top_candidate["Ticker"]
    score = int(top_candidate["Score"])
    strategy = top_candidate["Strategy"]

    trade = build_trade(ticker, strategy)
    if "message" in trade:
        st.info("No valid trade today")
        # Diagnostics for all bull put tickers
        _render_bull_put_diagnostics(scan_results)
        return

    st.markdown("---")
    st.subheader("Trade of the Day")
    card_columns = st.columns([1, 1, 1, 1])
    with card_columns[0]:
        st.markdown(f"**Ticker**\n{ticker}")
        st.markdown(f"**Strategy**\n{strategy}")
        st.markdown(f"**Expiration**\n{trade.get('expiration', 'N/A')}")
    with card_columns[1]:
        st.markdown(f"**Short Strike**\n{trade.get('short_strike', 'N/A')}")
        st.markdown(f"**Long Strike**\n{trade.get('long_strike', 'N/A')}")
    with card_columns[2]:
        st.markdown(f"**Estimated Credit**\n${trade.get('estimated_credit', 'N/A')}")
        st.markdown(f"**Max Risk**\n${trade.get('max_risk', 'N/A')}")
    with card_columns[3]:
        return_on_risk = trade.get('return_on_risk')
        return_on_risk_text = f"{return_on_risk:.2f}" if isinstance(return_on_risk, (int, float)) else "N/A"
        st.markdown(f"**Return on Risk**\n{return_on_risk_text}")
        st.markdown(f"**Rating**\n{_star_rating_from_score(score)}")


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

render_trade_of_the_day_card(scan_results)

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
