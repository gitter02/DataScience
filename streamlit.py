import streamlit as st
import pandas as pd

st.set_page_config(page_title="Trader Performance Dashboard", layout="wide")

st.title("Trader Performance vs Market Sentiment")
st.subheader("Hyperliquid × Fear & Greed Analysis")

# Load files
summary = pd.read_csv("output/key_metrics.csv")
daily = pd.read_csv("output/daily_summary.csv")

# KPIs
st.header("Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Closing Trades", "104,402")
col2.metric("Accounts Analyzed", "32")
col3.metric("Date Range", "2023–2025")

st.divider()

# Key metrics table
st.header("Performance by Sentiment")

st.dataframe(summary, use_container_width=True)

st.divider()

# Charts
st.header("Key Visual Insights")

st.image("charts/02_winrate_avgpnl.png", caption="Win Rate and Avg PnL")
st.image("charts/04_long_short_bias.png", caption="Long vs Short Bias")
st.image("charts/03_size_frequency.png", caption="Position Size and Trade Frequency")
st.image("charts/07_trader_segments.png", caption="Trader Segments")

st.divider()

# Strategy Recommendations
st.header("Strategy Recommendations")

st.markdown("""
### Rule 1 — Buy during Fear, avoid chasing Greed

Fear days showed the highest win rate and larger long positions.
This suggests panic creates stronger opportunities than greed.

### Rule 2 — Use larger size only during high-volatility Fear regimes

During Fear, average position size was much higher.
During Greed, smaller and more precise trades worked better.
""")