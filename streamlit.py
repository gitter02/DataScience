import streamlit as st
import pandas as pd

st.set_page_config(page_title="Trader Sentiment Dashboard", layout="wide")

# ----------------------------
# LOAD DATA
# ----------------------------
key_metrics = pd.read_csv("output/key_metrics.csv", index_col=0)
daily_summary = pd.read_csv("output/daily_summary.csv")

daily_summary["date"] = pd.to_datetime(daily_summary["date"])

# ----------------------------
# YEARLY AGGREGATION 
# ----------------------------
yearly_summary = (
    daily_summary
    .set_index("date")
    .resample("YE")
    .sum()
)

yearly_summary.index = yearly_summary.index.year

# ----------------------------
# TITLE
# ----------------------------
st.title("Trader Performance vs Market Sentiment")

st.markdown(
    "Analysis of trading performance across Fear, Neutral, and Greed market conditions."
)

# ----------------------------
# KEY METRICS
# ----------------------------
st.subheader("Key Metrics by Sentiment")
st.dataframe(key_metrics, use_container_width=True)

# ----------------------------
# YEARLY PERFORMANCE (CLEAN X-AXIS)
# ----------------------------
st.subheader("Yearly Performance Trends")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Total PnL by Year")
    st.line_chart(yearly_summary["total_pnl"])

with col2:
    st.markdown("Trade Volume by Year")
    st.line_chart(yearly_summary["trades"])

# ----------------------------
# WIN RATE TREND (YEARLY)
# ----------------------------
st.subheader("Win Rate Trend (Yearly)")
st.line_chart(yearly_summary["win_rate"])

# ----------------------------
# SENTIMENT INSIGHTS
# ----------------------------
st.subheader("Key Insights")

best_sentiment = key_metrics["total_pnl"].idxmax()
worst_sentiment = key_metrics["total_pnl"].idxmin()

st.write(f"- Best performing sentiment: {best_sentiment}")
st.write(f"- Worst performing sentiment: {worst_sentiment}")
st.write(f"- Average win rate: {key_metrics['win_rate'].mean():.2f}")