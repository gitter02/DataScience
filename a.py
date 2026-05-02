"""
Primetrade.ai — Data Science Intern Assignment
Trader Performance vs Market Sentiment
Fully corrected pipeline
"""

import warnings, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ----------------------------
# SETUP
# ----------------------------
os.makedirs("charts", exist_ok=True)
os.makedirs("output", exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

PAL = {
    "Extreme Fear": "#C0392B",
    "Fear": "#E67E22",
    "Neutral": "#7F8C8D",
    "Greed": "#27AE60",
    "Extreme Greed": "#1ABC9C",
}

SIMPLE = {"Fear": "#E67E22", "Neutral": "#7F8C8D", "Greed": "#27AE60"}


def savefig(name):
    plt.savefig(f"charts/{name}.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved → charts/{name}.png")


# ----------------------------
# DATA LOADING
# ----------------------------
print("\nDATA PREPARATION\n")

sent = pd.read_csv("data/sentiment.csv")
trades = pd.read_csv("data/trades.csv")

print(f"Sentiment: {sent.shape}")
print(f"Trades: {trades.shape}")


# ----------------------------
# FIX DATE TYPES (IMPORTANT FIX)
# ----------------------------
trades["date"] = pd.to_datetime(
    trades["Timestamp IST"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
).dt.normalize()

sent["date"] = pd.to_datetime(
    sent["date"],
    errors="coerce"
).dt.normalize()


# ----------------------------
# SENTIMENT CLEANING
# ----------------------------
def simplify(x):
    if "Fear" in str(x):
        return "Fear"
    elif "Greed" in str(x):
        return "Greed"
    return "Neutral"


sent["sentiment"] = sent["classification"].apply(simplify)


# ----------------------------
# MERGE DATA
# ----------------------------
df = trades.merge(
    sent[["date", "classification", "value", "sentiment"]],
    on="date",
    how="inner"
)

df = df[df["Closed PnL"] != 0].copy()

print(f"Merged rows: {len(df):,}")
print(f"Date range: {df['date'].min()} → {df['date'].max()}")


# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
df["trade_side"] = df["Direction"].apply(
    lambda x: "Long" if "Long" in str(x) or "Buy" in str(x)
    else ("Short" if "Short" in str(x) or "Sell" in str(x) else "Other")
)


# ----------------------------
# DAILY SUMMARY (OUTPUT 2)
# ----------------------------
daily_summary = df.groupby("date").agg(
    trades=("Closed PnL", "count"),
    total_pnl=("Closed PnL", "sum"),
    avg_pnl=("Closed PnL", "mean"),
    win_rate=("Closed PnL", lambda x: (x > 0).mean())
).reset_index()

daily_summary.to_csv("output/daily_summary.csv", index=False)


# ----------------------------
# KEY METRICS (OUTPUT 1)
# ----------------------------
key_metrics = df.groupby("sentiment").agg(
    trades=("Closed PnL", "count"),
    total_pnl=("Closed PnL", "sum"),
    avg_pnl=("Closed PnL", "mean"),
    win_rate=("Closed PnL", lambda x: (x > 0).mean()),
    avg_size=("Size USD", "mean"),
    long_ratio=("trade_side", lambda x: (x == "Long").mean())
).reindex(["Fear", "Neutral", "Greed"])

key_metrics.to_csv("output/key_metrics.csv")


# ----------------------------
# CHART 1: DISTRIBUTION
# ----------------------------
clip_val = df["Closed PnL"].quantile(0.97)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

data_bp = [
    df[df["sentiment"] == s]["Closed PnL"].clip(-clip_val, clip_val)
    for s in ["Fear", "Neutral", "Greed"]
]

bp = axes[0].boxplot(data_bp, patch_artist=True)

for patch, s in zip(bp["boxes"], ["Fear", "Neutral", "Greed"]):
    patch.set_facecolor(SIMPLE[s])

axes[0].set_xticklabels(["Fear", "Neutral", "Greed"])
axes[0].set_title("PnL Distribution")

for s in ["Fear", "Neutral", "Greed"]:
    axes[1].hist(
        df[df["sentiment"] == s]["Closed PnL"],
        bins=80,
        alpha=0.5,
        label=s
    )

axes[1].legend()

plt.tight_layout()
savefig("01_pnl_distribution")


# ----------------------------
# CHART 2: PERFORMANCE
# ----------------------------
perf = df.groupby("sentiment").agg(
    win_rate=("Closed PnL", lambda x: (x > 0).mean()),
    avg_pnl=("Closed PnL", "mean")
).reindex(["Fear", "Neutral", "Greed"])

fig, ax = plt.subplots()
ax.bar(perf.index, perf["win_rate"] * 100)
ax.set_title("Win Rate (%)")
savefig("02_winrate")


# ----------------------------
# CHART 3: SIZE
# ----------------------------
size = df.groupby("sentiment")["Size USD"].mean().reindex(["Fear", "Neutral", "Greed"])

fig, ax = plt.subplots()
ax.bar(size.index, size.values)
ax.set_title("Avg Trade Size")
savefig("03_size")


# ----------------------------
# CHART 4: LONG/SHORT
# ----------------------------
ls = df.groupby(["sentiment", "trade_side"]).size().unstack().fillna(0)
ls = ls.div(ls.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots()
ls.plot(kind="bar", ax=ax)
savefig("04_long_short")


# ----------------------------
# CHART 5: CUMULATIVE PNL
# ----------------------------
daily = df.groupby("date")["Closed PnL"].sum().reset_index()
daily["cum"] = daily["Closed PnL"].cumsum()

fig, ax = plt.subplots()
ax.plot(daily["date"], daily["cum"])
savefig("05_cumulative_pnl")


# ----------------------------
# CHART 6: COINS
# ----------------------------
coin = df.groupby("Coin")["Closed PnL"].sum().sort_values().tail(15)

fig, ax = plt.subplots()
coin.plot(kind="barh", ax=ax)
savefig("06_coin_pnl")


# ----------------------------
# CHART 7: TRADERS
# ----------------------------
traders = df.groupby("Account").agg(
    pnl=("Closed PnL", "sum"),
    trades=("Closed PnL", "count")
)

fig, ax = plt.subplots()
ax.scatter(traders["trades"], traders["pnl"])
savefig("07_traders")


# ----------------------------
# CHART 8: HEATMAP
# ----------------------------
heat = df.groupby(["Direction", "sentiment"])["Closed PnL"].mean().unstack()

fig, ax = plt.subplots()
sns.heatmap(heat, cmap="RdYlGn", center=0, ax=ax)
savefig("08_heatmap")


# ----------------------------
# DONE
# ----------------------------
print("\nDONE")
print("Outputs saved:")
print("- output/key_metrics.csv")
print("- output/daily_summary.csv")
print("- charts/ (8 images)")