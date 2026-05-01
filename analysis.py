"""
Primetrade.ai — Data Science Intern Assignment
Trader Performance vs Market Sentiment

"""

import warnings, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

os.makedirs("charts", exist_ok=True)
os.makedirs("output", exist_ok=True)

PAL = {
    "Extreme Fear": "#C0392B",
    "Fear":         "#E67E22",
    "Neutral":      "#7F8C8D",
    "Greed":        "#27AE60",
    "Extreme Greed":"#1ABC9C",
}
ORDER  = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SIMPLE = {"Fear": "#E67E22", "Neutral": "#7F8C8D", "Greed": "#27AE60"}

def savefig(name):
    plt.savefig(f"charts/{name}.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  saved → charts/{name}.png")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━  DATA PREPARATION  ━━━\n")

sent   = pd.read_csv("data/sentiment.csv", parse_dates=["date"])
trades = pd.read_csv("data/trades.csv")

print(f"Sentiment  : {sent.shape[0]:,} rows × {sent.shape[1]} columns")
print(f"Trades     : {trades.shape[0]:,} rows × {trades.shape[1]} columns")
print(f"\nTrades columns: {list(trades.columns)}")
print(f"\nMissing values (trades):\n{trades.isnull().sum()[trades.isnull().sum()>0]}")
print(f"\nDuplicate rows — Sentiment: {sent.duplicated().sum()}")
print(f"Duplicate rows — Trades   : {trades.duplicated().sum()}")

# Timestamp
trades["date"] = pd.to_datetime(
    trades["Timestamp IST"], format="%d-%m-%Y %H:%M"
).dt.normalize()

# Sentiment simplification
def simplify(c):
    return "Fear" if "Fear" in c else ("Greed" if "Greed" in c else "Neutral")

sent["sentiment"] = sent["classification"].apply(simplify)

# Merge
trades = trades.merge(
    sent[["date","classification","value","sentiment"]],
    on="date", how="inner"
)
print(f"\nAligned rows : {len(trades):,}")
print(f"Date range   : {trades['date'].min().date()} → {trades['date'].max().date()}")
print(f"Unique accounts: {trades['Account'].nunique()}")

# Trade-side from Direction column
def trade_side(d):
    if any(x in d for x in ["Long","Buy"]):   return "Long"
    if any(x in d for x in ["Short","Sell"]): return "Short"
    return "Other"

trades["trade_side"] = trades["Direction"].apply(trade_side)

# Closing trades only (realized PnL)
closing = trades[trades["Closed PnL"] != 0].copy()
print(f"\nClosing trades (non-zero PnL): {len(closing):,}")
print(f"Win rate on closing trades   : {(closing['Closed PnL']>0).mean()*100:.1f}%")

# Per-account stats
trader_stats = (
    closing.groupby("Account")
    .agg(
        total_pnl = ("Closed PnL","sum"),
        win_rate  = ("Closed PnL", lambda x: (x>0).mean()),
        n_trades  = ("Closed PnL","count"),
        avg_size  = ("Size USD",  "mean"),
        total_fee = ("Fee",       "sum"),
    ).reset_index()
)
trader_stats["size_seg"] = trader_stats["avg_size"].apply(
    lambda x: "Large Position" if x >= trader_stats["avg_size"].median() else "Small Position"
)
trader_stats["freq_seg"] = trader_stats["n_trades"].apply(
    lambda x: "Frequent" if x >= trader_stats["n_trades"].median() else "Infrequent"
)
trader_stats["pnl_seg"] = trader_stats["total_pnl"].apply(
    lambda x: "Winner" if x >= trader_stats["total_pnl"].median() else "Loser"
)

# Daily market metrics
daily_mkt = (
    closing.groupby(["date","sentiment","classification","value"])
    .agg(
        avg_pnl   = ("Closed PnL","mean"),
        total_pnl = ("Closed PnL","sum"),
        n_trades  = ("Closed PnL","count"),
        n_traders = ("Account",   "nunique"),
        avg_size  = ("Size USD",  "mean"),
        long_ratio= ("trade_side",lambda x: (x=="Long").sum()/len(x)),
    ).reset_index()
)

summary = (
    closing.groupby("sentiment")
    .agg(
        n_trades  = ("Closed PnL","count"),
        avg_pnl   = ("Closed PnL","mean"),
        win_rate  = ("Closed PnL",lambda x: (x>0).mean()),
        total_pnl = ("Closed PnL","sum"),
        avg_size  = ("Size USD",  "mean"),
        long_ratio= ("trade_side",lambda x: (x=="Long").sum()/len(x)),
    )
    .reindex(["Fear","Neutral","Greed"])
    .round(3)
)
summary.columns = ["# Trades","Avg PnL ($)","Win Rate","Total PnL ($)","Avg Size ($)","Long Ratio"]
print("\n── Summary by sentiment ──")
print(summary.to_string())
summary.to_csv("output/daily_summary.csv")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━ ANALYSIS  ━━━\n")

# Chart 1: PnL distribution
print("[Chart 1] PnL distribution …")
clip_val = closing["Closed PnL"].quantile(0.97)
fig, axes = plt.subplots(1,2,figsize=(13,5))
fig.suptitle("Closed PnL by Market Sentiment — Hyperliquid Traders",
             fontsize=14, fontweight="bold", y=1.01)

data_bp = [closing.loc[closing["sentiment"]==s,"Closed PnL"].clip(-clip_val,clip_val)
           for s in ["Fear","Neutral","Greed"]]
bp = axes[0].boxplot(data_bp, patch_artist=True, widths=0.45,
                     medianprops=dict(color="white",linewidth=2.5),
                     flierprops=dict(marker=".",alpha=0.15,markersize=3))
for patch,s in zip(bp["boxes"],["Fear","Neutral","Greed"]):
    patch.set_facecolor(SIMPLE[s]); patch.set_alpha(0.85)
axes[0].set_xticklabels(["Fear","Neutral","Greed"],fontsize=12)
axes[0].set_ylabel("Closed PnL (USD)")
axes[0].set_title("Distribution per trade (97th pct clipped)")
axes[0].axhline(0,color="#666",linestyle="--",linewidth=0.8)

for s,col in SIMPLE.items():
    d = closing.loc[closing["sentiment"]==s,"Closed PnL"].clip(-clip_val,clip_val)
    axes[1].hist(d,bins=80,alpha=0.5,color=col,label=s,density=True)
axes[1].axvline(0,color="#555",linestyle="--",linewidth=0.8)
axes[1].set_xlabel("Closed PnL (USD)"); axes[1].set_ylabel("Density")
axes[1].set_title("PnL density by sentiment"); axes[1].legend()
plt.tight_layout(); savefig("01_pnl_by_sentiment")

# Chart 2: Win rate & avg PnL
print("[Chart 2] Win rate & avg PnL …")
perf = (
    closing.groupby("sentiment")
    .agg(win_rate=("Closed PnL",lambda x:(x>0).mean()),
         avg_pnl =("Closed PnL","mean"))
    .reindex(["Fear","Neutral","Greed"]).reset_index()
)
fig,axes = plt.subplots(1,2,figsize=(11,4.5))
fig.suptitle("Performance Metrics — Fear vs Greed",fontsize=14,fontweight="bold")
bars = axes[0].bar(perf["sentiment"],perf["win_rate"]*100,
                   color=[SIMPLE[s] for s in perf["sentiment"]],width=0.45,alpha=0.85)
axes[0].bar_label(bars,fmt="%.1f%%",padding=3,fontsize=11)
axes[0].set_ylabel("Win Rate (%)"); axes[0].set_title("Win rate by sentiment")
axes[0].set_ylim(0,100); axes[0].axhline(50,color="#bbb",linestyle="--",linewidth=0.8)
bars2 = axes[1].bar(perf["sentiment"],perf["avg_pnl"],
                    color=[SIMPLE[s] for s in perf["sentiment"]],width=0.45,alpha=0.85)
axes[1].bar_label(bars2,fmt="$%.1f",padding=3,fontsize=11)
axes[1].set_ylabel("Avg PnL per closing trade (USD)")
axes[1].set_title("Average PnL by sentiment")
axes[1].axhline(0,color="#555",linestyle="--",linewidth=0.8)
plt.tight_layout(); savefig("02_winrate_avgpnl")

# Chart 3: Trade size & frequency
print("[Chart 3] Trade size & frequency …")
fig,axes = plt.subplots(1,2,figsize=(12,4.5))
fig.suptitle("Trader Behavior — Fear vs Greed",fontsize=14,fontweight="bold")
p95 = closing["Size USD"].quantile(0.95)
size_data = [closing.loc[closing["sentiment"]==s,"Size USD"].clip(upper=p95)
             for s in ["Fear","Neutral","Greed"]]
vp = axes[0].violinplot(size_data,positions=[1,2,3],showmedians=True,showextrema=False)
for body,s in zip(vp["bodies"],["Fear","Neutral","Greed"]):
    body.set_facecolor(SIMPLE[s]); body.set_alpha(0.72)
vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)
axes[0].set_xticks([1,2,3]); axes[0].set_xticklabels(["Fear","Neutral","Greed"],fontsize=12)
axes[0].set_ylabel("Size USD (95th pct clipped)"); axes[0].set_title("Position size by sentiment")
freq = daily_mkt.groupby("sentiment")["n_trades"].mean().reindex(["Fear","Neutral","Greed"])
bars3 = axes[1].bar(freq.index,freq.values,color=[SIMPLE[s] for s in freq.index],width=0.45,alpha=0.85)
axes[1].bar_label(bars3,fmt="%.0f",padding=3,fontsize=11)
axes[1].set_ylabel("Avg closing trades per day"); axes[1].set_title("Trade frequency by sentiment")
plt.tight_layout(); savefig("03_size_frequency")

# Chart 4: Long/short bias
print("[Chart 4] Long/short bias …")
ls_pct = (
    closing.groupby(["sentiment","trade_side"]).size()
    .unstack("trade_side").fillna(0)
    .div(closing.groupby("sentiment").size(),axis=0)*100
).reindex(["Fear","Neutral","Greed"])
fig,ax = plt.subplots(figsize=(8,4.5))
x = np.arange(3); w = 0.38
if "Long" in ls_pct.columns:
    b1 = ax.bar(x-w/2,ls_pct["Long"],w,label="Long",color="#3498DB",alpha=0.85)
    ax.bar_label(b1,fmt="%.1f%%",padding=2,fontsize=10)
if "Short" in ls_pct.columns:
    b2 = ax.bar(x+w/2,ls_pct["Short"],w,label="Short",color="#E74C3C",alpha=0.85)
    ax.bar_label(b2,fmt="%.1f%%",padding=2,fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(ls_pct.index,fontsize=12)
ax.set_ylabel("% of closing trades"); ax.set_ylim(0,80)
ax.set_title("Long vs Short Bias by Market Sentiment",fontsize=13,fontweight="bold")
ax.axhline(50,color="#aaa",linestyle="--",linewidth=0.8); ax.legend()
plt.tight_layout(); savefig("04_long_short_bias")

# Chart 5: Cumulative PnL timeline
print("[Chart 5] Cumulative PnL timeline …")
daily_total = closing.groupby("date")["Closed PnL"].sum().reset_index().sort_values("date")
daily_total["cum_pnl"] = daily_total["Closed PnL"].cumsum()
fig,ax = plt.subplots(figsize=(14,5))
for _,row in sent.iterrows():
    ax.axvspan(row["date"],row["date"]+pd.Timedelta("1D"),
               alpha=0.07,color=PAL.get(row["classification"],"#ccc"),linewidth=0)
ax.fill_between(daily_total["date"],daily_total["cum_pnl"],alpha=0.12,color="#2C3E50")
ax.plot(daily_total["date"],daily_total["cum_pnl"],color="#2C3E50",linewidth=1.5)
ax.axhline(0,color="#aaa",linestyle="--",linewidth=0.8)
patches = [mpatches.Patch(color=PAL[k],label=k,alpha=0.6) for k in ORDER]
ax.legend(handles=patches,fontsize=8,ncol=5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M" if abs(x)>=1e6 else f"${x:,.0f}"))
ax.set_xlabel("Date"); ax.set_ylabel("Cumulative PnL (USD)")
ax.set_title("Aggregate Trader PnL Over Time vs Sentiment Regime",fontsize=13,fontweight="bold")
plt.tight_layout(); savefig("05_cumulative_pnl_timeline")

# Chart 6: Top coins by PnL
print("[Chart 6] PnL by coin …")
coin_pnl = (
    closing.groupby("Coin")["Closed PnL"]
    .agg(["sum","mean","count"])
    .rename(columns={"sum":"Total PnL","mean":"Avg PnL","count":"Trades"})
    .sort_values("Total PnL",ascending=True).tail(15)
)
fig,ax = plt.subplots(figsize=(9,6))
colors = ["#E74C3C" if v<0 else "#27AE60" for v in coin_pnl["Total PnL"]]
ax.barh(coin_pnl.index,coin_pnl["Total PnL"],color=colors,alpha=0.85)
ax.axvline(0,color="#555",linewidth=0.8,linestyle="--")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x,_: f"${x/1e6:.1f}M" if abs(x)>=1e6 else f"${x:,.0f}"))
ax.set_title("Total Closed PnL by Coin (top 15)",fontsize=13,fontweight="bold")
ax.set_xlabel("Total Closed PnL (USD)")
plt.tight_layout(); savefig("06_pnl_by_coin")

# Chart 7: Trader segments
print("[Chart 7] Trader segments …")
fig,axes = plt.subplots(1,3,figsize=(15,5))
fig.suptitle("Trader Segment Comparison",fontsize=14,fontweight="bold")
for ax,seg,metric,ylabel,title in zip(
    axes,
    ["size_seg","freq_seg","pnl_seg"],
    ["total_pnl","total_pnl","win_rate"],
    ["Total PnL ($)","Total PnL ($)","Win Rate"],
    ["Seg 1: Position Size","Seg 2: Activity","Seg 3: Consistency"]
):
    g = trader_stats.groupby(seg)[metric].mean().sort_index()
    threshold = 0 if metric=="total_pnl" else 0.5
    clr = ["#E74C3C" if v<threshold else "#27AE60" for v in g.values]
    bars = ax.bar(g.index,g.values,color=clr,alpha=0.85,width=0.45)
    ax.bar_label(bars,fmt=("$%.0f" if metric!="win_rate" else "%.2f"),padding=3,fontsize=10)
    ax.axhline(0,color="#bbb",linestyle="--",linewidth=0.7)
    ax.set_title(title,fontsize=11); ax.set_ylabel(ylabel)
plt.tight_layout(); savefig("07_trader_segments")

# Chart 8: Direction × Sentiment heatmap
print("[Chart 8] Direction × Sentiment heatmap …")
main_dirs = ["Open Long","Close Long","Open Short","Close Short","Buy","Sell"]
heat_df = closing[closing["Direction"].isin(main_dirs)]
heat = (
    heat_df.groupby(["Direction","sentiment"])["Closed PnL"]
    .mean().unstack("sentiment")
    .reindex(columns=["Fear","Neutral","Greed"]).reindex(main_dirs)
)
fig,ax = plt.subplots(figsize=(8,5))
sns.heatmap(heat,annot=True,fmt=".1f",cmap="RdYlGn",center=0,
            linewidths=0.5,ax=ax,cbar_kws={"label":"Avg PnL/trade (USD)"})
ax.set_title("Avg PnL per Trade: Direction × Sentiment",fontsize=13,fontweight="bold")
ax.set_xlabel("Market Sentiment"); ax.set_ylabel("")
plt.tight_layout(); savefig("08_heatmap_direction_sentiment")

# Statistical tests
print("\n── Statistical Tests ──")
fear_pnl  = closing.loc[closing["sentiment"]=="Fear","Closed PnL"]
greed_pnl = closing.loc[closing["sentiment"]=="Greed","Closed PnL"]
stat,p = stats.mannwhitneyu(fear_pnl,greed_pnl,alternative="two-sided")
print(f"Fear vs Greed PnL  : U={stat:.0f}, p={p:.4f}  {'✓ significant' if p<0.05 else '✗ not significant'}")

fear_sz  = closing.loc[closing["sentiment"]=="Fear","Size USD"]
greed_sz = closing.loc[closing["sentiment"]=="Greed","Size USD"]
stat2,p2 = stats.mannwhitneyu(fear_sz,greed_sz,alternative="two-sided")
print(f"Fear vs Greed size : U={stat2:.0f}, p={p2:.4f}  {'✓ significant' if p2<0.05 else '✗ not significant'}")

# Final metrics
metrics_out = (
    closing.groupby("sentiment")
    .agg(
        Trades    =("Closed PnL","count"),
        Avg_PnL   =("Closed PnL","mean"),
        Win_Rate  =("Closed PnL",lambda x:(x>0).mean()),
        Total_PnL =("Closed PnL","sum"),
        Avg_Size  =("Size USD","mean"),
        Long_Ratio=("trade_side",lambda x:(x=="Long").sum()/len(x)),
    )
    .reindex(["Fear","Neutral","Greed"])
    .round(3)
)
metrics_out.to_csv("output/key_metrics.csv")
print("\n── Key Metrics by Sentiment ──")
print(metrics_out.to_string())
print("\n✓  All 8 charts saved to charts/")
