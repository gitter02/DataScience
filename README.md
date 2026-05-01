# Trader Performance vs Market Sentiment
Data Science Assignment

## The What 
Analysis of how Bitcoin Fear & Greed sentiment relates to trader behavior and performance on Hyperliquid.

Dataset
- **211,224 real trades** across **32 accounts** (May 2023 – May 2025)
- **104,402 closing trades** with realized PnL analysed
- **731 days** of real Fear/Greed index data (full overlap with trade history)
- Coins: HYPE, BTC, ETH, SOL, FARTCOIN, MELANIA, and 50+ others

> **Most surprising finding:** PnL barely differs between Fear and Greed ($101 vs $106 avg) — but *how* traders behave changes dramatically. Fear days have large long positions ($7,375 avg, 58.5% long). Greed days smaller, short-biased positions ($4,234 avg, 33% long).                                            These traders *fade* the sentiment, not follow it.
---

## Key Findings
> PnL difference Fear vs Greed is **not statistically significant** (p=0.53) so the alternate hypothesis becomes true that performance is consistent.  
> Position size difference is significant (p < 0.0001) so behavior changes strongly.
---

## Setup for Windows with April 2026 Versions of everything

```bash
# 1. clone
cd primetrade_assignment

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn nbformat jupyter

# 4. Run analysis (8 charts + output CSVs)
python analysis.py

# 5. Open annotated notebook
jupyter notebook notebook.ipynb
```
---

## Project Structure
```
primetrade_assignment/
├── data/
│   ├── sentiment.csv         # Real Fear & Greed index (2,644 days, 2018–2025)
│   └── trades.csv            # Hyperliquid trade history (211,224 rows, 32 accounts)
├── charts/                
│   ├── 01_pnl_by_sentiment.png
│   ├── 02_winrate_avgpnl.png
│   ├── 03_size_frequency.png
│   ├── 04_long_short_bias.png
│   ├── 05_cumulative_pnl_timeline.png
│   ├── 06_pnl_by_coin.png
│   ├── 07_trader_segments.png
│   └── 08_heatmap_direction_sentiment.png
├── output/
│   ├── key_metrics.csv      
│   └── daily_summary.csv     
├── analysis.py              # to be ran first
├── notebook.ipynb            
├── requirements.txt
└── README.md
```
---

## Charts Produced

| # | Chart | Interpretation |
|---|---|---|
| 1 | PnL distribution (box + density) | Consistent across regimes; Fear has wider spread |
| 2 | Win rate & avg PnL by sentiment | Fear = highest win rate (84.4%) |
| 3 | Position size & trade frequency | Fear → much larger positions ($7,375 avg) |
| 4 | Long/Short bias | Fear → 58.5% long; Greed → only 33% long (contrarian behavior) |
| 5 | Cumulative PnL with sentiment background | Steady upward curve across all regimes |
| 6 | Total PnL by Coin | HYPE & ETH dominate; BTC consistent |
| 7 | Trader segments (size / activity / consistency) | Large-position & frequent traders outperform |
| 8 | Direction × Sentiment heatmap | Close Short on Fear = highest avg PnL |

---

## Findings/Analysis
**1 — Be contrarian: go long during Fear, reduce longs during Greed**
> These traders use Greed as a signal to short (33% long on Greed days) and Fear as a buying opportunity (58.5% long). Win rate is highest on Fear days (84.4%). The crowd panics — the profitable traders buy.

**2 — Scale position size to volatility, not sentiment direction**
> Position sizes are 74% larger on Fear days ($7,375 vs $4,234). This is deliberate: high-volatility fear regimes offer larger dislocations to exploit. During Greed, take smaller, tighter trades with high precision.
---

## Methodology
1. Data 
    Given datasets were loaded and manually checked for missing values, duplicate rows, and column consistency. (Basic EDA)

2. Cleaning
    The trader dataset used the `Timestamp IST` column, which was converted to datetime format and normalized to daily date level for merging with the Fear & Greed dataset.

    The sentiment dataset mainly used:
    - `date`
    - `value` (Fear & Greed score from 0–100)
    - `classification`

    Since sentiment data is available daily, both datasets were merged using the date column.
    For performance analysis, only rows with `Closed PnL != 0` were considered, as these represent completed trades with realized profit or loss.

3. Analysis
    metrics created for analysis:
    - average PnL
    - total PnL
    - win rate
    - average position size
    - number of trades per day
    - long vs short ratio

    Trader segmentation was done by splitting accounts into groups such as:
    - frequent vs infrequent traders
    - large position vs small position traders
    - higher-performing vs lower-performing traders

    Charts and Mann-Whitney U tests were used to compare Fear vs Greed performance and check whether observed differences were statistically meaningful.

4. Final summary file

![alt text](screenshot.png)
First, during Fear days, increasing long exposure with controlled leverage may be beneficial, especially for experienced traders. Since win rate is highest during Fear and traders tend to buy during panic, Fear can be treated as an opportunity rather than a warning signal.
![alt text](ss2.png)
Second, during Greed days, reducing position size and avoiding aggressive longs may improve consistency. Since traders become more short-biased during Greed and profits do not increase significantly, smaller and more precise trades appear to be the better strategy.

Finally a simple predictive model can be built using sentiment, trade size, and trade frequency to predict next-day profitability buckets. A lightweight Streamlit dashboard is added to make exploration of trader segments and sentiment-based behavior more interactive.

link to streamlit
https://simplebonus.streamlit.app/