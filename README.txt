PRICEWISE (real data version)
=============================
Checks whether a seller's price is fair and whether to sell now or wait, using REAL market data.
The fair price here means the long-run, inflation-adjusted level the market keeps returning to.
Short-run swings (a demand spike, a bad season) move the price away from it for a while.
A lasting shift changes the fair price itself.


RUN
---
1. pip install -r requirements.txt
2. streamlit run app.py
3. Tests: python -m unittest tests/test_core.py


HOW THE DATA WORKS
------------------
Pick a market. The app recommends the best official file for it and shows the publisher,
the page, a direct download link, the file format and how to cite it.
  - "Download the recommended file" fetches it from the source and saves it in the data folder.
  - Or upload your own CSV/Excel file (a date column and a price column, monthly or weekly, 10+ years).
  - An inflation file (US CPI) is downloaded automatically so prices can be adjusted for inflation.
Tomatoes and Copper come bundled (FRED data tables, downloaded Nov 2026) so they work straight away.

Markets: Tomatoes, Eggs, Ground beef, Gasoline, Wheat, Copper, and Custom file.
All recommended files are on FRED (Federal Reserve Bank of St. Louis); the underlying publishers are
the U.S. Bureau of Labor Statistics, U.S. Energy Information Administration and the International Monetary Fund.
Prices are in US dollars. Use "Your local price vs this benchmark" to adjust for your own market.


THE PAGE
--------
Data                    Recommended file, citation, upload, data quality checks.
1. Is your price fair?  Fair price and range, verdict, suggested price, fair price history.
2. Sell now or wait?    Advice, income table, seasonal pattern, 1-3 month forecast with ranges.
3. How the model learned
     Backtest           Accuracy on history the model never saw, compared with simple methods.
     Live training      Trains a new network on this market while you watch.
     Shock replay       Replays a real shock; a fixed network vs one that retrains on new information.


FILES
-----
sources.py         Recommended file for each market, links and citations.
data_loader.py     Reads CSV/Excel, handles weekly data and gaps, downloads, saves, adjusts for inflation.
features.py        Trend filter, seasonality and the 21 features the network sees (no look-ahead).
nn.py              Neural network from scratch (Adam, L2, Huber loss, early stopping, ensembles).
engine.py          Learns the market: rolling backtests, model blending, fair price, forecasts.
advisor.py         Verdicts, suggested price, sell/wait/split decision, income table.
learning_demo.py   Live training and shock replay.
charts.py, app.py  The page.
tests/test_core.py Automatic checks.


HOW THE MODEL WORKS
-------------------
Prices are converted to log "today's money". A smoothing filter gives the long-run level in hindsight.
A network learns to estimate that level using only past data. Its estimate is blended with simple methods
(last price, 12 and 36 month averages, slow trend) with weights chosen from three rolling backtest windows,
so a method only gets weight if it actually works on that market. Forecasts of 1-3 month moves are shrunk
towards "no change" unless backtests show they beat it. Ranges come from the real backtest errors.


LIMITS
------
- On the real series tested (tomatoes, copper) the network helps most in trending markets; in
  mean-reverting ones simple averages do as well. The blend and the "trust" figures reflect this honestly.
- Short-term direction is hard to predict. Forecast trust is low or zero for some markets.
- Retraining on new information is delayed (outcomes are known 12 months later), so the shock replay
  shows both the gains and the losses.
- Gasoline, Eggs, Ground beef and Wheat download links were not tested here; Tomatoes and Copper were
  checked against the real FRED data. Guidance only, not financial advice.
