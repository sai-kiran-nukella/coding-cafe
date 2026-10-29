PRICEWISE
=========
A tool for small sellers (farmers, shopkeepers, makers, landlords) that answers two questions:
  1. Is my price fair?
  2. Should I sell now or wait?

Two neural networks, written from scratch in NumPy, make the predictions.
Prices are simulated for now. Real market data is the next step.


HOW TO RUN
----------
1. pip install -r requirements.txt
2. streamlit run app.py
3. Tests: python -m unittest tests/test_core.py

The first run trains the models (about 15 seconds) and saves them in the .cache folder.


HOW TO USE
----------
Enter your product, cost, price, quantity, demand, competition and urgency in the sidebar.
You can also upload a CSV of your own monthly prices (needs a "price" column, at least 6 rows).

The page has three sections:
  1. Is your price fair?       Fair price, range, margin, profit, suggested price.
  2. Sell now or wait?         Advice, income table, 12 month outlook, price forecast.
  3. How the model learned     Live training, and a shock test that shows the model repairing itself.


FILES
-----
simulator.py         Pretend market: supply, demand, seasons, shocks. Creates the training data.
market_feed.py       Gives the app the price history. Also reads your CSV.
nn.py                Neural network from scratch, ensemble of 5, straight-line baseline.
fair_price_net.py    Predicts the fair price from cost, demand, competition and month.
trend_net.py         Predicts whether the price will rise, fall or stay flat in 1-2 months.
models.py            Trains both models per product and caches them.
advisor.py           Turns predictions into verdicts, advice, income table and outlook.
learning_demo.py     Live training and the shock test with self-repair.
charts.py            Plots.
app.py               The Streamlit page.
tests/test_core.py   15 automatic checks.


HOW IT FITS TOGETHER
--------------------
simulator -> training data -> nn + fair_price_net + trend_net (via models)
-> advisor (decisions) -> app (page, with charts and learning demo)


LIMITS
------
All prices are simulated, so accuracy figures describe the simulator and will be lower on real data.
This is guidance, not financial advice.
