import re
import numpy as np
import pandas as pd
from simulator import GOODS, simulate

FEED_SEEDS = {"Tomatoes": 7, "Handmade candles": 4, "Room for rent": 33}


class SimulatedFeed:
    """Stand-in for a real price feed. Part 2 replaces this class with live data."""

    def __init__(self, good, today=None, months=60):
        end = pd.Timestamp(today or pd.Timestamp.today()).to_period("M").to_timestamp()
        self.good, self.dates = good, pd.date_range(end=end, periods=months, freq="MS")
        self.df = simulate(good, months, FEED_SEEDS[good], int(self.dates[0].month)).assign(date=self.dates)

    def latest(self):
        return self.df.iloc[-1].to_dict()

    def history(self, n=24):
        return self.df.tail(n).reset_index(drop=True)

    def cost_momentum(self):
        c = self.df["cost"].values
        return float(np.log(c[-1] / c[-4]))


def parse_prices(text):
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", (text or "").replace("₹", ""))]
    if len(nums) < 6:
        raise ValueError("Please enter at least 6 monthly prices, oldest first.")
    if any(n <= 0 for n in nums):
        raise ValueError("Prices must be positive numbers.")
    return nums


def read_price_csv(file):
    import pandas as pd
    df = pd.read_csv(file)
    col = next((c for c in df.columns if c.strip().lower() == "price"), None)
    if col is None:
        nums = df.select_dtypes("number").columns
        if not len(nums):
            raise ValueError("The CSV needs a 'price' column.")
        col = nums[-1]
    vals = [float(v) for v in df[col].dropna()]
    if len(vals) < 6 or any(v <= 0 for v in vals):
        raise ValueError("Need at least 6 positive monthly prices.")
    return vals
