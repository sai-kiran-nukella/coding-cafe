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
