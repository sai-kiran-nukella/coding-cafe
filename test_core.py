import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import advisor as A
import data_loader as D
import engine as E
import features as F
import learning_demo as L
from sources import CPI, MARKETS


def make_files(n=300, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    gap = np.zeros(n)
    for i in range(1, n):
        gap[i] = 0.7 * gap[i - 1] + rng.normal(0, 0.04)
    price = np.exp(np.log(5) + np.cumsum(rng.normal(0, 0.004, n)) + gap + 0.06 * np.sin(2 * np.pi * (t % 12) / 12))
    cpi = 100 * np.exp(np.cumsum(np.full(n, 0.0025)))
    d = pd.date_range("2000-01-01", periods=n, freq="MS").strftime("%Y-%m-%d")
    mk = lambda v: pd.DataFrame({"observation_date": d, "VALUE": v}).to_csv(index=False).encode()
    p = pd.Series(price.round(4).astype(str))
    p[[10, 11, 100]] = "."
    return mk(p), mk(cpi.round(3))


PRICE, CPI_B = make_files()
OUT = lambda mu, resid=None: {h: dict(mu=mu * h / 2, sd=.04, change_pct=np.exp(mu * h / 2) - 1, low_pct=np.exp(mu * h / 2 - .06) - 1, high_pct=np.exp(mu * h / 2 + .06) - 1,
                                      probs={"rise": .8 if mu > 0 else .1, "fall": .1 if mu > 0 else .8, "flat": .1}, direction="rise" if mu > 0 else "fall", confidence=.8,
                                      resid=(resid if resid is not None else np.random.default_rng(0).normal(0, .03, 200)).tolist()) for h in (1, 2, 3)}
CFG = dict(MARKETS["Tomatoes"])


class TestSources(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(len(MARKETS), 6)
        for k, m in MARKETS.items():
            self.assertIn(m["series"], m["citation"]); self.assertIn(m["series"], m["csv"]); self.assertTrue(m["csv"].startswith("https://fred.stlouisfed.org/"))
            self.assertGreater(m["hold"], 0)
        self.assertIn("CPIAUCSL", CPI["csv"])


class TestLoader(unittest.TestCase):
    def test_fred_format_and_gaps(self):
        s = D.read_series("x.csv", PRICE)
        self.assertEqual(len(s), 300); self.assertFalse(s.isna().any()); self.assertEqual(s.attrs["filled"], 3)

    def test_weekly_to_monthly(self):
        w = pd.DataFrame({"date": pd.date_range("2000-01-03", periods=1000, freq="W-MON"), "price": np.linspace(1, 3, 1000)})
        s = D.read_series("g.csv", w.to_csv(index=False).encode())
        self.assertTrue(s.attrs["weekly"]); self.assertLess(len(s), 250)

    def test_rejects_bad_files(self):
        q = pd.DataFrame({"date": pd.date_range("2000", periods=40, freq="QS"), "v": range(1, 41)}).to_csv(index=False).encode()
        for bad in (q, b"a,b\n1,2\n", pd.DataFrame({"date": pd.date_range("2000", periods=50, freq="MS"), "price": -1.0}).to_csv(index=False).encode()):
            with self.assertRaises(D.DataError):
                D.read_series("b.csv", bad)

    def test_inflation_adjustment_and_save(self):
        md = D.build_market("x.csv", PRICE, CPI_B)
        self.assertTrue(md["info"]["inflation"]); self.assertEqual(len(md["cpi"]), len(md["price"]))
        old = D.DATA
        D.DATA = Path(tempfile.mkdtemp())
        try:
            D.save("k", PRICE, "test"); self.assertEqual(D.saved("k")["bytes"], PRICE)
        finally:
            D.DATA = old

    def test_short_history_rejected(self):
        short = pd.DataFrame({"date": pd.date_range("2000", periods=60, freq="MS"), "price": np.linspace(1, 2, 60)}).to_csv(index=False).encode()
        with self.assertRaises(D.DataError):
            D.build_market("s.csv", short)


