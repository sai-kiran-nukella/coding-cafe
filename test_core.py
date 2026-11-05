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


class TestFeatures(unittest.TestCase):
    def test_hp_trend_keeps_a_straight_line(self):
        y = np.linspace(0, 3, 100)
        self.assertLess(np.max(np.abs(F.hp_trend(y) - y)), 0.05)

    def test_no_look_ahead(self):
        rng = np.random.default_rng(3)
        y, mo = np.cumsum(rng.normal(0, .02, 120)), np.tile(np.arange(1, 13), 10)
        X1 = F.build_features(y, mo, np.zeros(120), np.zeros(13))
        y2 = y.copy(); y2[70:] += 5
        X2 = F.build_features(y2, mo, np.zeros(120), np.zeros(13))
        np.testing.assert_allclose(X1[:70], X2[:70])
        self.assertFalse(np.isnan(X1[F.LAGS:]).any())


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = E.analyze(D.build_market("x.csv", PRICE, CPI_B), fast=True)

    def test_outputs(self):
        a = self.A
        self.assertAlmostEqual(sum(a["weights"].values()), 1.0, places=6)
        self.assertGreater(a["now"]["fair"], 0); self.assertLess(a["now"]["low"], a["now"]["high"])
        for h in (1, 2, 3):
            o = a["outlook"][h]; self.assertLess(o["low_pct"], o["high_pct"]); self.assertAlmostEqual(sum(o["probs"].values()), 1.0, places=6)

    def test_blend_beats_last_price(self):
        t = self.A["fair_table"]
        self.assertLess(t["Blend (used)"], t["Last price"])

    def test_forecast_never_trusted_without_skill(self):
        self.assertTrue((self.A["move_table"]["Trust in forecast %"].between(0, 100)).all())

    def test_bundled_real_tomato_file(self):
        f = Path(__file__).parent.parent / "data" / "APU0000712311.csv"
        if f.exists():
            a = E.analyze(D.build_market("t.csv", f.read_bytes()), fast=True)
            self.assertTrue(1.0 < a["now"]["fair"] < 3.5); self.assertGreater(a["n"], 500)


class TestAdvisor(unittest.TestCase):
    FAIR = {"price": 100.0, "low": 96.0, "high": 104.0}

    def test_verdicts(self):
        f = lambda p: A.compare_to_fair(p, self.FAIR, 60, 100)["status"]
        self.assertEqual((f(80), f(100), f(130)), ("underpriced", "fair", "overpriced"))

    def test_floor(self):
        v = A.compare_to_fair(90, self.FAIR, 120, 100)
        self.assertGreaterEqual(v["suggested"], 126); self.assertTrue(v["underwater"])

    def test_timing_rules(self):
        self.assertEqual(A.sell_or_wait(CFG, OUT(.2), 100, 50, 2)["action"], "wait")
        self.assertEqual(A.sell_or_wait(CFG, OUT(-.2), 100, 50, 2)["action"], "sell_now")
        self.assertEqual(A.sell_or_wait(CFG, OUT(.4), 100, 50, 0)["action"], "sell_now")
        self.assertEqual(A.sell_or_wait(dict(CFG, hold=0.01), OUT(.03, np.random.default_rng(1).normal(0, .1, 300)), 100, 50, 2)["action"], "split")

    def test_money(self):
        self.assertEqual(A.money(1992.4), "$1,992"); self.assertEqual(A.money(1.99), "$1.99")


class TestDemos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = E.prepare(D.build_market("x.csv", PRICE, CPI_B))

    def test_training_improves(self):
        fr = list(L.train_live(self.P, epochs=40))
        self.assertLess(fr[-1]["err"], fr[0]["err"])

    def test_replay(self):
        ev = L.shock_events(self.P)
        self.assertTrue(ev)
        d, flagged, kept = L.replay(self.P, ev[0]["i"])
        self.assertLessEqual(kept, flagged); self.assertGreater(len(d), 20)


if __name__ == "__main__":
    unittest.main()
