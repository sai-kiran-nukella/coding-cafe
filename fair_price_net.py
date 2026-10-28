import numpy as np
from nn import Ensemble, Linear
from simulator import GOODS

YS = 0.3


def features(good, cost, demand, comp, month):
    g, m = GOODS[good], np.asarray(month, float)
    cols = [np.log(np.asarray(cost, float) / g["c0"]) / 0.3, np.log(np.asarray(demand, float)) / 0.3, np.log(np.asarray(comp, float)) / 0.3,
            np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]
    return np.column_stack(np.broadcast_arrays(*cols))


class FairPriceModel:
    def __init__(self, good, seed=0):
        self.good, self.p0, self.seed = good, GOODS[good]["p0"], seed
        self.ens, self.resid, self.metrics = Ensemble((5, 24, 12, 1), 5, seed), 0.03, {}

    def fit(self, df, epochs=120):
        paths = df["path"].unique()
        tr, va = df[df.path.isin(paths[:-10])], df[df.path.isin(paths[-10:])]
        X = features(self.good, tr.cost, tr.demand, tr.comp, tr.month)
        y = np.log(tr.fair.values / self.p0) / YS
        self.ens.fit(X, y, epochs)
        Xv, yv = features(self.good, va.cost, va.demand, va.comp, va.month), np.log(va.fair.values / self.p0) / YS
        mu, _ = self.ens.predict(Xv)
        self.resid = float(np.std(yv - mu.ravel()))
        lin = Linear(X, y).predict(Xv).ravel()
        ape = lambda z: float(np.mean(np.abs(self.p0 * np.exp(YS * z) - va.fair.values) / va.fair.values))
        self.metrics = {"ape": ape(mu.ravel()), "linear_ape": ape(lin)}
        return self

    def _band(self, X):
        mu, sd = self.ens.predict(X)
        mu, sd = mu.ravel(), np.sqrt(sd.ravel() ** 2 + self.resid ** 2)
        f = lambda z: self.p0 * np.exp(YS * z)
        return f(mu), f(mu - 1.28 * sd), f(mu + 1.28 * sd)

    def predict_many(self, cost, demand, comp, month):
        return self._band(features(self.good, cost, demand, comp, month))

    def predict(self, cost, demand, comp, month):
        g = GOODS[self.good]
        cost = float(np.clip(cost, 0.5 * g["c0"], 2.0 * g["c0"]))
        X = features(self.good, [cost], [demand], [comp], [month])
        p, lo, hi = (float(v[0]) for v in self._band(X))
        drivers = []
        for name, cols in [("cost", [0]), ("demand", [1]), ("competition", [2]), ("season", [3, 4])]:
            Xn = X.copy()
            Xn[:, cols] = 0
            drivers.append((name, p / float(self._band(Xn)[0][0]) - 1))
        return {"price": p, "low": lo, "high": hi, "drivers": sorted(drivers, key=lambda d: -abs(d[1]))}
