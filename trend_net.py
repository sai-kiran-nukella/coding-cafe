import math
import numpy as np
from nn import Ensemble
from fair_price_net import FairPriceModel

FLAT = 0.02
HORIZONS = (1, 2)


def features(prices, fair_now, cost_mom, month):
    p = np.asarray(prices, float)[-6:]
    r = np.diff(np.log(p)) / 0.1
    return np.r_[r, np.log(p[-1] / fair_now) / 0.15, cost_mom / 0.1, math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12)]


def cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class TrendModel:
    def __init__(self, good, seed=0):
        self.good, self.seed = good, seed
        self.ens, self.resid, self.metrics = Ensemble((9, 24, 12, 2), 5, seed), np.array([0.05, 0.07]), {}

    def _dataset(self, df, fair):
        X, Y, mom = [], [], []
        for _, d in df.groupby("path"):
            d = d.reset_index(drop=True)
            fp = fair.predict_many(d.cost, d.demand, d.comp, d.month)[0]
            lp, lc = np.log(d.price.values), np.log(d.cost.values)
            for t in range(5, len(d) - 2):
                X.append(features(d.price.values[t - 5:t + 1], fp[t], lc[t] - lc[max(0, t - 3)], d.month[t]))
                Y.append([lp[t + 1] - lp[t], lp[t + 2] - lp[t]])
                mom.append(lp[t] - lp[t - 3])
        return np.array(X), np.array(Y) / 0.1, np.array(mom)

    def fit(self, df, fair, epochs=120):
        paths = df["path"].unique()
        X, Y, _ = self._dataset(df[df.path.isin(paths[:-10])], fair)
        Xv, Yv, mom = self._dataset(df[df.path.isin(paths[-10:])], fair)
        self.ens.fit(X, Y, epochs)
        mu, _ = self.ens.predict(Xv)
        self.resid = np.std(Yv - mu, axis=0) * 0.1
        cls = lambda a: np.where(a > FLAT, 1, np.where(a < -FLAT, -1, 0))
        truth = cls(Yv[:, 1] * 0.1)
        self.metrics = {"dir_acc": float(np.mean(cls(mu[:, 1] * 0.1) == truth)), "momentum_acc": float(np.mean(cls(mom) == truth)),
                        "majority_acc": float(max(np.mean(truth == c) for c in (-1, 0, 1))),
                        "mae_pct": float(np.mean(np.abs(mu[:, 1] - Yv[:, 1])) * 0.1 * 100)}
        return self

    def predict(self, prices, fair_now, cost_mom, month):
        mu, sd = self.ens.predict(features(prices, fair_now, cost_mom, month).reshape(1, -1))
        out = {}
        for i, h in enumerate(HORIZONS):
            m, s = float(mu[0, i]) * 0.1, math.sqrt((float(sd[0, i]) * 0.1) ** 2 + float(self.resid[i]) ** 2)
            pr, pf = 1 - cdf((FLAT - m) / s), cdf((-FLAT - m) / s)
            probs = {"rise": pr, "fall": pf, "flat": max(0.0, 1 - pr - pf)}
            d = max(probs, key=probs.get)
            out[h] = {"mu": m, "sd": s, "direction": d, "confidence": probs[d], "probs": probs, "change_pct": math.exp(m) - 1,
                      "low_pct": math.exp(m - 1.28 * s) - 1, "high_pct": math.exp(m + 1.28 * s) - 1}
        return out
