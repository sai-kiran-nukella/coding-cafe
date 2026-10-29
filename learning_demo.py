import copy
import numpy as np
from fair_price_net import YS, features
from nn import Linear, Net
from simulator import EVENTS, GOODS, simulate, training_paths

SIZES = (5, 24, 12, 1)


def _xy(df, good):
    return features(good, df.cost, df.demand, df.comp, df.month), np.log(df.fair.values / GOODS[good]["p0"]) / YS


def _price(good, z):
    return GOODS[good]["p0"] * np.exp(YS * np.ravel(z))


def _ape(pred, true):
    return float(np.mean(np.abs(pred - true) / true))


def train_live(good, epochs=60, step=2, lr=0.01, seed=0):
    """Yields the state of a brand-new network every `step` epochs."""
    X, y = _xy(training_paths(good, 24, 60, 500 + seed), good)
    val = simulate(good, 60, 9000 + seed, 1)
    Xv, _ = _xy(val, good)
    net, rng = Net(SIZES, seed), np.random.default_rng(seed)
    lin_ape = _ape(_price(good, Linear(X, y).predict(Xv)), val.fair.values)
    for e in range(0, epochs + 1, step):
        for k in range(step if e else 0):
            net.epoch(X, y, lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * (e - step + k) / epochs))), 64, rng)
        pred = _price(good, net.forward(Xv))
        yield {"epoch": e, "ape": _ape(pred, val.fair.values), "linear_ape": lin_ape, "pred": pred, "true": val.fair.values}


def calm_model(good, seed=0, epochs=80):
    """A network that has only ever seen quiet markets."""
    parts = [simulate(good, 60, 700 + seed * 100 + i, i % 12 + 1, event_prob=0) for i in range(24)]
    X, y = zip(*[_xy(p, good) for p in parts])
    X, y = np.vstack(X), np.concatenate(y)
    net = Net(SIZES, seed)
    net.fit(X, y, epochs, 0.01, rng=np.random.default_rng(seed))
    val = simulate(good, 60, 9500 + seed, 5, event_prob=0)
    e = np.abs(_price(good, net.forward(_xy(val, good)[0])) - val.fair.values) / val.fair.values
    return net, X, y, float(e.mean()), float(e.std())


def shock_test(good, event=0, shock_at=10, months=36, seed=3):
    net, Xtr, ytr, m, s = calm_model(good)
    plain, fixed, thr = copy.deepcopy(net), copy.deepcopy(net), max(0.03, m + 3 * s)
    d = simulate(good, months, seed, 1, event_prob=0, force=(shock_at, event))
    X, y = _xy(d, good)
    rng, errs, rows, retrains = np.random.default_rng(1), [], [], 0
    for t in range(months):
        pa, pb = _price(good, plain.forward(X[t:t + 1]))[0], _price(good, fixed.forward(X[t:t + 1]))[0]
        true = d.fair.values[t]
        errs.append(abs(pb - true) / true)
        retrained = len(errs) >= 2 and np.mean(errs[-2:]) > thr
        if retrained:
            k = min(12, t + 1)
            j = rng.integers(0, len(Xtr), 150)
            fixed.fit(np.vstack([np.repeat(X[t + 1 - k:t + 1], 6, axis=0), Xtr[j]]), np.concatenate([np.repeat(y[t + 1 - k:t + 1], 6), ytr[j]]), 25, 0.004, 32, rng)
            retrains += 1
        rows.append({"month": t + 1, "true": true, "no_repair": pa, "repair": pb, "err_plain": abs(pa - true) / true, "err_repair": abs(pb - true) / true,
                     "retrained": bool(retrained), "shock": bool(d.active[t])})
    import pandas as pd
    return pd.DataFrame(rows), thr, retrains
