import copy
import numpy as np
import pandas as pd
from engine import features_upto, fwd_mean
from features import LAGS, NAMES, Scaler, hp_trend
from nn import Net


def train_live(P, epochs=80, step=2, seed=0):
    """Trains a brand-new fair-price network and yields its test accuracy as it learns."""
    y, n = P["y"], P["n"]
    T1 = int(0.70 * n)
    T2 = min(T1 + int(0.10 * n), n - 24)
    X, _, _ = features_upto(P, T1)
    tr, va, te = np.arange(LAGS, T1 - 12), np.arange(T1, T2 - 12), np.arange(T2, n - 12)
    L = hp_trend(y[:T1])[tr] - y[tr]
    ls, sc, rng = L.std() + 1e-9, Scaler().fit(X[tr]), np.random.default_rng(seed)
    net = Net((len(NAMES), 32, 16, 1), seed)
    R = np.array([fwd_mean(y, t) for t in te])
    base = {"12-month average": np.mean(np.abs(R - (y[te] - X[te, 6]))), "Last price": np.mean(np.abs(R - y[te]))}
    for e in range(0, epochs + 1, step):
        for k in range(step if e else 0):
            net.epoch(sc(X[tr]), L / ls, 0.005 * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (e - step + k) / epochs))), 32, rng, 1.5)
        F = y[te] + ls * net.forward(sc(X[te])).ravel()
        yield {"epoch": e, "err": 100 * float(np.mean(np.abs(R - F))), "base": {k: 100 * float(v) for k, v in base.items()},
               "dates": P["dates"][te], "price": np.exp(y[te]), "realized": np.exp(R), "estimate": np.exp(F)}


def shock_events(P, k=3):
    y, n = P["y"], P["n"]
    gap = y - hp_trend(y)
    out = []
    for j in np.argsort(-np.abs(gap)):
        if LAGS + 110 <= j <= n - 8 and all(abs(j - o["i"]) > 24 for o in out):
            out.append({"i": int(j), "date": P["dates"][j], "gap": float(np.exp(gap[j]) - 1)})
        if len(out) >= k:
            break
    return out


def replay(P, i, seed=0):
    """Replays a real shock month by month. The fair-price label for month r (where prices settled over the next
    12 months) is only known at month r+12. A frozen network is compared with one that tries to retrain when its
    revealed errors grow, and keeps the update only if it does better on the newest revealed months."""
    y, n = P["y"], P["n"]
    s0, e0 = i - 6, min(i + 36, n - 13)
    X, _, _ = features_upto(P, s0)
    R = lambda t: fwd_mean(y, t)
    tr = np.arange(LAGS, s0 - 12)
    sc, rng = Scaler().fit(X[tr]), np.random.default_rng(seed)
    L = np.array([R(t) - y[t] for t in tr])
    ls = L.std() + 1e-9
    net = Net((len(NAMES), 32, 16, 1), seed)
    net.fit(sc(X[tr]), L / ls, 150, 0.005, 32, rng, 1.5)
    upd = copy.deepcopy(net)
    pred = lambda m, t: y[t] + ls * m.forward(sc(X[t:t + 1]))[0, 0]
    err = lambda m, rows: float(np.mean([abs(R(q) - pred(m, q)) for q in rows]))
    thr = max(0.04, 2 * err(net, tr[-36:]))
    rows, revealed, last, flagged_n, kept = [], [], -99, 0, 0
    for t in range(s0, e0 + 1):
        revealed.append(t - 12)
        flagged = retrained = False
        if len(revealed) >= 8 and t - last >= 3 and err(upd, revealed[-3:]) > thr:
            flagged, last = True, t
            flagged_n += 1
            k, jj = revealed[-15:-3], rng.integers(0, len(tr), 150)
            cand = copy.deepcopy(upd)
            cand.fit(np.vstack([np.repeat(sc(X[k]), 6, axis=0), sc(X[tr][jj])]), np.concatenate([np.repeat([(R(q) - y[q]) / ls for q in k], 6), L[jj] / ls]), 25, 0.004, 32, rng, 1.5)
            if err(cand, revealed[-3:]) < err(upd, revealed[-3:]):
                upd, retrained, kept = cand, True, kept + 1
        pf, pu = pred(net, t), pred(upd, t)
        rows.append({"date": P["dates"][t], "settled": np.exp(R(t)), "frozen": np.exp(pf), "updating": np.exp(pu), "price": np.exp(y[t]),
                     "err_frozen": abs(np.exp(R(t) - pf) - 1), "err_updating": abs(np.exp(R(t) - pu) - 1), "flagged": flagged, "retrained": retrained})
    return pd.DataFrame(rows), flagged_n, kept
