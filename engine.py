import numpy as np
import pandas as pd
from features import LAGS, NAMES, Scaler, build_features, hp_trend, nxt, seasonal_profile
from nn import Ensemble

FLAT, HS, NF = 0.015, (1, 2, 3), len(NAMES)
FOLDS = ((0.55, 0.08, 0.12), (0.68, 0.08, 0.12), (0.80, 0.08, 0.12))


def prepare(md):
    p, c = md["price"], md["cpi"]
    n = len(p)
    cpi_log = np.log(c.values) if c is not None else np.zeros(n)
    y = np.log(p.values) - cpi_log + cpi_log[-1]                 # log price in today's money
    return dict(y=y, months=p.index.month.values, dates=p.index, cpi_log=cpi_log, n=n, nominal=p.values, has_cpi=c is not None)


def features_upto(P, upto):
    """Feature matrix whose seasonal profile is learned only from rows before `upto` (no leakage)."""
    y, mo = P["y"], P["months"]
    prof, gain = seasonal_profile(y[:upto] - hp_trend(y[:upto]), mo[:upto])
    return build_features(y, mo, P["cpi_log"], prof), prof, gain


def fwd_mean(y, t):
    return y[t + 1:t + 13].mean()


def fit_fold(P, T1, T2, members, epochs, seed):
    """Train the fair-level and move networks on rows before T1, early-stopping on T1..T2."""
    y, n = P["y"], P["n"]
    X, prof, _ = features_upto(P, T1)
    tr, va = np.arange(LAGS, T1 - 12), np.arange(T1, T2 - 12)
    trend_tr, trend_va = hp_trend(y[:T1]), hp_trend(y[:T2])
    sc = Scaler().fit(X[tr])
    Ltr, Lva = trend_tr[tr] - y[tr], trend_va[va] - y[va]
    ls = Ltr.std() + 1e-9
    ef = Ensemble((NF, 32, 16, 1), members, seed)
    best_f = ef.fit_es(sc(X[tr]), Ltr / ls, sc(X[va]), Lva / ls, max_epochs=epochs)
    mtr, mva = np.arange(LAGS, T1 - 3), np.arange(T1, T2 - 3)
    tg = lambda r: np.column_stack([y[r + h] - y[r] for h in HS])
    ys = tg(mtr).std(0) + 1e-9
    em = Ensemble((NF, 32, 16, len(HS)), members, seed + 1)
    best_m = em.fit_es(sc(X[mtr]), tg(mtr) / ys, sc(X[mva]), tg(mva) / ys, max_epochs=epochs)
    allr = np.arange(LAGS, n)
    Fnn = np.full(n, np.nan)
    Fnn[allr] = y[allr] + ls * ef.predict(sc(X[allr]))[0].ravel()
    Mnn = np.full((n, len(HS)), np.nan)
    Mnn[allr] = em.predict(sc(X[allr]))[0] * ys
    return dict(X=X, prof=prof, F=Fnn, M=Mnn, ls=ls, best_f=best_f, best_m=best_m)


def candidates(y, X, Fnn):
    return {"Network": Fnn, "Last price": y, "12-month average": y - X[:, 6], "36-month average": y - X[:, 7], "Slow trend (EMA)": y - X[:, 9]}

