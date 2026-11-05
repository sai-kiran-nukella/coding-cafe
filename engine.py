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


def analyze(md, fast=False, seed=0, progress=None):
    P = prepare(md)
    y, months, n = P["y"], P["months"], P["n"]
    if n < 120:
        raise ValueError("At least 120 months of data are needed.")
    members, epochs = (3 if fast else 5), (50 if fast else 200)
    say = progress or (lambda s: None)
    errs, pooled_va, folds, rows_info = {}, [], [], []
    for k, (a, wv, wt) in enumerate(FOLDS):
        say(f"Backtest window {k + 1} of {len(FOLDS)}")
        T1 = int(a * n)
        T2, T3 = T1 + int(wv * n), min(T1 + int((wv + wt) * n), n)
        f = fit_fold(P, T1, T2, members, epochs, seed + 10 * k)
        f.update(T1=T1, T2=T2, T3=T3, cands=candidates(y, f["X"], f["F"]))
        f["va"], f["te"] = np.arange(T1, T2 - 12), np.arange(T2, min(T3, n - 12))
        f["mva"], f["mte"] = np.arange(T1, T2 - 3), np.arange(T2, min(T3, n - 3))
        folds.append(f)
    names = list(folds[0]["cands"])
    err_te = {nm: np.mean([np.mean([abs(fwd_mean(y, t) - f["cands"][nm][t]) for t in f["te"]]) for f in folds]) for nm in names}
    w = np.array([1 / (err_te[nm] ** 6 + 1e-18) for nm in names])
    w /= w.sum()
    fair_tab = dict(err_te)
    blend = lambda f: sum(wi * f["cands"][nm] for wi, nm in zip(w, names))
    fair_tab["Blend (used)"] = float(np.mean([np.mean([abs(fwd_mean(y, t) - blend(f)[t]) for t in f["te"]]) for f in folds]))
    rr = np.array([fwd_mean(y, t) - blend(f)[t] for f in folds for t in np.r_[f["va"], f["te"]]])
    q_lo, q_hi = np.quantile(rr, [0.10, 0.90])
    last = folds[-1]
    test = pd.DataFrame({"date": P["dates"][last["te"]], "realized": [np.exp(fwd_mean(y, t) + P["cpi_log"][t] - P["cpi_log"][-1]) for t in last["te"]],
                         "estimate": np.exp(blend(last)[last["te"]] + P["cpi_log"][last["te"]] - P["cpi_log"][-1]), "price": P["nominal"][last["te"]]})

    # 1-3 month moves: stack [network, seasonal, reversion to 12m average] on pooled validation rows
    say("Combining forecasts")
    cls = lambda v: np.where(v > FLAT, 1, np.where(v < -FLAT, -1, 0))
    coef, mt, resid = [], [], {}
    comps = lambda f, rows, i, h: np.column_stack([f["M"][rows, i], [f["prof"][nxt(int(months[t]), h)] - f["prof"][months[t]] for t in rows], -f["X"][rows, 6]])
    for i, h in enumerate(HS):
        A = np.vstack([comps(f, f["mva"], i, h) for f in folds])
        b = np.concatenate([y[f["mva"] + h] - y[f["mva"]] for f in folds])
        ab = np.clip(np.linalg.lstsq(A, b, rcond=None)[0], 0, 1.2)
        rows_t = [(comps(f, f["mte"], i, h) @ ab, y[f["mte"] + h] - y[f["mte"]], y[f["mte"]] - y[f["mte"] - 3], f["M"][f["mte"], i]) for f in folds]
        pred, act, mom, nn_only = (np.concatenate([r[j] for r in rows_t]) for j in range(4))
        imp = 1 - np.mean(np.abs(act - pred)) / np.mean(np.abs(act))        # skill over "no change"
        scale = float(np.clip((imp - 0.02) / 0.06, 0, 1))                     # forecasts with no proven skill are shrunk to zero
        ab, pred = ab * scale, pred * scale
        coef.append(ab)
        mt.append({"Horizon (months)": h, "Forecast error %": 100 * np.mean(np.abs(act - pred)), "'No change' error %": 100 * np.mean(np.abs(act)),
                   "Network alone error %": 100 * np.mean(np.abs(act - nn_only)), "Direction right %": 100 * np.mean(cls(pred) == cls(act)),
                   "Direction right % (momentum)": 100 * np.mean(cls(mom) == cls(act)), "Direction right % (always flat)": 100 * np.mean(cls(act) == 0),
                   "Trust in forecast %": 100 * scale})
        resid[h] = np.concatenate([(y[f["mva"] + h] - y[f["mva"]]) - comps(f, f["mva"], i, h) @ ab for f in folds] + [act - pred])

    # ---- final models on all data
    say("Final training")
    hp_all = hp_trend(y)
    prof_all, gain_all = seasonal_profile(y - hp_all, months)
    Xf = build_features(y, months, P["cpi_log"], prof_all)
    rf, rm = np.arange(LAGS, n - 12), np.arange(LAGS, n - 3)
    sc = Scaler().fit(Xf[rf])
    ls = (hp_all[rf] - y[rf]).std() + 1e-9
    bf, bm = int(np.mean([f["best_f"] for f in folds]) * 1.2) + 5, int(np.mean([f["best_m"] for f in folds]) * 1.2) + 5
    ef = Ensemble((NF, 32, 16, 1), members, seed + 100)
    ef.fit(sc(Xf[rf]), (hp_all[rf] - y[rf]) / ls, bf, 0.005, delta=1.5)
    Ym = np.column_stack([y[rm + h] - y[rm] for h in HS])
    ys = Ym.std(0) + 1e-9
    em = Ensemble((NF, 32, 16, len(HS)), members, seed + 101)
    em.fit(sc(Xf[rm]), Ym / ys, bm, 0.005, delta=1.5)
    allr = np.arange(LAGS, n)
    Ffin = np.full(n, np.nan)
    Ffin[allr] = y[allr] + ls * ef.predict(sc(Xf[allr]))[0].ravel()
    Fb = sum(wi * c for wi, c in zip(w, candidates(y, Xf, Ffin).values()))
    mnow = em.predict(sc(Xf[n - 1:n]))[0][0] * ys
    infl = float(P["cpi_log"][-1] - P["cpi_log"][-13]) if P["has_cpi"] else 0.0
    im, m_now = infl / 12, int(months[-1])
    outlook = {}
    for i, h in enumerate(HS):
        comp = np.array([mnow[i], prof_all[nxt(m_now, h)] - prof_all[m_now], -Xf[n - 1, 6]])
        mu = float(comp @ coef[i] + h * im)
        r = resid[h]
        lo, hi = mu + np.quantile(r, 0.10), mu + np.quantile(r, 0.90)
        pr, pf = float(np.mean(mu + r > FLAT)), float(np.mean(mu + r < -FLAT))
        probs = {"rise": pr, "fall": pf, "flat": max(0.0, 1 - pr - pf)}
        d = max(probs, key=probs.get)
        outlook[h] = {"mu": mu, "sd": float((hi - lo) / 2.56), "change_pct": float(np.exp(mu) - 1), "low_pct": float(np.exp(lo) - 1), "high_pct": float(np.exp(hi) - 1),
                      "probs": probs, "direction": d, "confidence": probs[d], "resid": r.tolist()}
    gap = y - hp_all
    k = np.arange(LAGS, n - 13)
    phi = float(np.clip(np.polyfit(gap[k], gap[k + 1], 1)[0], 0, 0.999))
    shifts, d24 = [], hp_all[24:] - hp_all[:-24]
    for j in np.argsort(-np.abs(d24)):
        if abs(d24[j]) < 0.15 or len(shifts) >= 3:
            break
        if all(abs(j - s["i"]) > 24 for s in shifts):
            shifts.append({"i": int(j), "start": P["dates"][j], "end": P["dates"][j + 24], "change": float(np.exp(d24[j]) - 1)})
    fair = float(np.exp(Fb[-1]))
    now_price = float(P["nominal"][-1])
    hist = pd.DataFrame({"date": P["dates"], "price": P["nominal"], "fair": np.exp(Fb + P["cpi_log"] - P["cpi_log"][-1])})
    return {"now": {"date": P["dates"][-1], "price": now_price, "fair": fair, "low": float(np.exp(Fb[-1] + q_lo)), "high": float(np.exp(Fb[-1] + q_hi)), "gap": now_price / fair - 1},
            "outlook": outlook, "infl_annual": float(np.exp(infl) - 1), "half_life": float(np.log(0.5) / np.log(phi)) if 0 < phi < 1 else None, "shifts": shifts,
            "season": {m: float(np.exp(prof_all[m]) - 1) for m in range(1, 13)}, "season_gain": float(gain_all), "hist": hist, "test": test,
            "fair_table": {k_: float(v) for k_, v in fair_tab.items()}, "move_table": pd.DataFrame(mt), "weights": dict(zip(names, [float(x) for x in w])),
            "epochs": {"fair": bf, "move": bm}, "n": n, "has_cpi": P["has_cpi"], "info": md["info"], "windows": len(FOLDS)}
