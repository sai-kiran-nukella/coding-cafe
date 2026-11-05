import numpy as np

LAGS = 36
NAMES = ["chg1", "chg3", "chg6", "chg12", "chg24", "dev6", "dev12", "dev36", "ema_fast_gap", "ema_slow_gap", "vol6", "vol12", "range36",
         "sin", "cos", "season_now", "season_next1", "season_next3", "season_dev12", "infl12", "infl3"]


def hp_trend(y, lam=129600.0):
    """Two-sided Hodrick-Prescott trend: the smooth long-run level of a (log) price series."""
    n = len(y)
    if n < 8:
        return np.asarray(y, float).copy()
    D = np.diff(np.eye(n), 2, axis=0)
    return np.linalg.solve(np.eye(n) + lam * D.T @ D, y)


def ema(y, a):
    out = np.empty(len(y))
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = a * y[i] + (1 - a) * out[i - 1]
    return out


def nxt(m, k):
    return (m - 1 + k) % 12 + 1


def seasonal_profile(dev, months, min_gain=0.04):
    """Typical gap from the long-run level for each calendar month (index 1..12). Zero if seasonality is weak."""
    prof = np.zeros(13)
    for m in range(1, 13):
        sel = months == m
        prof[m] = np.median(dev[sel]) if sel.any() else 0.0
    prof[1:] -= prof[1:].mean()
    v = np.var(dev)
    gain = 1 - np.var(dev - prof[months]) / v if v > 0 else 0.0
    return (prof, gain) if gain >= min_gain else (np.zeros(13), max(gain, 0.0))


def build_features(y, months, cpi_log, prof):
    """Features for every month t using only data up to t (no look-ahead). Rows before LAGS are NaN."""
    n = len(y)
    X = np.full((n, len(NAMES)), np.nan)
    ef, es, dy, sy = ema(y, 0.25), ema(y, 0.03), np.r_[0, np.diff(y)], y - prof[months]
    for t in range(LAGS, n):
        w = lambda k: y[t - k + 1:t + 1]
        m = int(months[t])
        X[t] = [y[t] - y[t - 1], y[t] - y[t - 3], y[t] - y[t - 6], y[t] - y[t - 12], y[t] - y[t - 24],
                y[t] - w(6).mean(), y[t] - w(12).mean(), y[t] - w(36).mean(), y[t] - ef[t], y[t] - es[t],
                dy[t - 5:t + 1].std(), dy[t - 11:t + 1].std(), (y[t] - w(36).min()) / (np.ptp(w(36)) + 1e-9),
                np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12), prof[m], prof[nxt(m, 1)] - prof[m], prof[nxt(m, 3)] - prof[m],
                sy[t] - sy[t - 11:t + 1].mean(), cpi_log[t] - cpi_log[t - 12], (cpi_log[t] - cpi_log[t - 3]) * 4]
    return X


class Scaler:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        return self

    def __call__(self, X):
        return np.clip((X - self.mu) / self.sd, -6, 6)
