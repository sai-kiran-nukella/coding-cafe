import numpy as np
import pandas as pd

GOODS = {
    "Tomatoes": dict(kind="farm crop", unit="kg", verb="sell", p0=28, c0=17, eps=0.7, sig=0.9, dem_amp=0.05, dem_peak=12, sup_amp=0.18, sup_peak=2,
                     hold=0.05, vol=1.0, now="Sell now", ing="selling", noun="kg", qty=200, cost_q="What does it cost you to grow and bring 1 kg to market?", price_q="What price per kg are you asking?",
                     qty_q="How many kg do you have to sell?", hold_why="spoilage and storage"),
    "Handmade candles": dict(kind="handmade product", unit="piece", verb="sell", p0=180, c0=95, eps=1.1, sig=1.3, dem_amp=0.30, dem_peak=10.5, sup_amp=0.03,
                             sup_peak=6, hold=0.01, vol=0.8, now="Sell now", ing="selling", noun="candle", qty=60, cost_q="What does one candle cost you in materials, labour and packing?",
                             price_q="What are you charging per candle?", qty_q="How many candles are in stock?", hold_why="storage and tied-up cash"),
    "Room for rent": dict(kind="rental", unit="month", verb="let", p0=9000, c0=4200, eps=0.6, sig=0.5, dem_amp=0.12, dem_peak=7.5, sup_amp=0.03, sup_peak=1,
                          hold=0.09, vol=0.45, now="Let it out now", ing="letting it out", noun="room", qty=1, cost_q="What does the room cost you per month (upkeep, loan interest, tax)?",
                          price_q="What rent are you asking per month?", qty_q="How many rooms are you letting?", hold_why="rent lost while it stays empty"),
}
DEMAND_OPTIONS = {"Much weaker than usual": 0.70, "Weaker than usual": 0.85, "Normal": 1.0, "Stronger than usual": 1.15, "Much stronger than usual": 1.30}
COMP_OPTIONS = {"Far fewer sellers than usual": 0.75, "Fewer sellers": 0.88, "Usual number of sellers": 1.0, "More sellers": 1.15, "Flooded with sellers": 1.35}
LAMBDA, NOISE, THETA = 0.55, 0.02, 0.1
EVENTS = [("Input costs jumped", 0, (1.15, 1.5), (3, 9)), ("Demand dipped", 1, (0.65, 0.85), (2, 7)),
          ("Demand surged", 1, (1.15, 1.35), (2, 6)), ("More sellers entered", 2, (1.2, 1.45), (3, 9))]
CORR = np.array([[1, -0.2, 0], [-0.2, 1, -0.1], [0, -0.1, 1]])
VOLS = np.array([0.025, 0.03, 0.025])


def equilibrium(good, cost, demand, comp, month):
    g = GOODS[good]
    ang = lambda pk: 2 * np.pi * (np.asarray(month) - pk) / 12
    D = demand * (1 + g["dem_amp"] * np.cos(ang(g["dem_peak"])))
    K = comp * (1 + g["sup_amp"] * np.cos(ang(g["sup_peak"])))
    lD = 0.35 * np.tanh(np.log(D) / 0.35)
    lK = np.log(K)
    lK = lK + 3 * np.maximum(0, lK - 0.12) ** 2
    x = np.log(np.asarray(cost) / g["c0"])
    h = x + 1.2 * np.maximum(0, x - 0.15) ** 2
    return g["p0"] * np.exp((lD - lK + g["sig"] * h) / (g["eps"] + g["sig"]))


FORCED = {0: (1.6, 8), 1: (0.55, 6), 2: (1.6, 5), 3: (1.7, 8)}


def simulate(good, months=60, seed=0, start_month=1, event_prob=0.07, force=None):
    g, rng, chol = GOODS[good], np.random.default_rng(seed), np.linalg.cholesky(CORR)
    dev, active, rows, lnp = np.zeros(3), [], [], None
    for t in range(months):
        cal = (start_month - 1 + t) % 12 + 1
        dev = dev * (1 - THETA) + g["vol"] * VOLS * (chol @ rng.standard_normal(3))
        started = ""
        if rng.random() < event_prob:
            name, idx, (a, b), (d0, d1) = EVENTS[rng.integers(len(EVENTS))]
            active.append([name, idx, rng.uniform(a, b), int(rng.integers(d0, d1))])
            started = name
        if force and t == force[0]:
            name, idx = EVENTS[force[1]][0], EVENTS[force[1]][1]
            active.append([name, idx, FORCED[force[1]][0], FORCED[force[1]][1]])
            started = name
        mult = np.ones(3)
        for a in active:
            mult[a[1]] *= a[2] ** g["vol"]
        cost, demand, comp = g["c0"] * np.exp(dev[0]) * mult[0], np.exp(dev[1]) * mult[1], np.exp(dev[2]) * mult[2]
        fair = float(equilibrium(good, cost, demand, comp, cal))
        lnp = np.log(fair) if lnp is None else (1 - LAMBDA) * lnp + LAMBDA * np.log(fair)
        rows.append(dict(t=t, month=cal, cost=float(cost), demand=float(demand), comp=float(comp), fair=fair,
                         price=float(np.exp(lnp + rng.normal(0, NOISE))), event=started, active=bool(active)))
        for a in active:
            a[3] -= 1
        active = [a for a in active if a[3] > 0]
    return pd.DataFrame(rows)


def training_paths(good, n=48, months=60, seed=1000):
    rng = np.random.default_rng(seed)
    parts = [simulate(good, months, int(rng.integers(1 << 30)), int(rng.integers(1, 13))).assign(path=i) for i in range(n)]
    return pd.concat(parts, ignore_index=True)


def nearest_option(options, value):
    return min(options, key=lambda k: abs(options[k] - value))
