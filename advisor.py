import calendar
import math
import numpy as np
from simulator import GOODS

FAIR_TOL, WAIT_GAIN, SELL_LOSS = 0.04, 0.025, -0.005
DRIVER_NAMES = {"cost": "Your cost", "demand": "Demand", "competition": "Competition", "season": "Time of year"}


def inr(v):
    v = float(v)
    if abs(v) < 100:
        s = f"{abs(v):,.1f}"
        return ("-" if v < 0 else "") + "₹" + (s[:-2] if s.endswith(".0") else s)
    s = str(int(round(abs(v))))
    head, tail = s[:-3], s[-3:]
    while len(head) > 2:
        tail, head = head[-2:] + "," + tail, head[:-2]
    return ("-" if v < 0 else "") + "₹" + (head + "," if head else "") + tail


def nice(v):
    step = 50 if v >= 1000 else 5 if v >= 100 else 0.5
    return round(v / step) * step


def confidence_label(p):
    return "high" if p >= 0.7 else "medium" if p >= 0.5 else "low"


def count(qty, good):
    n = GOODS[good]["noun"]
    return f"{int(qty):,} {n}" + ("s" if int(qty) != 1 and n != "kg" else "")


def compare_to_fair(good, user_price, fair, cost, market_price):
    lo, hi = min(fair["low"], fair["price"] * (1 - FAIR_TOL)), max(fair["high"], fair["price"] * (1 + FAIR_TOL))
    gap = user_price / fair["price"] - 1
    status = "underpriced" if user_price < lo else "overpriced" if user_price > hi else "fair"
    floor = cost * 1.05
    sug = {"underpriced": nice(fair["price"]), "overpriced": nice(hi), "fair": user_price}[status]
    return {"status": status, "gap_pct": gap, "margin_pct": (user_price - cost) / user_price, "suggested": max(sug, floor) if status != "fair" else sug,
            "floor": floor, "low": lo, "high": hi, "underwater": fair["price"] < cost * 1.03}


def verdict_line(v):
    gap = abs(v["gap_pct"]) * 100
    if v["status"] == "fair":
        return "Fair price."
    return f"{'Underpriced' if v['status'] == 'underpriced' else 'Above fair price'} by {gap:.0f}%. Suggested price: {inr(v['suggested'])}."


def drivers_table(fair):
    return [(DRIVER_NAMES[n], f"{d * 100:+.1f}%") for n, d in fair["drivers"]]


def sell_or_wait(good, outlook, price_now, qty, horizon):
    g = GOODS[good]
    base = {"horizon": horizon, "qty": qty, "now_value": qty * price_now, "scenarios": None, "risk": ""}
    if horizon == 0:
        return {**base, "action": "sell_now", "label": g["now"], "net_gain_pct": 0.0, "confidence": outlook[1]["confidence"],
                "line": "This week: no time to wait. Use the fair price as your guide."}
    o = outlook[horizon]
    hold = g["hold"] * horizon
    net = o["change_pct"] - hold
    p_pos = 1 - 0.5 * (1 + math.erf(((math.log(1 + hold) - o["mu"]) / o["sd"]) / math.sqrt(2)))
    scen = {"now": qty * price_now, "expected": qty * price_now * (1 + net), "bad": qty * price_now * (1 + o["low_pct"] - hold),
            "good": qty * price_now * (1 + o["high_pct"] - hold)}
    if net >= WAIT_GAIN and p_pos >= 0.6:
        act, label = "wait", "Wait"
    elif net <= SELL_LOSS or p_pos < 0.4:
        act, label = "sell_now", g["now"]
    else:
        act, label = ("split", "Split: sell half now") if qty > 1 else ("sell_now", g["now"])
    conf = o["probs"][o["direction"]]
    return {**base, "action": act, "label": label, "net_gain_pct": net, "confidence": conf, "direction": o["direction"], "scenarios": scen,
            "line": f"{label}. Expected {net * 100:+.0f}% over {horizon} month{'s' if horizon > 1 else ''} after allowing for {g['hold_why']} ({confidence_label(conf)} confidence).",
            "risk": f"Chance that waiting is worse than {g['ing']} today: {min(95, max(5, round((1 - p_pos) * 100)))}%."}


def year_outlook(fair_model, cost, demand, comp, month0):
    months = [(month0 - 1 + i) % 12 + 1 for i in range(12)]
    g = GOODS[fair_model.good]
    prices = fair_model.predict_many(np.full(12, np.clip(cost, 0.5 * g["c0"], 2 * g["c0"])), np.full(12, demand), np.full(12, comp), months)[0]
    return [calendar.month_abbr[m] for m in months], prices
