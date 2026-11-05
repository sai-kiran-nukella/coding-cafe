import numpy as np

FAIR_TOL, WAIT_GAIN, SELL_LOSS = 0.04, 0.025, -0.005


def money(v, cur="$"):
    v = float(v)
    s = f"{abs(v):,.2f}" if abs(v) < 1000 else f"{abs(v):,.0f}"
    return ("-" if v < 0 else "") + cur + s


def nice(v):
    step = 0.05 if v < 10 else 0.5 if v < 100 else 1 if v < 1000 else 10
    return round(v / step) * step


def confidence_label(p):
    return "high" if p >= 0.7 else "medium" if p >= 0.5 else "low"


def count(qty, cfg):
    n = cfg["noun"]
    return f"{qty:,.0f} {n}" + ("s" if qty != 1 and n not in ("lb",) else "")


def compare_to_fair(user_price, fair, cost, market_price):
    lo, hi = min(fair["low"], fair["price"] * (1 - FAIR_TOL)), max(fair["high"], fair["price"] * (1 + FAIR_TOL))
    gap = user_price / fair["price"] - 1
    status = "underpriced" if user_price < lo else "overpriced" if user_price > hi else "fair"
    floor = cost * 1.05
    sug = {"underpriced": nice(fair["price"]), "overpriced": nice(hi), "fair": user_price}[status]
    return {"status": status, "gap_pct": gap, "margin_pct": (user_price - cost) / user_price, "suggested": max(sug, floor) if status != "fair" else sug,
            "floor": floor, "low": lo, "high": hi, "underwater": fair["price"] < cost * 1.03}


def verdict_line(v, cur="$"):
    if v["status"] == "fair":
        return "Fair price."
    return f"{'Underpriced' if v['status'] == 'underpriced' else 'Above fair price'} by {abs(v['gap_pct']) * 100:.0f}%. Suggested price: {money(v['suggested'], cur)}."


def sell_or_wait(cfg, outlook, price_now, qty, horizon):
    base = {"horizon": horizon, "qty": qty, "scenarios": None, "risk": ""}
    if horizon == 0:
        return {**base, "action": "sell_now", "label": cfg["now"], "net_gain_pct": 0.0, "confidence": outlook[1]["confidence"],
                "line": "This week: no time to wait. Use the fair price as your guide."}
    o, hold = outlook[horizon], cfg["hold"] * horizon
    net = o["change_pct"] - hold
    p_pos = float(np.mean(o["mu"] + np.array(o["resid"]) > np.log1p(hold)))
    scen = {"now": qty * price_now, "expected": qty * price_now * (1 + net), "bad": qty * price_now * (1 + o["low_pct"] - hold), "good": qty * price_now * (1 + o["high_pct"] - hold)}
    if net >= WAIT_GAIN and p_pos >= 0.6:
        act, label = "wait", "Wait"
    elif net <= SELL_LOSS or p_pos < 0.4:
        act, label = "sell_now", cfg["now"]
    else:
        act, label = ("split", "Split: sell half now") if qty > 1 else ("sell_now", cfg["now"])
    conf = o["probs"][o["direction"]]
    return {**base, "action": act, "label": label, "net_gain_pct": net, "confidence": conf, "direction": o["direction"], "scenarios": scen,
            "line": f"{label}. Expected {net * 100:+.0f}% over {horizon} month{'s' if horizon > 1 else ''} after allowing for {cfg['hold_why']} ({confidence_label(conf)} confidence).",
            "risk": f"Chance that waiting is worse than {cfg['ing']} today: {min(95, max(5, round((1 - p_pos) * 100)))}%."}
