import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import advisor as A
import charts
import learning_demo as L
from market_feed import SimulatedFeed, read_price_csv
from models import load_or_train
from simulator import COMP_OPTIONS, DEMAND_OPTIONS, EVENTS, GOODS, nearest_option

st.set_page_config(page_title="PriceWise", layout="wide")
URGENCY = {"Flexible (up to 2 months)": 2, "Within a month": 1, "This week": 0}
month = datetime.date.today().month


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


@st.cache_resource(show_spinner="Loading models...")
def get_bundle(good):
    return load_or_train(good)


@st.cache_resource
def get_feed(good, stamp):
    return SimulatedFeed(good)


S = st.sidebar
S.header("Your product")
good = S.selectbox("Product", list(GOODS))
g, feed, bundle = GOODS[good], get_feed(good, datetime.date.today().strftime("%Y-%m")), get_bundle(good)
latest, step = feed.latest(), (1.0 if GOODS[good]["p0"] >= 100 else 0.5)
cost = S.number_input("Your cost per " + g["noun"] + " (₹)", min_value=1.0, value=float(round(latest["cost"])), step=step, key=f"cost_{good}")
price = S.number_input("Your asking price (₹)", min_value=1.0, value=float(round(latest["price"])), step=step, key=f"price_{good}")
qty = S.number_input("Quantity", min_value=1, value=int(g["qty"]), step=1, key=f"qty_{good}")
dl, cl = list(DEMAND_OPTIONS), list(COMP_OPTIONS)
demand = S.selectbox("Demand", dl, index=dl.index(nearest_option(DEMAND_OPTIONS, latest["demand"])), key=f"d_{good}")
comp = S.selectbox("Competition", cl, index=cl.index(nearest_option(COMP_OPTIONS, latest["comp"])), key=f"c_{good}")
urgency = S.selectbox("Need to sell", list(URGENCY), key=f"u_{good}")
upload = S.file_uploader("Your monthly prices (CSV, optional)", type="csv")

hist = feed.history(24)
prices, dates, own = hist["price"].values, list(hist["date"]), False
if upload is not None:
    try:
        prices, own = np.array(read_price_csv(upload)), True
        dates = [hist["date"].iloc[-1] - pd.DateOffset(months=len(prices) - 1 - i) for i in range(len(prices))]
    except ValueError as err:
        S.error(str(err))
market = float(prices[-1])

fair = bundle.fair.predict(cost, DEMAND_OPTIONS[demand], COMP_OPTIONS[comp], month)
v = A.compare_to_fair(good, price, fair, cost, market)
trend = bundle.trend.predict(prices, fair["price"], feed.cost_momentum(), month)
rec = A.sell_or_wait(good, trend, market, int(qty), URGENCY[urgency])

st.title("PriceWise")
st.caption("Check whether your price is fair and whether to sell now or wait. Prices are simulated.")

st.header("1. Is your price fair?")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fair price", A.inr(fair["price"]), f"{A.inr(v['low'])} to {A.inr(v['high'])}", delta_color="off")
c2.metric("Market price today", A.inr(market))
c3.metric("Your margin", f"{v['margin_pct'] * 100:.0f}%")
c4.metric("Profit at your price", A.inr((price - cost) * qty), f"{A.inr((v['suggested'] - cost) * qty)} at suggested", delta_color="off")
{"underpriced": st.warning, "fair": st.success, "overpriced": st.error}[v["status"]](A.verdict_line(v))
if v["underwater"]:
    st.caption("The fair price is within 3% of your cost.")
left, right = st.columns([3, 2])
with left:
    show(charts.price_chart(cost, price, v["low"], v["high"], market, v["suggested"]))
with right:
    st.table(pd.DataFrame(A.drivers_table(fair), columns=["Factor", "Effect on fair price"]).set_index("Factor"))

st.header("2. Sell now or wait?")
st.info(rec["line"])
if rec["risk"]:
    st.caption(rec["risk"])
left, right = st.columns(2)
with left:
    if rec["scenarios"]:
        s, h = rec["scenarios"], URGENCY[urgency]
        st.table(pd.DataFrame({"Income": [A.inr(s["now"]), A.inr(s["expected"]), A.inr(s["bad"]), A.inr(s["good"])]},
                              index=[g["now"], f"Wait {h} month{'s' if h > 1 else ''} (expected)", "Wait (bad case)", "Wait (good case)"]))
    labels, op = A.year_outlook(bundle.fair, cost, DEMAND_OPTIONS[demand], COMP_OPTIONS[comp], month)
    show(charts.outlook_chart(labels, op))
    st.caption(f"Highest fair price: {labels[int(np.argmax(op))]} ({A.inr(op.max())}), if conditions stay as they are.")
with right:
    fl = bundle.fair.predict_many(hist.cost, hist.demand, hist.comp, hist.month)[0] if not own else fair["price"]
    show(charts.forecast_chart(dates, prices, fl, trend, market))

st.header("3. How the model learned")
t1, t2 = st.tabs(["Training", "Shock test"])
with t1:
    st.write("Trains a new network from scratch on simulated months of this market.")
    run = st.button("Train", key="train")
    info, c_err, c_fit = st.empty(), st.empty(), st.empty()

    def draw(fr, hist_df):
        info.caption(f"Epoch {fr['epoch']}: network error {fr['ape'] * 100:.1f}%, straight-line formula {fr['linear_ape'] * 100:.1f}%")
        c_err.line_chart(hist_df, height=220)
        c_fit.line_chart(pd.DataFrame({"Real fair price": fr["true"], "Network": fr["pred"]}), height=220)

    if run:
        rows = []
        for fr in L.train_live(good):
            rows.append({"epoch": fr["epoch"], "Network error %": fr["ape"] * 100, "Straight-line error %": fr["linear_ape"] * 100})
            draw(fr, pd.DataFrame(rows).set_index("epoch"))
            time.sleep(0.12)
        st.session_state["train_last"] = (good, fr, pd.DataFrame(rows).set_index("epoch"))
    elif st.session_state.get("train_last", (None,))[0] == good:
        draw(*st.session_state["train_last"][1:])
with t2:
    ev = st.selectbox("Shock", [e[0] for e in EVENTS], index=3)
    run2 = st.button("Run shock test", key="shock")
    sum_ph, ch1, ch2 = st.empty(), st.empty(), st.empty()

    def draw2(d, n, thr, retr):
        d = d.iloc[:n]
        ch1.line_chart(d.set_index("month")[["true", "no_repair", "repair"]].rename(columns={"true": "Real price", "no_repair": "Without repair", "repair": "With repair"}), height=240)
        ch2.line_chart(d.set_index("month")[["err_plain", "err_repair"]].mul(100).rename(columns={"err_plain": "Error without repair %", "err_repair": "Error with repair %"}), height=200)
        sum_ph.caption(f"Month {n}. Retrained {int(d.retrained.sum())} times. Peak error: {d.err_plain.max() * 100:.0f}% without repair, {d.err_repair.max() * 100:.0f}% with repair.")

    if run2:
        d, thr, retr = L.shock_test(good, [e[0] for e in EVENTS].index(ev))
        for n in range(1, len(d) + 1):
            draw2(d, n, thr, retr)
            time.sleep(0.07)
        st.session_state["shock_last"] = (good, ev, d, thr, retr)
    elif st.session_state.get("shock_last", (None, None))[:2] == (good, ev):
        _, _, d, thr, retr = st.session_state["shock_last"]
        draw2(d, len(d), thr, retr)

result = pd.DataFrame([{"product": good, "cost": cost, "your_price": price, "fair_price": round(fair["price"], 2), "fair_low": round(v["low"], 2),
                        "fair_high": round(v["high"], 2), "verdict": v["status"], "suggested_price": round(v["suggested"], 2), "advice": rec["label"]}])
S.download_button("Download result (CSV)", result.to_csv(index=False), "pricewise_result.csv", "text/csv")
