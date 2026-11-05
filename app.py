import datetime
import json
import pickle
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import advisor as A
import charts
import data_loader as D
import engine as E
import learning_demo as L
from sources import CPI, CUSTOM, MARKETS

st.set_page_config(page_title="PriceWise", layout="wide")
URGENCY = {"Flexible (up to 3 months)": 3, "Within 2 months": 2, "Within a month": 1, "This week": 0}
CACHE, VERSION, SUMMARY = Path(__file__).parent / ".cache", 1, D.DATA / "summary.json"
ss = st.session_state


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def link(label, url):
    st.link_button(label, url) if hasattr(st, "link_button") else st.markdown(f"[{label}]({url})")


@st.cache_resource(show_spinner=False)
def get_analysis(dig, _md):
    f = CACHE / f"analysis_v{VERSION}_{dig}.pkl"
    try:
        if f.exists():
            return pickle.loads(f.read_bytes())
    except Exception:
        pass
    a = E.analyze(_md)
    try:
        CACHE.mkdir(exist_ok=True)
        f.write_bytes(pickle.dumps(a))
    except OSError:
        pass
    return a


def remember(name, a, cfg):
    try:
        s = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
        o = a["outlook"][2]
        s[name] = {"Latest data": f"{a['now']['date']:%b %Y}", "Price": A.money(a["now"]["price"], cfg["currency"]), "Fair price": A.money(a["now"]["fair"], cfg["currency"]),
                   "Price vs fair": f"{a['now']['gap'] * 100:+.0f}%", "Next 2 months": f"{o['change_pct'] * 100:+.1f}% ({o['direction']}, {A.confidence_label(o['confidence'])} confidence)"}
        D.DATA.mkdir(exist_ok=True)
        SUMMARY.write_text(json.dumps(s))
    except Exception:
        pass


# ---------------------------------------------------------------- sidebar: product
S = st.sidebar
S.header("Product")
choice = S.selectbox("Market", list(MARKETS) + ["Custom file"])
custom = choice == "Custom file"
cfg = dict(CUSTOM if custom else MARKETS[choice])
if custom:
    cfg["noun"] = S.text_input("Unit (for example kg)", "unit")
    cfg["hold"] = S.number_input("Cost of holding stock (% of value per month)", 0.0, 20.0, 1.0, 0.1) / 100
    cfg["currency"] = S.text_input("Currency symbol", "$")
    choice = S.text_input("Name of this market", "My market") or "My market"
key, cur = cfg["series"], cfg["currency"]

st.title("PriceWise")
st.caption("Check whether your price is fair and whether to sell now or wait, using real market data.")

# ---------------------------------------------------------------- data file
st.header("Data")
c1, c2 = st.columns([3, 2])
with c1:
    if custom:
        st.write("Upload a CSV or Excel file with a date column and a price column. Monthly or weekly data, at least 10 years.")
    else:
        st.markdown(f"**Recommended file:** {cfg['title']}")
        st.caption(f"{cfg['publisher']} | {cfg['frequency']} | US dollars {cfg['per']}")
        st.caption(cfg["why"])
        a, b = st.columns(2)
        with a:
            link("Open source page", cfg["page"])
        with b:
            link("Direct file download", cfg["csv"])
        st.caption("Format: " + cfg["fmt"])
        st.caption("Cite as: " + cfg["citation"])
with c2:
    if not custom and st.button("Download the recommended file", key=f"dl_{key}"):
        try:
            raw = D.fetch(cfg["csv"])
            D.read_series("x.csv", raw)
            D.save(key, raw, "Downloaded from the source", cfg["csv"])
            st.rerun()
        except D.DataError as e:
            st.error(str(e))
    up = st.file_uploader("Or use a different file (CSV or Excel)", type=["csv", "xlsx", "xls", "txt"], key=f"up_{key}")
    if up is not None:
        try:
            D.save(key + "_mine", D.normalise_upload(up.name, up.getvalue()), "Uploaded by you")
        except D.DataError as e:
            st.error(str(e))
rec, mine = (None if custom else D.saved(key)), D.saved(key + "_mine")
options = [n for n, v in (("Recommended file", rec), ("My uploaded file", mine)) if v]
if not options:
    st.info("Download the recommended file or upload your own to begin.")
    st.stop()
use = options[0] if len(options) == 1 else st.radio("File in use", options, horizontal=True)
active = rec if use == "Recommended file" else mine

cpi = D.saved("CPIAUCSL")
if cpi is None and not ss.get("cpi_tried"):
    ss["cpi_tried"] = True
    try:
        raw = D.fetch(CPI["csv"])
        D.read_series("c.csv", raw)
        D.save("CPIAUCSL", raw, "Downloaded from the source", CPI["csv"])
        cpi = D.saved("CPIAUCSL")
    except D.DataError:
        pass
if cpi is None:
    with st.expander("Inflation file needed to adjust prices for inflation", expanded=True):
        st.write(f"{CPI['title']} ({CPI['publisher']})")
        link("Direct file download", CPI["csv"])
        cu = st.file_uploader("Upload the inflation file", type=["csv"], key="cpi_up")
        if cu is not None:
            try:
                D.read_series("c.csv", cu.getvalue())
                D.save("CPIAUCSL", cu.getvalue(), "Uploaded by you")
                st.rerun()
            except D.DataError as e:
                st.error(str(e))
use_infl = S.checkbox("Adjust for inflation", value=cpi is not None, disabled=cpi is None)
try:
    md = D.build_market(active["name"], active["bytes"], cpi["bytes"] if (cpi and use_infl) else None)
except D.DataError as e:
    st.error(str(e))
    st.stop()
info = md["info"]
st.caption(f"{info['months']} months of data, {info['first']:%b %Y} to {info['last']:%b %Y}"
           + (f", weekly values averaged to months" if info["weekly"] else "") + (f", {info['filled']} missing months filled in" if info["filled"] else "")
           + f". Inflation adjusted: {'yes' if info['inflation'] else 'no'}. Source file saved {active['meta'].get('fetched', '')}.")
if info["stale_months"] > 3:
    st.warning(f"The latest data point is {info['last']:%b %Y}, {info['stale_months']} months ago. Download a newer file for current advice.")
if not info["inflation"]:
    st.warning("Prices are not adjusted for inflation, so the fair price will follow the long-run drift of prices.")

dig = D.digest(active["bytes"]) + (D.digest(cpi["bytes"]) if info["inflation"] else "n")
with st.spinner(f"Learning this market from {info['months']} months of data (about 10 seconds the first time)..."):
    A_ = get_analysis(dig, md)
remember(choice, A_, cfg)
now = A_["now"]

# ---------------------------------------------------------------- sidebar: your numbers
S.header("Your numbers")
stp = float(10 ** (np.floor(np.log10(now["price"])) - 2))
cost = S.number_input(f"Your cost per {cfg['noun']} ({cur})", min_value=0.0, value=round(0.6 * now["price"], 2), step=stp, key=f"cost_{choice}_{dig}")
price = S.number_input(f"Your asking price per {cfg['noun']} ({cur})", min_value=0.0, value=round(now["price"], 2), step=stp, key=f"price_{choice}_{dig}")
qty = S.number_input(f"Quantity ({cfg['noun']}s)", min_value=1.0, value=float(cfg["qty"]), step=1.0, key=f"qty_{choice}")
urgency = S.selectbox("Need to sell", list(URGENCY), key=f"urg_{choice}")
local = S.number_input("Your local price vs this benchmark (%)", -50.0, 100.0, 0.0, 1.0, key=f"loc_{choice}", help="Enter +5 if your market usually trades 5% above this series.")
if price <= 0 or cost < 0:
    st.error("Enter a positive asking price.")
    st.stop()
f = 1 + local / 100
fair = {"price": now["fair"] * f, "low": now["low"] * f, "high": now["high"] * f}
market = now["price"] * f
v = A.compare_to_fair(price, fair, cost, market)
rec_ = A.sell_or_wait(cfg, A_["outlook"], market, qty, URGENCY[urgency])

# ---------------------------------------------------------------- 1. fair price
st.header("1. Is your price fair?")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Fair price", A.money(fair["price"], cur), f"{A.money(v['low'], cur)} to {A.money(v['high'], cur)}", delta_color="off")
m2.metric("Market price today", A.money(market, cur), f"{now['gap'] * 100:+.0f}% vs fair", delta_color="off")
m3.metric("Your margin", f"{v['margin_pct'] * 100:.0f}%")
m4.metric("Profit at your price", A.money((price - cost) * qty, cur), f"{A.money((v['suggested'] - cost) * qty, cur)} at suggested", delta_color="off")
{"underpriced": st.warning, "fair": st.success, "overpriced": st.error}[v["status"]](A.verdict_line(v, cur))
if v["underwater"]:
    st.caption("The fair price is within 3% of your cost.")
left, right = st.columns([3, 2])
with left:
    show(charts.price_chart(cost, price, v["low"], v["high"], market, v["suggested"], cur))
    yrs = st.radio("History shown", [5, 10, 20], index=1, horizontal=True, format_func=lambda y: f"{y} years")
    show(charts.history_chart(A_["hist"], yrs, cur, "Market price and the model's fair price"))
with right:
    hl, mo = A_["half_life"], datetime.date.today().month
    rows = [("Fair price (long-run, inflation-adjusted)", A.money(now["fair"], cur)), ("Price vs fair today", f"{now['gap'] * 100:+.1f}%"),
            ("Gap halves in about", f"{hl:.1f} months" if hl else "no clear reversion"), (f"Seasonal effect in {datetime.date(2000, mo, 1):%B}", f"{A_['season'][mo] * 100:+.1f}%"),
            ("Inflation assumed", f"{A_['infl_annual'] * 100:.1f}% per year")]
    rows += [(f"Lasting shift {s['start']:%b %Y} to {s['end']:%b %Y}", f"{s['change'] * 100:+.0f}% in fair price") for s in A_["shifts"]]
    st.table(pd.DataFrame(rows, columns=["What the data says", ""]).set_index("What the data says"))

# ---------------------------------------------------------------- 2. sell or wait
st.header("2. Sell now or wait?")
st.info(rec_["line"])
if rec_["risk"]:
    st.caption(rec_["risk"])
left, right = st.columns(2)
with left:
    if rec_["scenarios"]:
        s, h = rec_["scenarios"], URGENCY[urgency]
        st.table(pd.DataFrame({f"Income for {A.count(qty, cfg)}": [A.money(s["now"], cur), A.money(s["expected"], cur), A.money(s["bad"], cur), A.money(s["good"], cur)]},
                              index=[cfg["now"], f"Wait {h} month{'s' if h > 1 else ''} (expected)", "Wait (bad case)", "Wait (good case)"]))
    show(charts.season_chart(A_["season"]))
    best = max(A_["season"], key=A_["season"].get)
    st.caption(f"Seasonally strongest month: {datetime.date(2000, best, 1):%B} ({A_['season'][best] * 100:+.1f}% vs fair)." if A_["season_gain"] >= 0.04
               else "No reliable seasonal pattern in this market.")
with right:
    show(charts.forecast_chart(A_["hist"], A_["outlook"], now["price"], cur))
    mt = A_["move_table"]
    st.caption("Trust in the forecast (from backtests): " + ", ".join(f"{int(r['Horizon (months)'])} month {r['Trust in forecast %']:.0f}%" for _, r in mt.iterrows()))

# ---------------------------------------------------------------- 3. how it learned
st.header("3. How the model learned")
P = E.prepare(md)
t1, t2, t3 = st.tabs(["Backtest", "Live training", "Shock replay"])
with t1:
    ft = A_["fair_table"]
    st.write(f"Tested on {A_['windows']} windows of history the model had not seen. Error is the average gap between the fair price estimate and where prices actually settled over the next 12 months.")
    a, b = st.columns(2)
    with a:
        st.table(pd.DataFrame({"Average error": [f"{100 * ft[k]:.1f}%" for k in ft]}, index=list(ft)).rename_axis("Method"))
        st.caption("Blend weights: " + ", ".join(f"{k} {100 * w:.0f}%" for k, w in A_["weights"].items() if w >= 0.01))
    with b:
        show(charts.test_chart(A_["test"], cur))
    cols = ["Horizon (months)", "Forecast error %", "'No change' error %", "Direction right %", "Direction right % (momentum)", "Direction right % (always flat)", "Trust in forecast %"]
    st.table(mt[cols].round(1).assign(**{"Horizon (months)": mt["Horizon (months)"].astype(int)}).set_index("Horizon (months)"))
with t2:
    st.write("Trains a new network from scratch on this market and shows its accuracy on months it has never seen.")
    go = st.button("Train", key="train")
    info_ph, ch1, ch2 = st.empty(), st.empty(), st.empty()

    def draw(fr, hist_df):
        info_ph.caption(f"Epoch {fr['epoch']}: network error {fr['err']:.1f}%, 12-month average {fr['base']['12-month average']:.1f}%, last price {fr['base']['Last price']:.1f}%")
        ch1.line_chart(hist_df, height=220)
        ch2.line_chart(pd.DataFrame({"Market price": fr["price"], "Settled price": fr["realized"], "Network's fair price": fr["estimate"]}, index=fr["dates"]), height=240)

    if go:
        rows_ = []
        for fr in L.train_live(P):
            rows_.append({"epoch": fr["epoch"], "Network error %": fr["err"], "12-month average %": fr["base"]["12-month average"], "Last price %": fr["base"]["Last price"]})
            draw(fr, pd.DataFrame(rows_).set_index("epoch"))
            time.sleep(0.1)
        ss["train_last"] = (dig, fr, pd.DataFrame(rows_).set_index("epoch"))
    elif ss.get("train_last", (None,))[0] == dig:
        draw(*ss["train_last"][1:])
with t3:
    evs = L.shock_events(P)
    if not evs:
        st.write("Not enough history to replay a shock.")
    else:
        st.write("Replays a real shock in this market. One network stays fixed. The other tries to retrain when its errors grow, and keeps the update only if it helps on the newest months.")
        pick = st.selectbox("Shock", range(len(evs)), format_func=lambda k: f"{evs[k]['date']:%b %Y}: price {evs[k]['gap'] * 100:+.0f}% from fair")
        go2 = st.button("Replay", key="replay")
        sum_ph, r1, r2 = st.empty(), st.empty(), st.empty()

        def draw2(d, n, fl, kept):
            d = d.iloc[:n].set_index("date")
            r1.line_chart(d[["settled", "frozen", "updating"]].rename(columns={"settled": "Settled price", "frozen": "Fixed network", "updating": "Updating network"}), height=240)
            r2.line_chart(d[["err_frozen", "err_updating"]].mul(100).rename(columns={"err_frozen": "Fixed network error %", "err_updating": "Updating network error %"}), height=200)
            sum_ph.caption(f"{d.index[-1]:%b %Y}. Retraining attempted {fl} times, kept {kept}. Average error: fixed {100 * d.err_frozen.mean():.1f}%, updating {100 * d.err_updating.mean():.1f}%.")

        if go2:
            d, fl, kept = L.replay(P, evs[pick]["i"])
            for n_ in range(1, len(d) + 1):
                draw2(d, n_, int(d.flagged.iloc[:n_].sum()), int(d.retrained.iloc[:n_].sum()))
                time.sleep(0.07)
            ss["replay_last"] = (dig, pick, d, fl, kept)
        elif ss.get("replay_last", (None, None))[:2] == (dig, pick):
            _, _, d, fl, kept = ss["replay_last"]
            draw2(d, len(d), fl, kept)

# ---------------------------------------------------------------- overview and export
if SUMMARY.exists():
    try:
        st.header("Markets analysed so far")
        st.table(pd.DataFrame(json.loads(SUMMARY.read_text())).T.rename_axis("Market"))
    except Exception:
        pass
res = pd.DataFrame([{"market": choice, "data_to": f"{now['date']:%Y-%m}", "your_cost": cost, "your_price": price, "fair_price": round(fair["price"], 4), "fair_low": round(v["low"], 4),
                     "fair_high": round(v["high"], 4), "market_price": round(market, 4), "verdict": v["status"], "suggested_price": round(v["suggested"], 4), "advice": rec_["label"]}])
S.download_button("Download result (CSV)", res.to_csv(index=False), "pricewise_result.csv", "text/csv")
