import matplotlib
matplotlib.use("Agg")
import calendar
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from advisor import money


def _ax(w, h, cur):
    f, ax = plt.subplots(figsize=(w, h))
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: money(v, cur)))
    return f, ax


def price_chart(cost, user, low, high, market, suggested, cur="$"):
    f, ax = plt.subplots(figsize=(6.5, 2.4))
    ax.grid(alpha=0.3)
    ax.axvspan(low, high, color="tab:green", alpha=0.25, label="Fair range")
    for x, lab, c in [(cost, "Your cost", "tab:red"), (market, "Market today", "black"), (user, "Your price", "tab:blue"), (suggested, "Suggested", "tab:orange")]:
        ax.axvline(x, color=c, lw=2, label=f"{lab}: {money(x, cur)}")
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: money(v, cur)))
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    f.tight_layout()
    return f


def history_chart(hist, years, cur="$", title=""):
    h = hist.tail(years * 12)
    f, ax = _ax(6.8, 3.2, cur)
    ax.plot(h["date"], h["price"], color="black", lw=1.4, label="Market price")
    ax.plot(h["date"], h["fair"], "--", color="tab:green", lw=2, label="Model's fair price")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    ax.set_title(title, fontsize=9)
    f.tight_layout()
    return f


def forecast_chart(hist, outlook, now, cur="$"):
    h = hist.tail(24)
    f, ax = _ax(6.5, 3.2, cur)
    last = h["date"].iloc[-1]
    ax.plot(h["date"], h["price"], color="black", label="Market price")
    ax.plot(h["date"], h["fair"], "--", color="tab:green", label="Fair price")
    fut = [last] + [last + pd.DateOffset(months=k) for k in (1, 2, 3)]
    ax.fill_between(fut, [now] + [now * (1 + outlook[k]["low_pct"]) for k in (1, 2, 3)], [now] + [now * (1 + outlook[k]["high_pct"]) for k in (1, 2, 3)], color="tab:orange", alpha=0.3, label="Likely range (80%)")
    ax.plot(fut, [now] + [now * (1 + outlook[k]["change_pct"]) for k in (1, 2, 3)], "o:", color="tab:orange", label="Expected")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    f.tight_layout()
    return f


def season_chart(season):
    f, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.grid(alpha=0.3, axis="y")
    vals = [100 * season[m] for m in range(1, 13)]
    ax.bar([calendar.month_abbr[m] for m in range(1, 13)], vals, color=["tab:green" if v == max(vals) else "tab:red" if v == min(vals) else "tab:gray" for v in vals])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("% above or below fair")
    f.tight_layout()
    return f


def test_chart(test, cur="$"):
    f, ax = _ax(6.8, 3.2, cur)
    ax.plot(test["date"], test["price"], color="black", lw=1.2, label="Market price")
    ax.plot(test["date"], test["realized"], color="tab:blue", lw=1.5, label="Where prices actually settled (next 12 months)")
    ax.plot(test["date"], test["estimate"], "--", color="tab:green", lw=2, label="Model's fair price at the time")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    f.tight_layout()
    return f
