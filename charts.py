import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from advisor import inr

fmt = FuncFormatter(lambda v, _: inr(v))


def _ax(w, h):
    f, ax = plt.subplots(figsize=(w, h))
    ax.grid(alpha=0.3)
    return f, ax


def price_chart(cost, user, low, high, market, suggested):
    f, ax = _ax(6.5, 2.4)
    ax.axvspan(low, high, color="tab:green", alpha=0.25, label="Fair range")
    for x, lab, c in [(cost, "Your cost", "tab:red"), (market, "Market today", "black"), (user, "Your price", "tab:blue"), (suggested, "Suggested", "tab:orange")]:
        ax.axvline(x, color=c, lw=2, label=f"{lab}: {inr(x)}")
    ax.set_yticks([]); ax.xaxis.set_major_formatter(fmt)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    f.tight_layout()
    return f


def forecast_chart(dates, prices, fair_line, outlook, now):
    f, ax = _ax(6.5, 3.2)
    ax.plot(dates, prices, color="black", label="Market price")
    if len(np.atleast_1d(fair_line)) == len(dates):
        ax.plot(dates, fair_line, "--", color="tab:green", label="Fair price")
    else:
        ax.axhline(float(fair_line), ls="--", color="tab:green", label="Fair price")
    last = dates[-1]
    fut = [last, last + pd.DateOffset(months=1), last + pd.DateOffset(months=2)]
    ax.fill_between(fut, [now] + [now * (1 + outlook[h]["low_pct"]) for h in (1, 2)], [now] + [now * (1 + outlook[h]["high_pct"]) for h in (1, 2)], color="tab:orange", alpha=0.3, label="Likely range")
    ax.plot(fut, [now] + [now * (1 + outlook[h]["change_pct"]) for h in (1, 2)], "o:", color="tab:orange", label="Expected")
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    f.tight_layout()
    return f


def outlook_chart(labels, prices):
    f, ax = _ax(6.5, 2.6)
    colors = ["tab:green" if p == max(prices) else "tab:gray" for p in prices]
    ax.bar(labels, prices, color=colors)
    ax.set_ylim(min(prices) * 0.9, max(prices) * 1.04)
    ax.yaxis.set_major_formatter(fmt)
    f.tight_layout()
    return f
