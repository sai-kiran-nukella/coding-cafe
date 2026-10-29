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


