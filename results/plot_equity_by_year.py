"""Wheel bot vs SPY buy-and-hold, by year.

Reads bench/scorecards/exit-impact-model/live__f92ee3aba856bf77.json (relative to
the repo root) and writes results/equity_by_year.png. arms.variant is the
current config (exit-impact-model), arms.base the frozen base arm before the
exit-cost model, arms.spy_buy_hold is SPY. by_year values are fractional
returns; plotted as %.

    .venv/bin/python results/plot_equity_by_year.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "bench" / "scorecards" / "exit-impact-model" / "live__f92ee3aba856bf77.json"
OUT = ROOT / "results" / "equity_by_year.png"

# ---- chart chrome -----------------------------------------------------------
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASELINE = "#c3c2b7"
SERIES1 = "#2a78d6"; SERIES2 = "#eb6834"; SERIES1_LIGHT = "#86b6ef"; SERIES1_DARK = "#1c5cab"
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "hatch.linewidth": 0.9, "savefig.facecolor": SURFACE,
    })


def figure():
    # 1600 x 900 px at dpi 150
    return plt.figure(figsize=(1600 / 150, 900 / 150), dpi=150)


def px2data(ax, px):
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0, 0)); x1, y1 = inv.transform((px, px))
    return x1 - x0, y1 - y0


def bar(ax, x, h, w, color, r_px=4, hatch=None, hatch_color=None, z=3):
    """Column <= 24px thick, 4px rounded data-end, square at the baseline."""
    if h == 0:
        return
    rx, ry = px2data(ax, r_px)
    y0, hh = (0, h) if h > 0 else (h, -h)
    kw = dict(facecolor=color, edgecolor=hatch_color if hatch else "none",
              linewidth=0, hatch=hatch, zorder=z)
    ax.add_patch(FancyBboxPatch((x - w / 2, y0), w, hh,
                                boxstyle=f"round,pad=0,rounding_size={rx}",
                                mutation_aspect=ry / rx, **kw))
    # square off the baseline end by covering its two rounded corners
    yb = 0 if h > 0 else -ry
    ax.add_patch(Rectangle((x - w / 2, yb), w, ry, **kw))


def strip(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE); ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)


# ---- the chart --------------------------------------------------------------
def main():
    d = json.load(open(SRC)); arms = d["arms"]
    years = sorted(arms["variant"]["by_year"])
    # live tier window is 2024-08-01 -> 2026-07-01 (bench/tiers.py), so partial years:
    ylabel = {"2024": "2024 (Aug\u2013Dec)", "2025": "2025", "2026": "2026 (Jan\u2013Jun)"}
    series = [
        ("variant", "Wheel bot, current config", SERIES1, None),
        ("base", "Wheel bot, before exit-cost model", SERIES1_LIGHT, "///"),
        ("spy_buy_hold", "SPY buy-and-hold", SERIES2, None),
    ]
    vals = {k: [arms[k]["by_year"][y] * 100 for y in years] for k, *_ in series}

    setup(); fig = figure()
    ax = fig.add_axes([0.06, 0.17, 0.92, 0.60])
    ax.set_xlim(-0.5, len(years) - 0.5); ax.set_ylim(-12, 36)
    strip(ax)
    ticks = [-10, 0, 10, 20, 30]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:+d}%" if t else "0%" for t in ticks])
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(range(len(years))); ax.set_xticklabels([ylabel.get(y, y) for y in years], fontsize=11, color=INK2)
    ax.tick_params(axis="x", pad=8)

    wx, _ = px2data(ax, 24); gx, _ = px2data(ax, 30)  # 30px air between bars so each cap label clears its neighbour
    table = []
    for i, y in enumerate(years):
        for j, (k, label, color, hatch) in enumerate(series):
            v = vals[k][i]; x = i + (j - 1) * (wx + gx)
            bar(ax, x, v, wx, color, hatch=hatch, hatch_color=SERIES1_DARK if hatch else None)
            ax.text(x, v + (1.0 if v >= 0 else -1.0), f"{v:+.1f}%", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9, color=INK2)
            table.append((k, y, round(v, 2)))
    ax.legend(handles=[Patch(facecolor=c, hatch=h, edgecolor=SERIES1_DARK if h else "none", linewidth=0, label=l)
                       for _, l, c, h in series],
              loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False, fontsize=10, labelcolor=INK2,
              handlelength=1.0, handleheight=1.0, borderaxespad=0, columnspacing=1.6)

    fig.text(0.06, 0.93, "Wheel bot vs SPY buy-and-hold, by year", fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.875, "Calendar-year return, live tier: 2024-08-01 to 2026-07-01, 531 names, paper fills at Schwab bid/ask",
             fontsize=11, color=INK2, va="top")
    fig.text(0.06, 0.04, f"Scorecard {SRC.name} (run {d['created'][:10]}, engine {d['provenance']['engine']['sha']}). "
             "Current config = exit-impact-model variant; \"before\" = frozen base arm.\n2024 and 2026 are partial years (the window opens 2024-08-01 and closes 2026-07-01).",
             fontsize=8.5, color=MUTED, va="bottom")
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT.relative_to(ROOT))
    for t in table:
        print(t)


if __name__ == "__main__":
    main()
