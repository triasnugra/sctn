"""Chart candlestick (matplotlib, headless) dengan anotasi level, entry zone, stop, target."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ..config_loader import Config  # noqa: E402
from ..models import Signal, SymbolAnalysis  # noqa: E402
from ..utils import fmt_price  # noqa: E402

log = logging.getLogger("engine.chart")
PAD = 28


def generate_chart(an: SymbolAnalysis, sig: Signal, cfg: Config, out_path: Path) -> bool:
    ccfg = cfg.params.get("chart", {})
    tf = ccfg.get("timeframe", "1h")
    if ccfg.get("follow_trigger_tf", True) and sig.primary is not None and sig.primary.tf in an.frames:
        tf = sig.primary.tf
    if tf not in an.frames:
        tf = cfg.tfs["primary_trigger"]
    df = an.frames[tf].iloc[-int(ccfg.get("candles", 160)):]
    n = len(df)
    if n < 20:
        return False
    x = np.arange(n)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=int(ccfg.get("dpi", 110)))
    up = c >= o
    col = np.where(up, "#26a69a", "#ef5350")
    ax.vlines(x, l, h, colors=col, linewidth=0.8, zorder=3)
    ax.bar(x, np.maximum(np.abs(c - o), c.mean() * 1e-5), bottom=np.minimum(o, c), width=0.6, color=col, zorder=4)

    ymin, ymax = float(l.min()), float(h.max())
    plan = sig.plan
    if plan:
        ymin, ymax = min(ymin, plan.stop), max(ymax, plan.tp1)
        ymin, ymax = min(ymin, plan.tp1), max(ymax, plan.stop)
    span = ymax - ymin
    lo_v, hi_v = ymin - 0.15 * span, ymax + 0.15 * span

    def to_x(ts):
        return int(min(max(df.index.searchsorted(ts), 0), n - 1))

    labels = []

    def hline(price, label, color, ls="--", lw=1.0):
        if not (lo_v <= price <= hi_v):
            return
        ax.hlines(price, 0, n + 1, colors=color, linestyles=ls, linewidth=lw, zorder=2)
        labels.append([price, f"{label} {fmt_price(price)}", color])

    # range box (TF chart) & range TF lain
    for rtf, rb in an.ranges.items():
        if rtf == tf:
            x0 = to_x(rb.start_time)
            ax.add_patch(Rectangle((x0, rb.low), n - x0, rb.height, color="#9e9e9e", alpha=0.10, zorder=1))
            ax.add_patch(Rectangle((x0, rb.low), n - x0, rb.demand_top - rb.low, color="#26a69a", alpha=0.12, zorder=1))
            ax.add_patch(Rectangle((x0, rb.supply_bottom), n - x0, rb.high - rb.supply_bottom, color="#ef5350", alpha=0.12, zorder=1))
            hline(rb.high, f"Range H {rtf}", "#616161", "-", 1.2)
            hline(rb.low, f"Range L {rtf}", "#616161", "-", 1.2)
            hline(rb.mid, "Mid", "#9e9e9e", ":", 0.8)
        else:
            hline(rb.high, f"Range H {rtf}", "#7e57c2", ":", 1.0)
            hline(rb.low, f"Range L {rtf}", "#7e57c2", ":", 1.0)
    if an.monday:
        hline(an.monday.high, "Monday H", "#1e88e5", "--")
        hline(an.monday.low, "Monday L", "#1e88e5", "--")
    for fs in an.fibs:
        if fs.tf == tf and fs.state != "invalidated":
            ax.add_patch(Rectangle((to_x(fs.b_time), fs.zone_lo), n - to_x(fs.b_time), fs.zone_hi - fs.zone_lo,
                                   color="#ffb300", alpha=0.18, zorder=1))
            ax.plot([to_x(fs.a_time), to_x(fs.b_time)], [fs.a_price, fs.b_price], color="#ffb300", lw=1.2, ls="-.", zorder=2)
            hline(fs.level_price, "Fib 0.75", "#ff8f00", "--")
    near_fvgs = sorted([z for z in an.fvgs if z.tf == tf], key=lambda z: abs((z.low + z.high) / 2 - an.price))[:4]
    for z in near_fvgs:
        if True:
            x0 = to_x(z.time)
            ax.add_patch(Rectangle((x0, z.low), n - x0, z.high - z.low,
                                   color="#26a69a" if z.direction == "LONG" else "#ef5350", alpha=0.20, zorder=1))
    # trigger marker
    if sig.primary is not None and sig.primary.time is not None and sig.primary.tf == tf:
        xi = to_x(sig.primary.time)
        ax.annotate(sig.primary.kind.replace("_", " "), xy=(xi, c[xi]), xytext=(xi - 12, c[xi] + (0.06 if sig.primary.direction == "SHORT" else -0.06) * span),
                    arrowprops=dict(arrowstyle="->", color="black"), fontsize=8, zorder=6)
    # plan
    if plan:
        if plan.entry_zone:
            ax.axhspan(plan.entry_zone[0], plan.entry_zone[1], color="#1976d2", alpha=0.18, zorder=1)
        hline(plan.entry, "ENTRY", "#1976d2", "-", 1.6)
        hline(plan.stop, f"STOP ({plan.stop_pct:.2f}%)", "#d32f2f", "-", 1.6)
        hline(plan.tp1, f"TP1 ({plan.rr1:.1f}R)", "#2e7d32", "-", 1.4)
        hline(plan.tp2, f"TP2 ({plan.rr2:.1f}R)", "#66bb6a", "--", 1.2)
    # de-overlap label di margin kanan
    labels.sort(key=lambda t: t[0])
    gap = (hi_v - lo_v) * 0.028
    ys = [t[0] for t in labels]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    for (price, text, color), y in zip(labels, ys):
        ax.text(n + 1.5, y, text, fontsize=7.5, va="center", color=color)
        if abs(y - price) > 1e-12:
            ax.plot([n + 1, n + 1.4], [price, y], color=color, lw=0.6)
    ax.set_xlim(-1, n + PAD)
    ax.set_ylim(lo_v, hi_v)
    step = max(1, n // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([df.index[i].strftime("%m-%d %H:%M") for i in x[::step]], rotation=0, fontsize=8)
    ax.grid(alpha=0.2)
    title = f"{an.symbol} {tf} — {sig.action}" + (f" ({sig.status})" if sig.action != "NO TRADE" else "")
    ax.set_title(f"{title} | skor {sig.score:g}/{sig.max_score:g} (min {sig.min_score:g}) | bias HTF {sig.bias}", fontsize=11)
    fig.text(0.01, 0.005, f"asof {an.asof.strftime('%Y-%m-%d %H:%M')} UTC — alat analisis, bukan nasihat finansial; parameter = hipotesis.",
             fontsize=7, color="#757575")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True
