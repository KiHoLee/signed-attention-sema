"""Regenerates every result figure of the paper from data/ into fig/.
No experiment is rerun here. One style per scheme (STY), one legend order
(ORDER), one axes geometry (8:6 box) for every result figure, and a save
guard that refuses clipped axis labels and legends that overlap a curve.
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "fig"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.4, "grid.alpha": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 3.4,
    "figure.figsize": (3.5, 2.0625 + 0.45), "pdf.fonttype": 42,
})
# every result figure: 3.5 in canvas (the printed column width) and one axes
# box of 2.75 x 2.0625 in (8:6); figures with the legend below the axes add
# 0.68 in of canvas for the legend
CW, AX_W, AX_H = 3.5, 2.75, 2.0625
CH, CH_LEG = AX_H + 0.45, AX_H + 0.45 + 0.62
AXES_RECT = dict(left=0.64 / CW, right=(0.64 + AX_W) / CW, top=1 - 0.05 / CH, bottom=0.40 / CH)
AXES_RECT_LEG = dict(left=0.64 / CW, right=(0.64 + AX_W) / CW, top=1 - 0.05 / CH_LEG, bottom=1.02 / CH_LEG)

# label, color, marker, linestyle  (one triple per scheme, every figure)
STY = {
    "SC":     ("Proposed SC",          "#c0392b", "o", "-"),
    "SCsoft": ("softmax SC",  "#e67e22", "h", "--"),
    "SCh":    ("SC-g", "#a93226", "8", "-"),
    "SCsgn":  ("SC-s",  "#922b21", "*", "-"),
    "ATT":    ("Attention receiver",   "#d35400", "p", "--"),
    "TLC":    ("TLC",                  "#8e44ad", "P", "--"),
    "LMMSE":  ("Oracle LMMSE",       "#16a085", "v", "-."),
    "MOM":    ("Moment LMMSE",         "#0e6655", "<", ":"),
    "LMMSEe": ("empirical LMMSE", "#1abc9c", ">", "-."),
    "SR":     ("SR",                   "#2c5fa8", "s", "--"),
    "MF":     ("MF",       "#7d6608", "X", ":"),
    "OMA":    ("OMA",                  "#7f8c8d", "^", ":"),
    "OMAL":   ("OMA + LMMSE",          "#566573", "d", ":"),
    "cf_lmmse": ("LMMSE, closed form", "#16a085", "", "-"),
    "cf_sr":    ("SR, closed form",    "#2c5fa8", "", "--"),
    "cf_oma":   ("OMA, closed form",   "#7f8c8d", "", ":"),
    "cf_omal":  ("OMA + LMMSE, closed form", "#566573", "", "-."),
}
ORDER = ["SC", "SCsoft", "ATT", "TLC", "LMMSE", "MOM", "LMMSEe", "SR", "MF", "OMA", "OMAL",
         "cf_lmmse", "cf_sr", "cf_omal", "cf_oma"]


def rows(name):
    with open(DATA / name) as f:
        return [{k: (float(v) if v not in ("", None) else np.nan) for k, v in r.items()}
                for r in csv.DictReader(f)]


def col(rs, k):
    return np.array([r[k] for r in rs])


def plot(ax, x, y, key, fn=None, **kw):
    lb, c, mk, ls = STY[key]
    (fn or ax.plot)(x, y, color=c, marker=mk or None, ls=ls, label=lb, **kw)


LOCS = ["lower left", "lower right", "upper right", "upper left", "center left",
        "center right", "lower center", "upper center", "center"]


def legend(ax, **kw):
    """Ordered legend; the location given is tried first, then every other
    location and two smaller font sizes, until no curve enters the box."""
    h, l = ax.get_legend_handles_labels()
    rank = {STY[k][0]: i for i, k in enumerate(ORDER)}
    idx = sorted(range(len(l)), key=lambda i: rank.get(l[i], 99))
    d = dict(framealpha=0.95, handlelength=1.6, labelspacing=0.22, handletextpad=0.4,
             borderpad=0.3, borderaxespad=0.2, columnspacing=0.8)
    d.update(kw)
    ax._legend_spec = ([h[i] for i in idx], [l[i] for i in idx], d)
    return ax.legend(*ax._legend_spec[:2], **d)


def _overlaps(ax, box, r):
    for ln in ax.lines:
        if ln.get_label() == "_guide":
            continue
        P = ln.get_transform().transform(ln.get_xydata())
        P = P[np.isfinite(P).all(1)]
        if len(P) < 2:
            continue
        s = np.linspace(0, 1, 60)[:, None, None]
        pts = (P[:-1][None] + (P[1:] - P[:-1])[None] * s).reshape(-1, 2)
        if ((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1) & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1)).any():
            return ln.get_label()
    return None


def save(fig, name, inset=None, rect=None):
    """Enforced checks: every axis label inside the canvas; no curve point
    or segment between markers inside the legend box (or the inset box)."""
    fig.subplots_adjust(**(rect or AXES_RECT))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    F = fig.bbox
    for ax in fig.axes:
        for t in (ax.xaxis.label, ax.yaxis.label):
            b = t.get_window_extent(r)
            if b.x0 < F.x0 or b.y0 < F.y0 or b.x1 > F.x1 or b.y1 > F.y1:
                raise RuntimeError(f"{name}: clipped label {t.get_text()}")
        if inset is not None and ax is fig.axes[0]:
            hit = _overlaps(ax, inset.get_window_extent(r).expanded(1.03, 1.06), r)
            if hit:
                raise RuntimeError(f"{name}: inset overlaps {hit}")
        lg = ax.get_legend()
        if lg is None:
            continue
        spec = getattr(ax, "_legend_spec", None)
        hit = _overlaps(ax, lg.get_window_extent(r).expanded(1.03, 1.06), r)
        if hit and spec is not None:
            hh, ll, d = spec
            placed = False
            for fs in [d.get("fontsize", plt.rcParams["legend.fontsize"]), 5.8, 5.2]:
                for loc in [d.get("loc", "best")] + LOCS:
                    dd = dict(d); dd["loc"] = loc; dd["fontsize"] = fs
                    lg = ax.legend(hh, ll, **dd)
                    fig.canvas.draw()
                    if _overlaps(ax, lg.get_window_extent(r).expanded(1.03, 1.06), r) is None:
                        placed = True; break
                if placed:
                    break
            if not placed:
                raise RuntimeError(f"{name}: no clean legend placement (last hit {hit})")
            L_ = lg.get_window_extent(r)
            if L_.x0 < F.x0 or L_.y0 < F.y0 or L_.x1 > F.x1 or L_.y1 > F.y1:
                raise RuntimeError(f"{name}: legend leaves the canvas")
        elif hit:
            raise RuntimeError(f"{name}: legend overlaps {hit}")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("[OK]", name)


def floor(y, lo):
    y = np.array(y, dtype=float)
    y[y < lo] = np.nan
    return y


# ------------------------------------------------------------------ figures
def fig_theory():
    rs = rows("theory_snr.csv")
    snr = col(rs, "snr")
    fig, ax = plt.subplots()
    for k in ["cf_lmmse", "cf_sr", "cf_omal", "cf_oma"]:
        plot(ax, snr, col(rs, k), k, ax.semilogy)
    for k, mk, c in [("mc_lmmse", "v", STY["LMMSE"][1]), ("mc_sr", "s", STY["SR"][1]),
                     ("mc_omal", "d", STY["OMAL"][1]), ("mc_oma", "^", STY["OMA"][1])]:
        ax.semilogy(snr, col(rs, k), ls="none", marker=mk, color=c, markerfacecolor="none", label="_nolegend_")
    ax.semilogy([], [], ls="none", marker="o", color="k", markerfacecolor="none", label="Monte Carlo")
    ax.set_xlabel("per-user SNR (dB)"); ax.set_ylabel("per-user MSE")
    ax.set_xlim(-10, 30); ax.set_ylim(1e-3, 2e3)
    legend(ax, loc="upper right")
    save(fig, "fig_theory")


def fig_weights():
    """Beta is encoded by the shade of one color per scheme, so the proposed
    receiver keeps its line type, marker, and color of every other figure."""
    fig, ax = plt.subplots()
    ends = []
    for beta, al in [(0.3, 0.35), (0.64, 0.65), (0.9, 1.0)]:
        rs = rows(f"snr_sweep_b{beta}.csv")
        snr = col(rs, "snr")
        rho = 10 ** (snr / 10)
        l1, l2 = L.eig_sym(4, beta)
        w = rho * (l1 - l2) * (1 - l1 * l2 * rho) / (4 * (1 + l1 ** 2 * rho) * (1 + l2 ** 2 * rho))
        ax.plot(snr, w, color=STY["LMMSE"][1], ls=STY["LMMSE"][3], alpha=al, label=f"_lmmse{beta}")
        ax.plot(snr, col(rs, "SC_offdiag_mean"), color=STY["SC"][1], ls=STY["SC"][3], marker=STY["SC"][2],
                markevery=2, alpha=al, label=f"_sc{beta}")
        ax.axvline(L.snr_boundary_sym(4, beta), color="gray", lw=0.6, alpha=al, label="_guide")
        ends.append((beta, w[-1]))
    ax.axhline(0, color="k", lw=0.6, label="_guide")
    for (beta, _), xa, xt in zip(ends, [22, 17, 11], [23.5, 18, 12]):
        rs = rows(f"snr_sweep_b{beta}.csv"); snr = col(rs, "snr"); rho = 10 ** (snr / 10); l1, l2 = L.eig_sym(4, beta)
        w = rho * (l1 - l2) * (1 - l1 * l2 * rho) / (4 * (1 + l1 ** 2 * rho) * (1 + l2 ** 2 * rho))
        ya = float(np.interp(xa, snr, w))
        ax.annotate(f"$\\beta={beta}$", xy=(xa, ya), xytext=(xt, 0.5), ha="center", va="center", fontsize=7,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="gray", shrinkB=2))
    hb = [Line2D([], [], color=STY["SC"][1], ls=STY["SC"][3], marker=STY["SC"][2], label="Proposed SC"),
          Line2D([], [], color=STY["LMMSE"][1], ls=STY["LMMSE"][3], label="Oracle LMMSE")]
    ax.set_xlabel("per-user SNR (dB)"); ax.set_ylabel("weight $[\\mathbf{W}]_{uv}$")
    ax.set_xlim(-6, 24); ax.set_ylim(-2.3, 0.7)
    d = dict(framealpha=0.95, handlelength=1.6, labelspacing=0.22, handletextpad=0.4, borderpad=0.3,
             borderaxespad=0.2, loc="lower left")
    ax.legend(hb, [h.get_label() for h in hb], **d); ax._legend_spec = (hb, [h.get_label() for h in hb], d)
    save(fig, "fig_weights")


def fig_snr(metric="ser"):
    rs = rows("snr_sweep_b0.64.csv")
    snr = col(rs, "snr")
    fig, ax = plt.subplots()
    keys = ["SC", "SCsoft", "ATT", "TLC", "LMMSE", "MOM", "SR", "MF", "OMA", "OMAL"]
    for k in keys:
        if f"{metric}_{k}" not in rs[0]:
            continue
        y = col(rs, f"{metric}_{k}") if metric == "ser" else 1 - col(rs, f"cos_{k}")
        plot(ax, snr, floor(y, 1e-6) if metric == "ser" else y, k, ax.semilogy)
    ax.set_xlabel("per-user SNR (dB)")
    ax.set_ylabel("SER" if metric == "ser" else "cosine distortion $1-\\cos$")
    ax.set_xlim(-6, 24)
    ax.set_ylim(1e-6, 1.5) if metric == "ser" else ax.set_ylim(5e-3, 1)
    fig.set_size_inches(CW, CH_LEG); fig.subplots_adjust(**AXES_RECT_LEG)
    _shared_legend(fig, [ax], f"fig_snr_{metric}", 3, fs=7)


def fig_beta(snr=0, metric="ser"):
    rs = rows(f"beta_sweep_{snr}dB.csv")
    beta = col(rs, "beta")
    fig, ax = plt.subplots()
    keys = ["SC", "SCsoft", "ATT", "TLC", "LMMSE", "MOM", "SR", "MF", "OMA", "OMAL"]
    for k in keys:
        if f"{metric}_{k}" not in rs[0]:
            continue
        y = col(rs, f"{metric}_{k}") if metric == "ser" else 1 - col(rs, f"cos_{k}")
        plot(ax, beta, floor(y, 1e-6) if metric == "ser" else y, k, ax.semilogy, markevery=2)
    ax.set_xlabel("similarity $\\beta$")
    ax.set_ylabel("SER" if metric == "ser" else "cosine distortion $1-\\cos$")
    ax.set_xlim(0, 0.9)
    ax.set_ylim(1e-6, 1.5) if metric == "ser" else ax.set_ylim(2e-2, 1)
    fig.set_size_inches(CW, CH_LEG); fig.subplots_adjust(**AXES_RECT_LEG)
    _shared_legend(fig, [ax], f"fig_beta_{snr}dB_{metric}", 3, fs=7)


def fig_scaling_U():
    rs = rows("scaling_U.csv")
    fig, ax = plt.subplots()
    for snr, mk in [(0.0, "-"), (10.0, "--")]:
        sub = [r for r in rs if r["snr"] == snr]
        Us = col(sub, "U")
        for k in ["SC", "LMMSE", "MOM", "SR", "OMA"]:
            lb, c, m, _ = STY[k]
            ax.semilogy(Us, floor(col(sub, f"ser_{k}"), 1e-6), color=c, marker=m, ls=mk,
                        label=lb if snr == 0 else "_nolegend_")
    ax.plot([], [], color="k", ls="-", label="0 dB"); ax.plot([], [], color="k", ls="--", label="10 dB")
    ax.set_xlabel("number of users $U$"); ax.set_ylabel("SER")
    ax.set_xlim(2, 8); ax.set_ylim(1e-6, 1.5)
    legend(ax, loc="lower right", ncol=2)
    save(fig, "fig_scaling_U")


def fig_sigdesign():
    so = rows("sigopt.csv")
    snr = col(so, "snr")
    fig, ax = plt.subplots()
    ax.semilogy(snr, col(so, "mse_matched"), color=STY["LMMSE"][1], ls="-", label="matched, $\\beta_{\\rm s}=\\beta$")
    ax.semilogy(snr, col(so, "mse_opt"), color="k", ls="-", label="optimal $\\beta_{\\rm s}^\\star$")
    ax.semilogy(snr, col(so, "mse_orth"), color=STY["OMAL"][1], ls="-.", label="OMA + LMMSE")
    ax.semilogy(snr, col(so, "mse_oma_wiener"), color=STY["OMA"][1], ls=":", label="OMA")
    ax2 = ax.twinx()
    ax2.plot(snr, col(so, "beta_s_opt"), color="k", ls="--", lw=0.9, label="$\\beta_{\\rm s}^\\star$, right axis")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("$\\beta_{\\rm s}^\\star$"); ax2.grid(False)
    ax.set_xlabel("per-user SNR (dB)"); ax.set_ylabel("per-user MSE")
    ax.set_xlim(-10, 30); ax.set_ylim(1e-3, 1e3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    d = dict(framealpha=0.95, handlelength=1.6, labelspacing=0.22, loc="upper right")
    ax.legend(h1 + h2, l1 + l2, **d); ax._legend_spec = (h1 + h2, l1 + l2, d)
    rect = dict(AXES_RECT); rect["right"] = 1 - 0.5 / CW
    save(fig, "fig_sigdesign", rect=rect)


def fig_real(metric="ser"):
    rs = rows("real_ag_news_bert-base-uncased_snr.csv")
    snr = col(rs, "snr")
    fig, ax = plt.subplots(figsize=(CW, CH_LEG))
    fig.subplots_adjust(**AXES_RECT_LEG)
    keys = ["SC", "SCsoft", "ATT", "TLC", "LMMSEe", "LMMSE", "MOM", "SR", "MF", "OMA", "OMAL"]
    for k in keys:
        if f"{metric}_{k}" not in rs[0]:
            continue
        y = col(rs, f"{metric}_{k}")
        if metric == "ser":
            plot(ax, snr, floor(y, 1e-6), k, ax.semilogy)
        else:
            plot(ax, snr, y, k)
    ax.set_xlabel("per-user SNR (dB)")
    if metric == "ser":
        ax.set_ylabel("SER"); ax.set_ylim(1e-6, 1.5)
    else:
        ax.set_ylabel("topic accuracy"); ax.set_ylim(0.2, 1.0)
    ax.set_xlim(-4, 20)
    # eleven schemes do not fit inside a column-width axes: shared legend below the axes
    _shared_legend(fig, [ax], "fig_real" if metric == "ser" else f"fig_real_{metric}", 3, fs=7)


ALL = {"theory": fig_theory, "weights": fig_weights,
       "snr_ser": lambda: fig_snr("ser"), "snr_cos": lambda: fig_snr("cos"),
       "beta0": lambda: fig_beta(0, "ser"), "beta10": lambda: fig_beta(10, "cos"),
       "scaling": fig_scaling_U, "sigdesign": fig_sigdesign,
       "real_ser": lambda: fig_real("ser"), "real_acc": lambda: fig_real("acc")}

# ------------------------------------------------------ multi-panel figures
PANEL_W, PANEL_H = 2.4, 1.5


def _panel_axes(n):
    W = n * PANEL_W + 0.3
    fig, axs = plt.subplots(1, n, figsize=(W, PANEL_H + 0.75))
    fig.subplots_adjust(left=0.62 / W, right=1 - 0.1 / W, top=0.96, bottom=0.85 / (PANEL_H + 0.75), wspace=0.55)
    return fig, (list(axs) if n > 1 else [axs])


def _shared_legend(fig, axs, name, ncol, fs=6.4):
    """One legend below the panels in ORDER, outside every axes box."""
    hs, ls = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in ls and not ll.startswith("_"):
                hs.append(hh); ls.append(ll)
    rank = {STY[k][0]: i for i, k in enumerate(ORDER)}
    idx = sorted(range(len(ls)), key=lambda i: rank.get(ls[i], 99))
    fig.legend([hs[i] for i in idx], [ls[i] for i in idx], loc="lower center", ncol=ncol,
               frameon=True, framealpha=0.95, handlelength=1.6, labelspacing=0.25,
               handletextpad=0.4, columnspacing=0.8, borderaxespad=0.1, fontsize=fs)
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); F = fig.bbox
    for ax in axs:
        for t in (ax.xaxis.label, ax.yaxis.label):
            b = t.get_window_extent(r)
            if b.x0 < F.x0 or b.y0 < F.y0 or b.x1 > F.x1 or b.y1 > F.y1:
                raise RuntimeError(f"{name}: clipped label {t.get_text()}")
        for lg in fig.legends:
            Lb = lg.get_window_extent(r)
            if Lb.x0 < F.x0 or Lb.y0 < F.y0 or Lb.x1 > F.x1 or Lb.y1 > F.y1:
                raise RuntimeError(f"{name}: legend leaves the canvas")
            if Lb.overlaps(ax.get_window_extent(r)):
                raise RuntimeError(f"{name}: legend overlaps an axes box")
            if Lb.overlaps(ax.xaxis.label.get_window_extent(r)):
                raise RuntimeError(f"{name}: legend overlaps the x label")
    fig.savefig(FIG / f"{name}.pdf"); plt.close(fig); print("[OK]", name)


def fig_snr_combo():
    rs = rows("snr_sweep_b0.64.csv"); snr = col(rs, "snr")
    rb = rows("beta_sweep_0dB.csv"); beta = col(rb, "beta")
    keys = ["SC", "SCsoft", "ATT", "TLC", "LMMSE", "MOM", "SR", "MF", "OMA", "OMAL"]
    fig, axs = _panel_axes(3)
    for k in keys:
        plot(axs[0], snr, floor(col(rs, f"ser_{k}"), 1e-6), k, axs[0].semilogy, markevery=2)
        plot(axs[1], snr, 1 - col(rs, f"cos_{k}"), k, axs[1].semilogy, markevery=2)
        plot(axs[2], beta, floor(col(rb, f"ser_{k}"), 1e-6), k, axs[2].semilogy, markevery=3)
    axs[0].set_xlabel("per-user SNR (dB)\n(a)"); axs[0].set_ylabel("SER"); axs[0].set_xlim(-6, 24); axs[0].set_ylim(1e-6, 1.5)
    axs[1].set_xlabel("per-user SNR (dB)\n(b)"); axs[1].set_ylabel("cosine distortion $1-\\cos$"); axs[1].set_xlim(-6, 24); axs[1].set_ylim(5e-3, 1)
    axs[2].set_xlabel("similarity $\\beta$\n(c)"); axs[2].set_ylabel("SER"); axs[2].set_xlim(0, 0.9); axs[2].set_ylim(1e-6, 1.5)
    _shared_legend(fig, axs, "fig_synth", 5)


def fig_real_combo():
    rs = rows("real_ag_news_bert-base-uncased_snr.csv"); snr = col(rs, "snr")
    keys = ["SC", "SCsoft", "ATT", "TLC", "LMMSEe", "LMMSE", "MOM", "SR", "MF", "OMA", "OMAL"]
    fig, axs = _panel_axes(2)
    for k in keys:
        plot(axs[0], snr, floor(col(rs, f"ser_{k}"), 1e-6), k, axs[0].semilogy)
        plot(axs[1], snr, col(rs, f"acc_{k}"), k)
    axs[0].set_xlabel("per-user SNR (dB)\n(a)"); axs[0].set_ylabel("SER"); axs[0].set_xlim(-4, 20); axs[0].set_ylim(1e-6, 1.5)
    axs[1].set_xlabel("per-user SNR (dB)\n(b)"); axs[1].set_ylabel("topic accuracy"); axs[1].set_xlim(-4, 20); axs[1].set_ylim(0.4, 0.9)
    _shared_legend(fig, axs, "fig_real", 4)


def fig_scaling_combo():
    rs = rows("scaling_U.csv")
    fig, axs = _panel_axes(2)
    for i, snr in enumerate([0.0, 10.0]):
        sub = [r for r in rs if r["snr"] == snr]
        Us = col(sub, "U")
        for k in ["SC", "TLC", "LMMSE", "MOM", "SR", "MF", "OMA", "OMAL"]:
            plot(axs[i], Us, 1 - col(sub, f"cos_{k}"), k)
        axs[i].set_xlabel(f"number of users $U$\n({'ab'[i]}) {snr:.0f} dB"); axs[i].set_ylabel("distortion $1-\\cos$")
        axs[i].set_xlim(2, 8)
    axs[0].set_ylim(0.1, 0.6); axs[1].set_ylim(0.05, 0.25)
    _shared_legend(fig, axs, "fig_scaling", 4)


ALL.update({"synth": fig_snr_combo, "real": fig_real_combo, "scaling2": fig_scaling_combo})


def fig_analysis_combo():
    """(a) closed form vs Monte Carlo MSE, (b) off-diagonal weights, (c) signature design."""
    fig, axs = _panel_axes(3)
    # (a)
    rs = rows("theory_snr.csv"); snr = col(rs, "snr"); ax = axs[0]
    for k in ["cf_lmmse", "cf_sr", "cf_omal", "cf_oma"]:
        plot(ax, snr, col(rs, k), k, ax.semilogy)
    for k, mk, c in [("mc_lmmse", "v", STY["LMMSE"][1]), ("mc_sr", "s", STY["SR"][1]),
                     ("mc_omal", "d", STY["OMAL"][1]), ("mc_oma", "^", STY["OMA"][1])]:
        ax.semilogy(snr[::2], col(rs, k)[::2], ls="none", marker=mk, color=c, markerfacecolor="none", label="_nolegend_")
    ax.semilogy([], [], ls="none", marker="o", color="k", markerfacecolor="none", label="Monte Carlo")
    ax.set_xlabel("per-user SNR (dB)\n(a)"); ax.set_ylabel("per-user MSE"); ax.set_xlim(-10, 30); ax.set_ylim(1e-3, 2e3)
    # (b)
    ax = axs[1]
    for beta, ls in [(0.3, ":"), (0.64, "-"), (0.9, "--")]:
        rb = rows(f"snr_sweep_b{beta}.csv"); sb = col(rb, "snr"); rho = 10 ** (sb / 10)
        l1, l2 = L.eig_sym(4, beta)
        w = rho * (l1 - l2) * (1 - l1 * l2 * rho) / (4 * (1 + l1 ** 2 * rho) * (1 + l2 ** 2 * rho))
        ax.plot(sb, w, color=STY["LMMSE"][1], ls=ls, label=f"_lmmse{beta}")
        ax.plot(sb, col(rb, "SC_offdiag_mean"), color=STY["SC"][1], ls=ls, marker="o", markevery=2, label=f"_sc{beta}")
        ax.axvline(L.snr_boundary_sym(4, beta), color="gray", lw=0.6, ls=ls, label="_guide")
    ax.axhline(0, color="k", lw=0.6, label="_guide")
    # compact legend: scheme by color, beta by line style (two short columns fit inside the axes)
    hb = [Line2D([], [], color=STY["SC"][1], ls="-", marker="o", label="Proposed SC"),
          Line2D([], [], color=STY["LMMSE"][1], ls="-", label="Oracle LMMSE"),
          Line2D([], [], color="k", ls=":", label="$\\beta=0.3$"),
          Line2D([], [], color="k", ls="-", label="$\\beta=0.64$"),
          Line2D([], [], color="k", ls="--", label="$\\beta=0.9$")]
    ax.set_xlabel("per-user SNR (dB)\n(b)"); ax.set_ylabel("weight $[\\mathbf{W}]_{uv}$"); ax.set_xlim(-6, 24); ax.set_ylim(-2.9, 0.7)
    # (c)
    ax = axs[2]; so = rows("sigopt.csv"); s2 = col(so, "snr")
    ax.semilogy(s2, col(so, "mse_matched"), color=STY["LMMSE"][1], ls="-", label="matched, $\\beta_{\\rm s}=\\beta$")
    ax.semilogy(s2, col(so, "mse_opt"), color="k", ls="-", label="optimal $\\beta_{\\rm s}^\\star$")
    ax.semilogy(s2, col(so, "mse_orth"), color=STY["OMAL"][1], ls="-.", label="OMA + LMMSE")
    ax.semilogy(s2, col(so, "mse_oma_wiener"), color=STY["OMA"][1], ls=":", label="OMA")
    ax2 = ax.twinx(); ax2.plot(s2, col(so, "beta_s_opt"), color="k", ls="--", lw=0.9, label="$\\beta_{\\rm s}^\\star$, right axis")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("$\\beta_{\\rm s}^\\star$"); ax2.grid(False)
    ax.set_xlabel("per-user SNR (dB)\n(c)"); ax.set_ylabel("per-user MSE"); ax.set_xlim(-10, 30); ax.set_ylim(1e-3, 30)
    # legends: (a) and (c) in the empty upper-right corner, (b) lower left; each checked by the guard
    legend(axs[0], loc="upper right", fontsize=5.6)
    db = dict(fontsize=5.4, framealpha=0.95, handlelength=1.5, labelspacing=0.22, handletextpad=0.4,
              borderpad=0.3, borderaxespad=0.2, columnspacing=0.7, ncol=2, loc="lower left")
    axs[1].legend(hb, [h.get_label() for h in hb], **db)
    axs[1]._legend_spec = (hb, [h.get_label() for h in hb], db)
    h1, l1_ = ax.get_legend_handles_labels(); h2, l2_ = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1_ + l2_, loc="upper right", fontsize=5.6, framealpha=0.95, handlelength=1.6, labelspacing=0.22)
    ax._legend_spec = (h1 + h2, l1_ + l2_, dict(fontsize=5.6, framealpha=0.95, handlelength=1.6, labelspacing=0.22, loc="upper right"))
    fig.subplots_adjust(left=0.62 / (3 * PANEL_W + 0.3), right=1 - 0.75 / (3 * PANEL_W + 0.3), top=0.96, bottom=0.6 / (PANEL_H + 0.75), wspace=0.62)
    name = "fig_analysis"
    fig.canvas.draw(); r = fig.canvas.get_renderer(); F = fig.bbox
    for a_ in axs + [ax2]:
        for t in (a_.xaxis.label, a_.yaxis.label):
            b = t.get_window_extent(r)
            if b.x0 < F.x0 or b.y0 < F.y0 or b.x1 > F.x1 or b.y1 > F.y1:
                raise RuntimeError(f"{name}: clipped label {t.get_text()}")
    for a_ in axs:
        lg = a_.get_legend(); spec = getattr(a_, "_legend_spec", None)
        hit = _overlaps(a_, lg.get_window_extent(r).expanded(1.03, 1.06), r)
        if hit is None and not a_.bbox.contains(*lg.get_window_extent(r).p0) or hit is None and not a_.bbox.contains(*lg.get_window_extent(r).p1):
            hit = "legend outside axes"
        if hit:
            hh, ll, d = spec; placed = False
            for fs in [5.6, 5.2, 4.8]:
                for loc in LOCS:
                    dd = dict(d); dd["loc"] = loc; dd["fontsize"] = fs
                    lg = a_.legend(hh, ll, **dd); fig.canvas.draw()
                    bb = lg.get_window_extent(r)
                    if _overlaps(a_, bb.expanded(1.03, 1.06), r) is None and a_.bbox.contains(*bb.p0) and a_.bbox.contains(*bb.p1):
                        placed = True; break
                if placed: break
            if not placed:
                raise RuntimeError(f"{name}: no clean legend placement (last hit {hit})")
    fig.savefig(FIG / f"{name}.pdf"); plt.close(fig); print("[OK]", name)


ALL["analysis"] = fig_analysis_combo


def fig_real_stacked():
    """Real-data panels stacked in one column: (a) SER, (b) topic accuracy."""
    rs = rows("real_ag_news_bert-base-uncased_snr.csv"); snr = col(rs, "snr")
    keys = ["SC", "SCsoft", "ATT", "TLC", "LMMSEe", "LMMSE", "MOM", "SR", "MF", "OMA", "OMAL"]
    W, H = 3.4, 2 * 1.7 + 1.75
    fig, axs = plt.subplots(2, 1, figsize=(W, H))
    fig.subplots_adjust(left=0.62 / W, right=1 - 0.1 / W, top=0.98, bottom=1.6 / H, hspace=0.42)
    axs = list(axs)
    for k in keys:
        plot(axs[0], snr, floor(col(rs, f"ser_{k}"), 1e-6), k, axs[0].semilogy)
        plot(axs[1], snr, col(rs, f"acc_{k}"), k)
    axs[0].set_xlabel("per-user SNR (dB)\n(a)"); axs[0].set_ylabel("SER"); axs[0].set_xlim(-4, 20); axs[0].set_ylim(1e-6, 1.5)
    axs[1].set_xlabel("per-user SNR (dB)\n(b)"); axs[1].set_ylabel("topic accuracy"); axs[1].set_xlim(-4, 20); axs[1].set_ylim(0.4, 0.9)
    _shared_legend(fig, axs, "fig_real", 2)


ALL["real"] = lambda: fig_real("ser")


if __name__ == "__main__":
    PAPER = ["theory", "weights", "sigdesign", "snr_ser", "snr_cos", "beta0", "real_ser"]
    names = sys.argv[1:] or PAPER
    failed = []
    for n in names:
        try:
            ALL[n]()
        except FileNotFoundError as e:
            print("[skip]", n, e)
        except Exception as e:
            print("[FAIL]", n, e); failed.append(n)
    if failed:
        sys.exit("failed: " + ", ".join(failed))






