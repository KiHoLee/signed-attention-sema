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
    "figure.figsize": (2.975, 1.753125 + 0.45), "pdf.fonttype": 42,
})
# every result figure: 3.5 in canvas (the printed column width) and one axes
# box of 2.75 x 2.0625 in (8:6); figures with the legend below the axes add
# 0.68 in of canvas for the legend
CW, AX_W, AX_H = 2.975, 2.3375, 1.753125   # printed at 0.85 column width (standard floor), 8:6 box
CH, CH_LEG = AX_H + 0.45, AX_H + 0.45 + 0.62
AX_L = 0.56   # left margin in inches; leaves 0.08 in on the right so the last x tick label is not cut
AXES_RECT = dict(left=AX_L / CW, right=(AX_L + AX_W) / CW, top=1 - 0.05 / CH, bottom=0.40 / CH)
AXES_RECT_LEG = dict(left=AX_L / CW, right=(AX_L + AX_W) / CW, top=1 - 0.05 / CH_LEG, bottom=1.02 / CH_LEG)

# label, color, marker, linestyle  (one triple per scheme, every figure)
STY = {
    "SC":     ("Proposed SC",          "#c0392b", "o", "-"),
    "SCsoft": ("Softmax SC",  "#e67e22", "h", "--"),
    "SCh":    ("SC-g", "#a93226", "8", "-"),
    "SCsgn":  ("SC-s",  "#922b21", "*", "-"),
    "SCrho":  ("SC-$\\rho$",   "#e74c3c", "D", "--"),
    "TLCg":   ("TLC-g",             "#6c3483", "x", "-."),
    "ATT":    ("Attention receiver",   "#d35400", "p", "--"),
    "TLC":    ("TLC",                  "#8e44ad", "P", "--"),
    "LMMSE":  ("Genie-aided LMMSE",       "#16a085", "v", "-."),
    "PLG":    ("Plug-in LMMSE",        "#1b4f72", "*", "-"),
    "MOM":    ("Moment LMMSE",         "#0e6655", "<", ":"),
    "LMMSEe": ("Empirical LMMSE", "#1abc9c", ">", "-."),
    "SR":     ("SR",                   "#2c5fa8", "s", "--"),
    "MF":     ("MF",       "#7d6608", "X", ":"),
    "OMA":    ("OMA",                  "#7f8c8d", "^", ":"),
    "OMAL":   ("OMA + LMMSE",          "#566573", "d", ":"),
    "cf_lmmse": ("LMMSE, closed form", "#16a085", "", "-"),
    "cf_sr":    ("SR, closed form",    "#2c5fa8", "", "--"),
    "cf_oma":   ("OMA, closed form",   "#7f8c8d", "", ":"),
    "cf_omal":  ("OMA + LMMSE, closed form", "#566573", "", "-."),
}
ORDER = ["SC", "SCrho", "SCsoft", "ATT", "TLC", "TLCg", "PLG", "LMMSE", "LMMSEe", "MOM", "SR", "MF", "OMA", "OMAL",
         "cf_lmmse", "cf_sr", "cf_oma", "cf_omal"]


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
        xl, yl = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        shown = [t for t in ax.get_xticklabels() if xl[0] <= t.get_position()[0] <= xl[1]] + \
                [t for t in ax.get_yticklabels() if yl[0] <= t.get_position()[1] <= yl[1]]
        for t in [ax.xaxis.label, ax.yaxis.label] + shown:
            if not t.get_text():
                continue
            b = t.get_window_extent(r)
            if b.x0 < F.x0 - 0.5 or b.y0 < F.y0 - 0.5 or b.x1 > F.x1 + 0.5 or b.y1 > F.y1 + 0.5:
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
    # closed form as a line, Monte Carlo as the scheme's own hollow marker;
    # one legend entry per scheme shows both, so every marker is explained
    hb = []
    for cf, mc, k in [("cf_lmmse", "mc_lmmse", "LMMSE"), ("cf_sr", "mc_sr", "SR"),
                      ("cf_oma", "mc_oma", "OMA"), ("cf_omal", "mc_omal", "OMAL")]:
        lb, c, mk, _ = STY[k]
        ls = STY[cf][3]
        ax.semilogy(snr, col(rs, cf), color=c, ls=ls, label="_nolegend_")
        ax.semilogy(snr, col(rs, mc), ls="none", marker=mk, color=c, markerfacecolor="none", label="_nolegend_")
        hb.append(Line2D([], [], color=c, ls=ls, marker=mk, markerfacecolor="none", label=lb))
    ax.set_xlabel("per-user SNR (dB)"); ax.set_ylabel("per-user MSE")
    ax.set_xlim(-10, 30); ax.set_ylim(1e-3, 2e3)
    d = dict(framealpha=0.95, handlelength=2.2, labelspacing=0.22, handletextpad=0.4, borderpad=0.3,
             borderaxespad=0.2, loc="upper right")
    ax.legend(hb, [h.get_label() for h in hb], **d); ax._legend_spec = (hb, [h.get_label() for h in hb], d)
    save(fig, "fig_theory")


def fig_weights():
    """Off-diagonal weight relative to the diagonal weight, [W]_uv/[W]_uu, for
    the genie-aided LMMSE receiver (closed form, unit gains) and the proposed
    receiver (mean off-diagonal over mean diagonal, Rayleigh gains). The ratio
    removes the row scale that nrm() leaves undetermined in the proposed
    receiver. Beta is encoded by the shade of one color per scheme, so the
    proposed receiver keeps its line type, marker, and color of every figure."""
    fig, ax = plt.subplots()
    def lmmse_ratio(snr, beta):
        rho = 10 ** (snr / 10); l1, l2 = L.eig_sym(4, beta)
        g = lambda lam: lam * rho / (1 + lam ** 2 * rho)
        return (g(l1) - g(l2)) / (g(l1) + 3 * g(l2))
    curves = {}
    for beta, al in [(0.3, 0.35), (0.64, 0.65), (0.9, 1.0)]:
        rs = rows(f"snr_sweep_b{beta}.csv")
        snr = col(rs, "snr")
        wl = lmmse_ratio(snr, beta)
        ws = col(rs, "SC_offdiag_mean") / col(rs, "SC_diag_mean")
        curves[beta] = (snr, wl, ws)
        ax.plot(snr, wl, color=STY["LMMSE"][1], ls=STY["LMMSE"][3], alpha=al, label=f"_lmmse{beta}")
        ax.plot(snr, ws, color=STY["SC"][1], ls=STY["SC"][3], marker=STY["SC"][2], markevery=2, alpha=al, label=f"_sc{beta}")
        ax.axvline(L.snr_boundary_sym(4, beta), color="gray", lw=0.6, alpha=al, label="_guide")
    ax.axhline(0, color="k", lw=0.6, label="_guide")
    # one label per similarity, with a leader to each of its two curves
    for beta, xa, xt in zip([0.3, 0.64, 0.9], [12, 17, 22], [11, 16, 21]):
        snr, wl, ws = curves[beta]
        for y in (float(np.interp(xa, snr, wl)), float(np.interp(xa, snr, ws))):
            ax.plot(xa, y, marker="o", ms=5.5, mfc="none", mec="k", mew=0.8, ls="none", label="_guide", zorder=5)
            ax.annotate(f"$\\beta={beta}$", xy=(xa, y), xytext=(xt, 0.42), ha="center", va="center", fontsize=7,
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="0.35", shrinkB=3))
    hb = [Line2D([], [], color=STY["SC"][1], ls=STY["SC"][3], marker=STY["SC"][2], label="Proposed SC"),
          Line2D([], [], color=STY["LMMSE"][1], ls=STY["LMMSE"][3], label="Genie-aided LMMSE")]
    ax.set_xlabel("per-user SNR (dB)"); ax.set_ylabel("relative off-diagonal weight")
    ax.set_xlim(-6, 24); ax.set_ylim(-0.42, 0.62)
    d = dict(framealpha=0.95, handlelength=1.6, labelspacing=0.22, handletextpad=0.4, borderpad=0.3,
             borderaxespad=0.2, loc="lower left")
    ax.legend(hb, [h.get_label() for h in hb], **d); ax._legend_spec = (hb, [h.get_label() for h in hb], d)
    save(fig, "fig_weights")

def _overlay(ax, metric, xkey, path, pairs):
    """Adds schemes computed by a separate script (same model, same grid) to an axis."""
    if not path.exists():
        return
    rv = rows(path); xv = col(rv, xkey)
    for k, src in pairs:
        if f"{metric}_{src}" not in rv[0] and f"cos_{src}" not in rv[0]:
            continue
        y = col(rv, f"{metric}_{src}") if metric == "ser" else 1 - col(rv, f"cos_{src}")
        plot(ax, xv, floor(y, 1e-6) if metric == "ser" else y, k, ax.semilogy)


def fig_snr(metric="ser"):
    rs = rows("snr_sweep_b0.64.csv")
    snr = col(rs, "snr")
    fig, ax = plt.subplots()
    paired = DATA / "plugin" / "snr_b0.64.csv"   # SC, genie, SR, MF, plug-in on identical blocks (code/exp_plugin.py)
    same = ["SC", "LMMSE", "SR", "MF", "PLG"] if paired.exists() else []
    keys = ["SC", "SCsoft", "ATT", "TLC", "LMMSE", "SR", "MF", "OMA", "OMAL"]
    for k in keys:
        if k in same or f"{metric}_{k}" not in rs[0]:
            continue
        y = col(rs, f"{metric}_{k}") if metric == "ser" else 1 - col(rs, f"cos_{k}")
        plot(ax, snr, floor(y, 1e-6) if metric == "ser" else y, k, ax.semilogy)
    _overlay(ax, metric, "snr", DATA / "variants_floor" / "snr_b0.64.csv", [("TLCg", "TLCg")])      # code/exp_floor_variants.py
    _overlay(ax, metric, "snr", paired, [(k, k) for k in same])
    ax.set_xlabel("per-user SNR (dB)")
    ax.set_ylabel("SER" if metric == "ser" else "cosine distortion $1-\\cos$")
    ax.set_xlim(-6, 24)
    ax.set_ylim(1e-6, 1.5) if metric == "ser" else ax.set_ylim(5e-3, 1.3)
    fig.set_size_inches(CW, CH_LEG); fig.subplots_adjust(**AXES_RECT_LEG)
    _shared_legend(fig, [ax], f"fig_snr_{metric}", 3, fs=7)


def fig_beta(snr=0, metric="ser"):
    rs = rows(f"beta_sweep_{snr}dB.csv")
    beta = col(rs, "beta")
    fig, ax = plt.subplots()
    paired = DATA / "plugin" / "beta_0dB.csv"
    same = ["SC", "LMMSE", "SR", "MF", "PLG"] if (paired.exists() and snr == 0) else []
    keys = ["SC", "SCsoft", "ATT", "TLC", "LMMSE", "SR", "MF", "OMA", "OMAL"]
    for k in keys:
        if k in same or f"{metric}_{k}" not in rs[0]:
            continue
        y = col(rs, f"{metric}_{k}") if metric == "ser" else 1 - col(rs, f"cos_{k}")
        plot(ax, beta, floor(y, 1e-6) if metric == "ser" else y, k, ax.semilogy, markevery=2)
    if snr == 0:
        _overlay(ax, metric, "beta", DATA / "variants_floor" / "beta_0dB.csv", [("TLCg", "TLCg")])
        _overlay(ax, metric, "beta", paired, [(k, k) for k in same])
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
    """One axis: LMMSE error of the four signature designs; the optimal
    correlation beta_s* is written along its own curve at a few SNRs."""
    so = rows("sigopt.csv")
    snr = col(so, "snr"); bopt = col(so, "beta_s_opt"); mopt = col(so, "mse_opt")
    fig, ax = plt.subplots()
    ax.semilogy(snr, col(so, "mse_matched"), color=STY["LMMSE"][1], ls="-", label="matched, $\\beta_{\\rm s}=\\beta$")
    ax.semilogy(snr, mopt, color="k", ls="-", label="optimal $\\beta_{\\rm s}^\\star$")
    ax.semilogy(snr, col(so, "mse_oma_wiener"), color=STY["OMA"][1], ls=":", label="OMA")
    ax.semilogy(snr, col(so, "mse_orth"), color=STY["OMAL"][1], ls="-.", label="OMA + LMMSE")
    # the optimal correlation at five SNRs: a dot on the optimal curve, a dark leader, and the value
    for x, ha, dx, dy in [(-6, "left", 2, -16), (0, "center", 0, -36), (5, "left", 7, -16), (10, "center", 0, -36), (20, "center", 0, -16)]:
        k = int(np.argmin(np.abs(snr - x)))
        ax.plot(snr[k], mopt[k], marker="o", ms=5.5, mfc="none", mec="k", mew=0.8, ls="none", label="_guide", zorder=5)
        ax.annotate(f"$\\beta_{{\\rm s}}^\\star={bopt[k]:.2f}$", xy=(snr[k], mopt[k]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="top", fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.7, color="0.35", shrinkB=2))
    ax.set_xlabel("per-user SNR (dB)"); ax.set_ylabel("per-user MSE")
    ax.set_xlim(-10, 30); ax.set_ylim(1e-3, 30)
    h, l = ax.get_legend_handles_labels()   # plot order: matched, optimal, OMA + LMMSE, OMA
    d = dict(framealpha=0.95, handlelength=1.6, labelspacing=0.22, handletextpad=0.4, borderpad=0.3, borderaxespad=0.2, loc="upper right")
    ax.legend(h, l, **d); ax._legend_spec = (h, l, d)
    save(fig, "fig_sigdesign")


def fig_real(metric="ser"):
    rs = rows("real_ag_news_bert-base-uncased_snr.csv")
    snr = col(rs, "snr")
    H5 = CH_LEG + 0.16                       # thirteen schemes: five legend rows
    fig, ax = plt.subplots(figsize=(CW, H5))
    rect = dict(AXES_RECT_LEG); rect["top"] = 1 - 0.05 / H5; rect["bottom"] = 1.18 / H5
    fig.subplots_adjust(**rect)
    keys = ["SC", "SCsoft", "ATT", "TLC", "TLCg", "LMMSEe", "LMMSE", "SR", "MF", "OMA", "OMAL"]
    for k in keys:
        if f"{metric}_{k}" not in rs[0]:
            continue
        y = col(rs, f"{metric}_{k}")
        if metric == "ser":
            plot(ax, snr, floor(y, 1e-6), k, ax.semilogy)
        else:
            plot(ax, snr, y, k)
    pf = DATA / "plugin" / "real_ag_news_bert-base-uncased_snr.csv"
    if pf.exists():
        rp = rows(pf); sp = col(rp, "snr"); y = col(rp, f"{metric}_PLG")
        plot(ax, sp, floor(y, 1e-6) if metric == "ser" else y, "PLG", ax.semilogy if metric == "ser" else None)
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
    # centered under the axes box (not the canvas), so the legend aligns with the plot;
    # spacing and font shrink in steps until the centered box fits inside the canvas
    x0 = min(a.get_position().x0 for a in axs); x1 = max(a.get_position().x1 for a in axs)
    steps = [(fs, 1.6, 0.8, 0.4), (fs, 1.3, 0.6, 0.3), (fs - 0.5, 1.2, 0.5, 0.3), (fs - 1.0, 1.1, 0.4, 0.25)]
    # first centered under the axes box; if no spacing fits, centered on the canvas
    placed = False
    for xc in [(x0 + x1) / 2, 0.5]:
        for fs_, hl, cs_, htp in steps:
            for lg_ in list(fig.legends):
                lg_.remove()
            fig.legend([hs[i] for i in idx], [ls[i] for i in idx], loc="lower center", ncol=ncol,
                       bbox_to_anchor=(xc, 0.0),
                       frameon=True, framealpha=0.95, handlelength=hl, labelspacing=0.25,
                       handletextpad=htp, columnspacing=cs_, borderaxespad=0.1, fontsize=fs_)
            fig.canvas.draw(); r = fig.canvas.get_renderer(); F = fig.bbox
            Lb = fig.legends[0].get_window_extent(r)
            if Lb.x0 >= F.x0 and Lb.x1 <= F.x1:
                placed = True; break
        if placed:
            break
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); F = fig.bbox
    for ax in axs:
        xl, yl = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        shown = [t for t in ax.get_xticklabels() if xl[0] <= t.get_position()[0] <= xl[1]] + \
                [t for t in ax.get_yticklabels() if yl[0] <= t.get_position()[1] <= yl[1]]
        for t in [ax.xaxis.label, ax.yaxis.label] + shown:
            if not t.get_text():
                continue
            b = t.get_window_extent(r)
            if b.x0 < F.x0 - 0.5 or b.y0 < F.y0 - 0.5 or b.x1 > F.x1 + 0.5 or b.y1 > F.y1 + 0.5:
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
    for k in ["cf_lmmse", "cf_sr", "cf_oma", "cf_omal"]:
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
          Line2D([], [], color=STY["LMMSE"][1], ls="-", label="Genie-aided LMMSE"),
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






