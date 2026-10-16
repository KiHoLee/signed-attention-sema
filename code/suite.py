"""Receiver suite shared by the experiments: loads the trained receivers
and evaluates every scheme on one batch of blocks through the physical
chain. Scheme labels are fixed here (one label dictionary for figures,
tables and text; see replot.LABELS which copies these keys).

  SC        proposed signed user-wise attention (GramSC)
  SCsoft    same architecture, softmax (convex) weights
  ATT       projection-based attention receiver (shared-embedding principle)
  TLC       static trainable linear combiner (retrained per experiment)
  LMMSE     informed LMMSE, isotropic prior B/d, needs B, H, sigma^2
  MOM       moment-matched LMMSE: prior and noise estimated from the Gram
  SR        similarity-rejecting (zero forcing), needs B, H
  MF        single-user matched filter, normalized
  OMA       orthogonal signatures (same resource and power), matched filter
  OMAL      orthogonal signatures + LMMSE combining with the prior B/d
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = DATA / "ckpt"
QUICK = os.environ.get("SEMA_QUICK", "0") == "1"      # tiny sizes for smoke tests
ORDER = ["SC", "SCsoft", "ATT", "TLC", "LMMSE", "MOM", "SR", "MF", "OMA", "OMAL"]


def n_blocks(n):
    return 2000 if QUICK else n


def load_sc(U, d=64, seed=1, lam=0.0, softmax=False, tag=None, feat=None, use_h=None):
    name = tag or (f"gramsc_U{U}_d{d}_s{seed}" + ("_lam" if lam > 0 else "") + ("_soft" if softmax else ""))
    feat = L.CFG["feat"] if feat is None else feat
    use_h = L.CFG["use_h"] if use_h is None else use_h
    m = L.GramSC(U, L.CFG["hidden"], softmax=softmax, feat=feat, use_h=use_h)
    m.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location=L.DEVICE, weights_only=True))
    return m.to(L.DEVICE).eval()


def load_att(U, d=64, seed=1):
    m = L.ProjAttention(U, d)
    m.load_state_dict(torch.load(CKPT / f"att_U{U}_d{d}_s{seed}.pt", map_location=L.DEVICE, weights_only=True))
    return m.to(L.DEVICE).eval()


def train_tlc(U, d, data_fn, iters=600, seed=1):
    """data_fn(batch, g) -> (e, Bsig, h, snr): the TLC is trained on the
    same front end and distribution it is evaluated on (strongest static
    linear scheme without side information)."""
    tlc = L.TLC(U)
    return L.train_receiver(tlc, U, d, iters if not QUICK else 50, 256, 1e-2, seed,
                            data_fn=data_fn, log=None)


@torch.no_grad()
def run_suite(e, Bsig, Bprior, h, snr, g, models, which=None, s=None, tau=None):
    """Evaluate all schemes on one batch. Bsig: signature Gram (U,U) or
    (batch,U,U); Bprior: embedding Gram for the LMMSE prior (same shape).
    Returns dict label -> metrics dict (cos, ser, nmse, n[, pcos]) and the
    SC mean off-diagonal weight."""
    tau = tau or L.CFG["tau"]
    batch, U, d = e.shape
    which = which or ORDER
    C = L.signatures(Bsig)
    t = L.transmit_receive(e, C, h, snr, g)
    out, extra = {}, {}
    if "SC" in which:
        W = models["SC"].weights(t, h)
        out["SC"] = L.metrics(models["SC"](t, h), e, tau, s)
        offd = W - torch.diag_embed(torch.diagonal(W, dim1=1, dim2=2))
        extra["SC_offdiag_mean"] = (offd.sum() / (batch * U * (U - 1))).item()
        extra["SC_diag_mean"] = torch.diagonal(W, dim1=1, dim2=2).mean().item()
    if "SCsoft" in which and "SCsoft" in models:
        out["SCsoft"] = L.metrics(models["SCsoft"](t, h), e, tau, s)
    if "ATT" in which and "ATT" in models:
        out["ATT"] = L.metrics(models["ATT"](t), e, tau, s)
    if "TLC" in which and "TLC" in models:
        out["TLC"] = L.metrics(models["TLC"](t), e, tau, s)
    if "LMMSE" in which:
        eh, W = L.rx_lmmse(t, Bsig, h, snr, Bprior)
        out["LMMSE"] = L.metrics(eh, e, tau, s)
        offd = W - torch.diag_embed(torch.diagonal(W, dim1=1, dim2=2))
        extra["LMMSE_offdiag_mean"] = (offd.sum() / (batch * U * (U - 1))).item()
    if "MOM" in which:
        eh, _ = L.rx_lmmse_mom(t, Bsig, h)
        out["MOM"] = L.metrics(eh, e, tau, s)
    if "SR" in which:
        out["SR"] = L.metrics(L.rx_sr(t, Bsig, h), e, tau, s)
    if "MF" in which:
        out["MF"] = L.metrics(L.rx_mf(t), e, tau, s)
    if "OMA" in which or "OMAL" in which:
        I = torch.eye(U, device=L.DEVICE)
        to = L.transmit_receive(e, I, h, snr, g)
        if "OMA" in which:
            out["OMA"] = L.metrics(L.rx_mf(to), e, tau, s)
        if "OMAL" in which:
            eh, _ = L.rx_lmmse(to, I, h, snr, Bprior)
            out["OMAL"] = L.metrics(eh, e, tau, s)
    return out, extra


def accumulate(acc, out, extra=None):
    """Weighted accumulation of metrics over chunks (by n)."""
    for m, r in out.items():
        a = acc.setdefault(m, {})
        n = r["n"]
        for k, v in r.items():
            if k == "n":
                a["n"] = a.get("n", 0) + n
            else:
                a[k] = a.get(k, 0.0) + v * n
    if extra:
        x = acc.setdefault("_extra", {})
        for k, v in extra.items():
            x[k] = x.get(k, 0.0) + v
        x["_chunks"] = x.get("_chunks", 0) + 1
    return acc


def finalize(acc):
    res = {}
    for m, a in acc.items():
        if m == "_extra":
            res["_extra"] = {k: v / a["_chunks"] for k, v in a.items() if k != "_chunks"}
            continue
        n = a["n"]
        res[m] = {k: (v / n if k != "n" else v) for k, v in a.items()}
    return res
