"""Floor study: SC-rho (long-term noise estimate input) and TLC-g (gain-scaled
static combiner) against SC, TLC, MF, SR, and the genie-aided LMMSE.
Modes (argv[1]): snr  -> SNR sweep at beta 0.64 (data/variants_floor/snr_b0.64.csv)
                 beta -> similarity sweep at 0 dB   (data/variants_floor/beta_0dB.csv)
Env: SEMA_N (blocks per point, default 2e5), SEMA_NFLOOR (default 2e6, used for
SNR >= 4 dB and beta >= 0.75), SEMA_CHUNK.
"""
from __future__ import annotations
import csv, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *

U, d = 4, 64
N0 = n_blocks(int(os.environ.get("SEMA_N", "200000"))); NF = n_blocks(int(os.environ.get("SEMA_NFLOOR", "2000000")))
CHUNK0 = int(os.environ.get("SEMA_CHUNK", "10000"))
OUT = DATA / "variants_floor"; OUT.mkdir(exist_ok=True)
KS = [1000, 100]


def offdiag_mean(W):
    b, U_, _ = W.shape
    return ((W - torch.diag_embed(torch.diagonal(W, dim1=1, dim2=2))).sum() / (b * U_ * (U_ - 1))).item()


@torch.no_grad()
def point(models, B, snr, g, N):
    """One (beta, snr) point: returns dict scheme -> metrics (+ weights)."""
    CHUNK = min(CHUNK0, N)
    a = torch.full((CHUNK, U), math.sqrt(float(B[0, 1])), device=L.DEVICE)
    # long-term noise estimates from separate draws of K blocks (same B, random gains)
    est = {}
    for K in KS:
        eK, _ = L.sample_embeddings(K, a[:K] if K <= CHUNK else a[:1].expand(K, U), d, g)
        hK = L.sample_gains(K, U, g)
        tK = L.transmit_receive(eK, L.signatures(B), hK, snr, g)
        est[K] = L.noise_estimate(tK, B, hK).mean()
    acc = {}
    for _ in range(N // CHUNK):
        e, s = L.sample_embeddings(CHUNK, a, d, g)
        h = L.sample_gains(CHUNK, U, g)
        t = L.transmit_receive(e, L.signatures(B), h, snr, g)
        out, extra = {}, {}
        W = models["SC"].weights(t, h); out["SC"] = L.metrics(models["SC"](t, h), e); extra["w_SC"] = offdiag_mean(W)
        for K in KS:
            m = models[f"SCrho{K}"]; ri = est[K].expand(CHUNK)
            W = m.weights(t, h, ri); out[f"SCrho{K}"] = L.metrics(m(t, h, ri), e); extra[f"w_SCrho{K}"] = offdiag_mean(W)
        out["TLC"] = L.metrics(models["TLC"](t), e)
        out["TLCg"] = L.metrics(models["TLCg"](t, h), e)
        eh, Wl = L.rx_lmmse(t, B, h, snr, B); out["LMMSE"] = L.metrics(eh, e); extra["w_LMMSE"] = offdiag_mean(Wl)
        out["SR"] = L.metrics(L.rx_sr(t, B, h), e)
        out["MF"] = L.metrics(L.rx_mf(t), e)
        accumulate(acc, out, extra)
    res = finalize(acc)
    res["_extra"].update({f"rhoinv_est{K}": est[K].item() for K in KS}); res["_extra"]["rhoinv_true"] = 1 / L.snr_lin(snr)
    return res


def models_for(snr, g):
    def data_fn(batch, gg, snr=snr):
        ap = L.sample_profile(batch, U, gg)
        e, _ = L.sample_embeddings(batch, ap, d, gg)
        return e, L.affinity_matrix(ap), L.sample_gains(batch, U, gg), snr
    return dict(TLC=train_tlc(U, d, data_fn), TLCg=train_tlc_g(U, d, data_fn))


def main(mode):
    base = dict(SC=load_sc(U))
    for K in KS:
        base[f"SCrho{K}"] = load_sc(U, tag="gramsc_U4_d64_s1_rho", use_rho=True, K=K)
    rows = []; t0 = time.time()
    if mode == "snr":
        beta = 0.64; B = L.affinity_matrix(torch.full((U,), math.sqrt(beta), device=L.DEVICE))
        for snr in range(-6, 25, 2):
            g = L.gen(5000 + snr); models = dict(base, **models_for(float(snr), g))
            res = point(models, B, float(snr), g, NF if snr >= 4 else N0)
            row = {"snr": snr, "n_decisions": res["SC"]["n"]}
            for m in ["SC", "SCrho1000", "SCrho100", "TLC", "TLCg", "LMMSE", "SR", "MF"]:
                row[f"ser_{m}"] = res[m]["ser"]; row[f"cos_{m}"] = res[m]["cos"]
            row.update(res["_extra"]); rows.append(row)
            print(f"[{snr:3d} dB] " + " ".join(f"{m}:{res[m]['ser']:.2e}" for m in ["SC", "SCrho1000", "SCrho100", "TLC", "TLCg", "LMMSE", "MF"])
                  + f" | w SC {res['_extra']['w_SC']:+.2f} rho {res['_extra']['w_SCrho1000']:+.2f} L {res['_extra']['w_LMMSE']:+.2f}"
                  + f" | 1/rho est {res['_extra']['rhoinv_est1000']:.3g}/{res['_extra']['rhoinv_est100']:.3g} true {res['_extra']['rhoinv_true']:.3g} ({time.time()-t0:.0f}s)", flush=True)
        name = "snr_b0.64.csv"
    else:
        snr = 0.0; g = L.gen(5100); models = dict(base, **models_for(snr, g))
        for beta in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
            B = L.affinity_matrix(torch.full((U,), math.sqrt(beta), device=L.DEVICE))
            res = point(models, B, snr, g, NF if beta >= 0.75 else N0)
            row = {"beta": beta, "n_decisions": res["SC"]["n"]}
            for m in ["SC", "SCrho1000", "SCrho100", "TLC", "TLCg", "LMMSE", "SR", "MF"]:
                row[f"ser_{m}"] = res[m]["ser"]; row[f"cos_{m}"] = res[m]["cos"]
            row.update(res["_extra"]); rows.append(row)
            print(f"[beta {beta:.2f}] " + " ".join(f"{m}:{res[m]['ser']:.2e}" for m in ["SC", "SCrho1000", "SCrho100", "TLC", "TLCg", "LMMSE", "MF"]) + f" ({time.time()-t0:.0f}s)", flush=True)
        name = "beta_0dB.csv"
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", OUT / name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "snr")
