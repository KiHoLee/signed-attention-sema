"""E7: three small studies.
 (a) similarity-estimation error: the signatures use the true beta = 0.64,
     the side-information receivers are given beta + delta; SC reads none.
     -> data/mismatch.csv
 (b) the expectation-only front end of prior work: d x d Haar-mixture masks
     M_u = sum_k A_uk U_k (Cholesky weights, independent Haar U_k) with the
     SAME U x U receivers; shows the interference floor the exact
     signature construction removes. -> data/haar_masks.csv
 (c) complexity: parameters and multiply-accumulates per block, measured
     batched inference time. -> data/complexity.json
"""
from __future__ import annotations
import csv, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *

U, d, BETA = 4, 64, 0.64
N = n_blocks(100_000); CHUNK = min(25_000, N)


def haar(n, dd, g):
    G = torch.randn(n, dd, dd, generator=g, device=L.DEVICE)
    Q, R = torch.linalg.qr(G)
    return Q * torch.sign(torch.diagonal(R, dim1=1, dim2=2)).unsqueeze(1)


def haar_mixture_chain(e, B, h, snr, g):
    """r = sum_v h_v M_v e_v + n with M_u = sum_k A_uk U_k (d x d), then
    t_u = M_u^T r / h_u. Masks redrawn per block (seed shared in practice)."""
    n, Uu, dd = e.shape
    A = torch.linalg.cholesky(B)                              # (U,U) lower
    Us = torch.stack([haar(n, dd, g) for _ in range(Uu)], dim=1)   # (n,U,d,d)
    M = torch.einsum('uk,nkij->nuij', A, Us)                  # (n,U,d,d)
    x = torch.einsum('nuij,nuj->nui', M, e)                   # masked embeddings
    r = (h.unsqueeze(-1) * x).sum(1)
    r = r + math.sqrt(L.sigma2_of(snr, dd)) * torch.randn(r.shape, generator=g, device=L.DEVICE)
    t = torch.einsum('nuji,nj->nui', M, r) / h.unsqueeze(-1)
    return t


def main():
    sc = load_sc(U)
    t0 = time.time()
    # ---------- (a) mismatch ----------
    a = torch.full((CHUNK, U), math.sqrt(BETA), device=L.DEVICE)
    B = L.affinity_matrix(a[0])
    rows = []
    for snr in [0.0, 10.0]:
        g = L.gen(8000 + int(snr))
        es, hs, ts = [], [], []
        for _ in range(N // CHUNK):
            e, _ = L.sample_embeddings(CHUNK, a, d, g)
            h = L.sample_gains(CHUNK, U, g)
            es.append(e); hs.append(h); ts.append(L.transmit_receive(e, L.signatures(B), h, snr, g))
        with torch.no_grad():
            ser_sc = np.mean([L.metrics(sc(t, h), e)["ser"] for t, e, h in zip(ts, es, hs)])
        for dlt in np.round(np.arange(-0.2, 0.21, 0.05), 2):
            bh = min(0.98, max(1e-4, BETA + dlt))
            Bh = L.affinity_matrix(torch.full((U,), math.sqrt(bh), device=L.DEVICE))
            ser_sr = np.mean([L.metrics(L.rx_sr(t, Bh, h), e)["ser"] for t, e, h in zip(ts, es, hs)])
            ser_lm = np.mean([L.metrics(L.rx_lmmse(t, Bh, h, snr)[0], e)["ser"] for t, e, h in zip(ts, es, hs)])
            rows.append(dict(snr=snr, delta=dlt, ser_SR=ser_sr, ser_LMMSE=ser_lm, ser_SC=ser_sc))
            print(f"(a) {snr:.0f} dB delta {dlt:+.2f} SR {ser_sr:.2e} LMMSE {ser_lm:.2e} SC {ser_sc:.2e}", flush=True)
    with open(DATA / "mismatch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---------- (b) Haar-mixture masks ----------
    rows = []
    NH = 2000 if QUICK else 20_000
    for snr in [0, 10, 20, 30, 40]:
        g = L.gen(8100 + snr)
        acc = {}
        for _ in range(max(1, NH // 2000)):
            e, s = L.sample_embeddings(2000, a[:2000], d, g)
            h = L.sample_gains(2000, U, g)
            t = haar_mixture_chain(e, B, h, snr, g)
            out = {"SC": L.metrics(sc(t, h), e, s=s),
                   "LMMSE": L.metrics(L.rx_lmmse(t, B, h, snr)[0], e, s=s),
                   "SR": L.metrics(L.rx_sr(t, B, h), e, s=s),
                   "MF": L.metrics(L.rx_mf(t), e, s=s)}
            # exact signature chain for reference
            te = L.transmit_receive(e, L.signatures(B), h, snr, g)
            out["LMMSE_exact"] = L.metrics(L.rx_lmmse(te, B, h, snr)[0], e, s=s)
            out["SR_exact"] = L.metrics(L.rx_sr(te, B, h), e, s=s)
            out["SC_exact"] = L.metrics(sc(te, h), e, s=s)
            accumulate(acc, out)
        res = finalize(acc)
        row = {"snr": snr}
        for m, r in res.items():
            row[f"ser_{m}"] = r["ser"]; row[f"cos_{m}"] = r["cos"]
        rows.append(row)
        print(f"(b) {snr} dB " + " ".join(f"{m}:{r['ser']:.2e}/{r['cos']:.3f}" for m, r in res.items()), flush=True)
    with open(DATA / "haar_masks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---------- (c) complexity ----------
    soft = load_sc(U, softmax=True); att = load_att(U)
    hid = L.CFG["hidden"]
    macs_sc = U * U * d + L.n_params(sc) + U * U * d  # Gram, one per network parameter, combination
    comp = dict(params_SC=L.n_params(sc), params_SCsoft=L.n_params(soft), params_ATT=L.n_params(att),
                macs_SC=macs_sc, macs_SR_apply=U * U * d, macs_LMMSE_apply=U * U * d,
                macs_linear_matrix=U ** 3)
    g = L.gen(1)
    e, _ = L.sample_embeddings(20_000, a[:1].expand(20_000, U), d, g)
    h = L.sample_gains(20_000, U, g)
    t = L.transmit_receive(e, L.signatures(B), h, 10.0, g)
    with torch.no_grad():
        for name, m in [("SC", sc), ("SCsoft", soft), ("ATT", att)]:
            call = (lambda: m(t, h)) if isinstance(m, L.GramSC) else (lambda: m(t))
            call()
            if L.DEVICE.type == "cuda": torch.cuda.synchronize()
            t1 = time.time(); call()
            if L.DEVICE.type == "cuda": torch.cuda.synchronize()
            comp[f"us_per_block_{name}"] = (time.time() - t1) / 20_000 * 1e6
    comp["device"] = str(L.DEVICE)
    json.dump(comp, open(DATA / "complexity.json", "w"), indent=1)
    print("(c)", json.dumps(comp), f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
