"""E2: SER versus per-user SNR (-6 to 24 dB) at beta in {0.3, 0.64, 0.9}
for the full receiver set; records the learned SC off-diagonal weight and
the LMMSE off-diagonal weight (sign change at the Prop. 2 boundary) and the
closed-form MSE of LMMSE, SR, OMA and OMA+LMMSE.
Output: data/snr_sweep_b{beta}.csv
"""
from __future__ import annotations
import csv, math, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *

U, d = 4, 64
N = n_blocks(200_000); CHUNK = min(25_000, N)
SNRS = list(range(-6, 25, 2))


def main():
    models = dict(SC=load_sc(U), SCsoft=load_sc(U, softmax=True), ATT=load_att(U))
    for beta in [0.3, 0.64, 0.9]:
        rows = []
        t0 = time.time()
        a = torch.full((CHUNK, U), math.sqrt(beta), device=L.DEVICE)
        B = L.affinity_matrix(a[0])
        for snr in SNRS:
            g = L.gen(2000 + snr + int(100 * beta))
            def data_fn(batch, gg, snr=snr):
                ap = L.sample_profile(batch, U, gg)
                e, _ = L.sample_embeddings(batch, ap, d, gg)
                return e, L.affinity_matrix(ap), L.sample_gains(batch, U, gg), snr
            models["TLC"] = train_tlc(U, d, data_fn)
            acc = {}
            for _ in range(N // CHUNK):
                e, s = L.sample_embeddings(CHUNK, a, d, g)
                h = L.sample_gains(CHUNK, U, g)
                out, extra = run_suite(e, B, B, h, snr, g, models, s=s)
                accumulate(acc, out, extra)
            res = finalize(acc)
            rho = L.snr_lin(snr)
            row = {"snr": snr}
            for m in ORDER:
                for k in ["ser", "cos", "pcos"]:
                    row[f"{k}_{m}"] = res[m][k]
            row["n_decisions"] = res["SC"]["n"]
            row.update(res["_extra"])
            row["mse_cf_lmmse"] = L.mse_lmmse_sym(U, beta, rho)
            row["mse_cf_sr"] = L.mse_sr_sym(U, beta, rho)
            row["mse_cf_omal"] = L.mse_lmmse_sym(U, beta, rho, orthogonal_sig=True)
            row["mse_cf_oma_wiener"] = 1 / (1 + rho)
            row["boundary_db"] = L.snr_boundary_sym(U, beta)
            rows.append(row)
            print(f"[b={beta}] {snr:3d} dB " + " ".join(f"{m}:{res[m]['ser']:.2e}" for m in ORDER)
                  + f"  Wsc {res['_extra']['SC_offdiag_mean']:+.3f} Wl {res['_extra']['LMMSE_offdiag_mean']:+.3f}"
                  + f" ({time.time()-t0:.0f}s)", flush=True)
        with open(DATA / f"snr_sweep_b{beta}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
