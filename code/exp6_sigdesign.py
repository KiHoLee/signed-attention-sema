"""E6: signature correlation as a design parameter. The embeddings keep
beta = 0.64 while the signature Gram correlation beta_s sweeps 0..0.9;
closed-form LMMSE/SR MSE (Prop. 3) and Monte Carlo SER of LMMSE, SR and
the SC receiver (trained with matched signatures, applied unchanged).
Also the Prop. 3 optimum beta_s*(rho) and the MSE at beta_s*, beta_s = beta
and beta_s = 0 over SNR. Outputs: data/sigdesign_{snr}dB.csv, data/sigopt.csv
"""
from __future__ import annotations
import csv, math, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *

U, d, BETA = 4, 64, 0.64
N = n_blocks(100_000); CHUNK = min(25_000, N)
BS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.64, 0.7, 0.8, 0.9]
WHICH = ["SC", "LMMSE", "MOM", "SR", "MF"]


def main():
    models = dict(SC=load_sc(U))
    t0 = time.time()
    a = torch.full((CHUNK, U), math.sqrt(BETA), device=L.DEVICE)
    Be = L.affinity_matrix(a[0])
    for snr in [0.0, 10.0]:
        rho = L.snr_lin(snr)
        rows = []
        for bs in BS:
            g = L.gen(7000 + int(snr) + int(100 * bs))
            Bs = L.affinity_matrix(torch.full((U,), math.sqrt(bs), device=L.DEVICE))
            acc = {}
            for _ in range(N // CHUNK):
                e, s = L.sample_embeddings(CHUNK, a, d, g)
                h = L.sample_gains(CHUNK, U, g)
                out, extra = run_suite(e, Bs, Be, h, snr, g, models, which=WHICH, s=s)
                accumulate(acc, out, extra)
            res = finalize(acc)
            row = {"beta_s": bs, "mse_cf_lmmse": L.mse_lmmse_sym(U, BETA, rho, beta_s=bs),
                   "mse_cf_sr": L.mse_sr_sym(U, bs, rho)}
            for m in WHICH:
                row[f"ser_{m}"] = res[m]["ser"]; row[f"cos_{m}"] = res[m]["cos"]
            rows.append(row)
            print(f"[{snr:.0f} dB] beta_s {bs:.2f} mseL {row['mse_cf_lmmse']:.4f} "
                  + " ".join(f"{m}:{res[m]['ser']:.2e}" for m in WHICH) + f" ({time.time()-t0:.0f}s)", flush=True)
        with open(DATA / f"sigdesign_{snr:.0f}dB.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    rows = []
    for snr in range(-10, 31, 1):
        rho = L.snr_lin(snr)
        bopt = L.beta_s_opt(U, BETA, rho)
        rows.append(dict(snr=snr, beta_s_opt=bopt,
                         mse_opt=L.mse_lmmse_sym(U, BETA, rho, beta_s=bopt),
                         mse_matched=L.mse_lmmse_sym(U, BETA, rho),
                         mse_orth=L.mse_lmmse_sym(U, BETA, rho, orthogonal_sig=True),
                         mse_oma_wiener=1 / (1 + rho), mse_sr_matched=L.mse_sr_sym(U, BETA, rho)))
    with open(DATA / "sigopt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
