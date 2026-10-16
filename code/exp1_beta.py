"""E1: SER, cosine and private cosine versus pairwise similarity beta for
the full receiver set at 0 and 10 dB (U = 4, d = 64, symmetric profile,
Rayleigh gains). 2e5 blocks (8e5 decisions) per point. Also records the
mean off-diagonal weight of the SC receiver and of the LMMSE receiver.
Output: data/beta_sweep_{snr}dB.csv
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
N = n_blocks(int(os.environ.get("SEMA_N", "200000"))); CHUNK = min(int(os.environ.get("SEMA_CHUNK", "25000")), N)
BETAS = [float(x) for x in os.environ["SEMA_BETAS"].split(",")] if os.environ.get("SEMA_BETAS") else list(np.round(np.arange(0.0, 0.91, 0.05), 2))
SNRS = [float(x) for x in os.environ["SEMA_SNRS"].split(",")] if os.environ.get("SEMA_SNRS") else [0.0, 10.0]
SUFFIX = os.environ.get("SEMA_SUFFIX", "")


def main():
    models = dict(SC=load_sc(U), SCsoft=load_sc(U, softmax=True), ATT=load_att(U))
    for snr in SNRS:
        g = L.gen(1000 + int(snr))
        # TLC on the mixed training distribution at this SNR
        def data_fn(batch, gg, snr=snr):
            a = L.sample_profile(batch, U, gg)
            e, _ = L.sample_embeddings(batch, a, d, gg)
            return e, L.affinity_matrix(a), L.sample_gains(batch, U, gg), snr
        models["TLC"] = train_tlc(U, d, data_fn)
        rows = []
        t0 = time.time()
        for beta in BETAS:
            acc = {}
            a = torch.full((CHUNK, U), math.sqrt(beta), device=L.DEVICE)
            B = L.affinity_matrix(a[0])
            for _ in range(N // CHUNK):
                e, s = L.sample_embeddings(CHUNK, a, d, g)
                h = L.sample_gains(CHUNK, U, g)
                out, extra = run_suite(e, B, B, h, snr, g, models, s=s)
                accumulate(acc, out, extra)
            res = finalize(acc)
            row = {"beta": beta}
            for m in ORDER:
                for k in ["ser", "cos", "pcos"]:
                    row[f"{k}_{m}"] = res[m][k]
            row["n_decisions"] = res["SC"]["n"]
            row.update(res["_extra"])
            rows.append(row)
            print(f"[{snr:.0f} dB] beta {beta:.2f} " + " ".join(f"{m}:{res[m]['ser']:.2e}" for m in ORDER)
                  + f"  Wsc {res['_extra']['SC_offdiag_mean']:+.3f} Wl {res['_extra']['LMMSE_offdiag_mean']:+.3f}"
                  + f" ({time.time()-t0:.0f}s)", flush=True)
        with open(DATA / f"beta_sweep_{snr:.0f}dB{SUFFIX}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
