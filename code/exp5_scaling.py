"""E5: scaling in the user count U (2,3,4,5,6,8; one SC receiver trained per
U at d = 64) and in the embedding dimension d (32,64,128,256; the SC
receiver trained at d = 64 applied unchanged, since its input is the
U x U Gram; the d = 256 trained copy is reported as a check).
beta = 0.64, SNR 0 and 10 dB. Outputs: data/scaling_U.csv, data/scaling_d.csv
"""
from __future__ import annotations
import csv, math, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *

BETA = 0.64
N = n_blocks(200_000); CHUNK = min(25_000, N)
WHICH = ["SC", "TLC", "LMMSE", "MOM", "SR", "MF", "OMA", "OMAL"]


def sweep(U, d, snr, models, g):
    a = torch.full((CHUNK, U), math.sqrt(BETA), device=L.DEVICE)
    B = L.affinity_matrix(a[0])
    acc = {}
    for _ in range(N // CHUNK):
        e, s = L.sample_embeddings(CHUNK, a, d, g)
        h = L.sample_gains(CHUNK, U, g)
        out, extra = run_suite(e, B, B, h, snr, g, models, which=WHICH, s=s)
        accumulate(acc, out, extra)
    return finalize(acc)


def main():
    t0 = time.time()
    rows = []
    for U in [2, 3, 4, 5, 6, 8]:
        d = 64
        models = dict(SC=load_sc(U, d))
        for snr in [0.0, 10.0]:
            def data_fn(batch, gg, snr=snr, U=U):
                ap = L.sample_profile(batch, U, gg)
                e, _ = L.sample_embeddings(batch, ap, d, gg)
                return e, L.affinity_matrix(ap), L.sample_gains(batch, U, gg), snr
            models["TLC"] = train_tlc(U, d, data_fn)
            res = sweep(U, d, snr, models, L.gen(5000 + U + int(snr)))
            row = {"U": U, "snr": snr, "n_decisions": res["SC"]["n"]}
            for m in WHICH:
                row[f"ser_{m}"] = res[m]["ser"]; row[f"cos_{m}"] = res[m]["cos"]
            row.update(res["_extra"])
            rows.append(row)
            print(f"[U={U} {snr:.0f} dB] " + " ".join(f"{m}:{res[m]['ser']:.2e}" for m in WHICH)
                  + f" ({time.time()-t0:.0f}s)", flush=True)
    with open(DATA / "scaling_U.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    rows = []
    U = 4
    sc64 = load_sc(U, 64)
    sc256 = load_sc(U, 256)
    for d in [32, 64, 128, 256]:
        for snr in [0.0, 10.0]:
            def data_fn(batch, gg, snr=snr, d=d):
                ap = L.sample_profile(batch, U, gg)
                e, _ = L.sample_embeddings(batch, ap, d, gg)
                return e, L.affinity_matrix(ap), L.sample_gains(batch, U, gg), snr
            models = dict(SC=sc64, TLC=train_tlc(U, d, data_fn))
            res = sweep(U, d, snr, models, L.gen(6000 + d + int(snr)))
            row = {"d": d, "snr": snr, "n_decisions": res["SC"]["n"]}
            for m in WHICH:
                row[f"ser_{m}"] = res[m]["ser"]; row[f"cos_{m}"] = res[m]["cos"]
            # SC trained at d=256 applied at this d
            res2 = sweep(U, d, snr, dict(SC=sc256), L.gen(6000 + d + int(snr)))
            row["ser_SC256"] = res2["SC"]["ser"]; row["cos_SC256"] = res2["SC"]["cos"]
            rows.append(row)
            print(f"[d={d} {snr:.0f} dB] " + " ".join(f"{m}:{res[m]['ser']:.2e}" for m in WHICH)
                  + f" SC256:{res2['SC']['ser']:.2e} ({time.time()-t0:.0f}s)", flush=True)
    with open(DATA / "scaling_d.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
