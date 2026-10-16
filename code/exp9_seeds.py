"""E9: training-seed spread and the decorrelation-penalty ablation.
Evaluates the three seeds of the main SC receiver and the lam = 0.1
variant at beta = 0.64 for SNR in {-6, 0, 6, 10, 16} dB (synthetic, U = 4,
d = 64, 1e5 blocks per point) and on the AG News BERT groups at 0 and
10 dB (2000 groups x 20 draws). Output: data/seeds.csv
"""
from __future__ import annotations
import csv, math, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *
import exp8_real as R

U, d, BETA = 4, 64, 0.64
N = n_blocks(100_000); CHUNK = min(25_000, N)
MODELS = {"seed1": dict(seed=1), "seed2": dict(seed=2), "seed3": dict(seed=3), "lam0.1": dict(seed=1, lam=0.1)}


@torch.no_grad()
def synth(model, snr, g):
    a = torch.full((CHUNK, U), math.sqrt(BETA), device=L.DEVICE)
    B = L.affinity_matrix(a[0]); acc = {}
    for _ in range(N // CHUNK):
        e, _ = L.sample_embeddings(CHUNK, a, d, g)
        h = L.sample_gains(CHUNK, U, g)
        t = L.transmit_receive(e, L.signatures(B), h, snr, g)
        accumulate(acc, {"SC": L.metrics(model(t, h), e), "LMMSE": L.metrics(L.rx_lmmse(t, B, h, snr)[0], e)})
    return finalize(acc)


@torch.no_grad()
def real(model, snr, g, Zte, grp, Bsig, probe, yte):
    acc = {}
    for _ in range(5 if QUICK else 20):
        for i in range(0, len(grp), 2000):
            E = Zte[grp[i:i + 2000]]; y = yte[grp[i:i + 2000]]
            h = L.sample_gains(len(E), U, g)
            t = L.transmit_receive(E, L.signatures(Bsig), h, snr, g)
            eh = model(t, h); mt = L.metrics(eh, E)
            mt["acc"] = (probe(eh.reshape(-1, d)).argmax(1) == y.reshape(-1)).float().mean().item()
            accumulate(acc, {"SC": mt})
    return finalize(acc)


def main():
    t0 = time.time()
    Ztr, Zte, ytr, yte, ncls = R.load_corpus("ag_news", "bert-base-uncased")
    g = L.gen(9000)
    grp_sup = R.nn_groups(Ztr, 200 if QUICK else 4000, g); grp_eval = R.nn_groups(Zte, 200 if QUICK else 2000, g)
    bb = R.group_stats(Ztr, grp_sup)["beta_mean"]
    Bsig = L.affinity_matrix(torch.full((U,), math.sqrt(bb), device=L.DEVICE))
    probe = R.train_probe(Ztr, ytr, ncls)
    rows = []
    for name, kw in MODELS.items():
        m = load_sc(U, d, **kw)
        for snr in [-6, 0, 6, 10, 16]:
            r = synth(m, float(snr), L.gen(12000 + snr))
            rows.append(dict(model=name, data="synthetic", snr=snr, ser=r["SC"]["ser"], cos=r["SC"]["cos"],
                             ser_lmmse=r["LMMSE"]["ser"], cos_lmmse=r["LMMSE"]["cos"], acc=float("nan")))
            print(f"[{name} synth {snr:3d} dB] SC {r['SC']['ser']:.2e}/{r['SC']['cos']:.4f} LMMSE {r['LMMSE']['ser']:.2e}/{r['LMMSE']['cos']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        for snr in [0, 10]:
            r = real(m, float(snr), L.gen(12100 + snr), Zte, grp_eval, Bsig, probe, yte)
            rows.append(dict(model=name, data="agnews", snr=snr, ser=r["SC"]["ser"], cos=r["SC"]["cos"],
                             ser_lmmse=float("nan"), cos_lmmse=float("nan"), acc=r["SC"]["acc"]))
            print(f"[{name} real {snr:3d} dB] SC {r['SC']['ser']:.2e}/{r['SC']['cos']:.4f} acc {r['SC']['acc']:.4f}", flush=True)
    with open(DATA / "seeds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
