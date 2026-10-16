"""Design study for the SC receiver (training-time choices), U = 4, d = 64.
Each variant is trained for ITERS iterations and evaluated at
(0 dB, beta 0.64), (10 dB, beta 0.45), (10 dB, beta 0.64) and on an SNR
sweep at beta 0.64 (N blocks per point) against LMMSE, MF and OMA.
Variants: SNR range of training, input features (raw Gram or
correlation + log energies), gain conditioning, tail-aware loss, longer
training. Output: data/variants.json and data/variants_snr.csv
"""
from __future__ import annotations
import csv, json, math, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *

U, d = 4, 64
ITERS = 100 if QUICK else 6000
CKPT.mkdir(parents=True, exist_ok=True)
N = n_blocks(100_000); CHUNK = min(25_000, N)
VARIANTS2 = {
    "raw_log":      dict(snr=list(range(-8, 25, 2)), feat="raw", use_h=False, tail=0.0, iters=ITERS, kind="log"),
    "norm_log":     dict(snr=list(range(-8, 25, 2)), feat="norm", use_h=False, tail=0.0, iters=ITERS, kind="log"),
    "norm_h_log":   dict(snr=list(range(-8, 25, 2)), feat="norm", use_h=True, tail=0.0, iters=ITERS, kind="log"),
    "norm_h_log_w": dict(snr=list(range(-8, 25, 2)), feat="norm", use_h=True, tail=0.0, iters=2 * ITERS, kind="log", hidden=128),
}
VARIANTS = {
    "base":        dict(snr=list(range(0, 25, 2)), feat="raw", use_h=False, tail=0.0, iters=ITERS),
    "lowsnr":      dict(snr=list(range(-8, 25, 2)), feat="raw", use_h=False, tail=0.0, iters=ITERS),
    "norm":        dict(snr=list(range(-8, 25, 2)), feat="norm", use_h=False, tail=0.0, iters=ITERS),
    "norm_h":      dict(snr=list(range(-8, 25, 2)), feat="norm", use_h=True, tail=0.0, iters=ITERS),
    "norm_h_tail": dict(snr=list(range(-8, 25, 2)), feat="norm", use_h=True, tail=1.0, iters=ITERS),
    "norm_tail":   dict(snr=list(range(-8, 25, 2)), feat="norm", use_h=False, tail=1.0, iters=ITERS),
    "norm_h_long": dict(snr=list(range(-8, 25, 2)), feat="norm", use_h=True, tail=0.0, iters=3 * ITERS),
}
POINTS = [(0.0, 0.64), (10.0, 0.45), (10.0, 0.64), (-6.0, 0.64), (4.0, 0.64), (16.0, 0.64)]


@torch.no_grad()
def eval_point(model, snr, beta, g):
    a = torch.full((CHUNK, U), math.sqrt(beta), device=L.DEVICE)
    B = L.affinity_matrix(a[0])
    acc = {}
    for _ in range(N // CHUNK):
        e, s = L.sample_embeddings(CHUNK, a, d, g)
        h = L.sample_gains(CHUNK, U, g)
        t = L.transmit_receive(e, L.signatures(B), h, snr, g)
        W = model.weights(t, h)
        offd = W - torch.diag_embed(torch.diagonal(W, dim1=1, dim2=2))
        out = {"SC": L.metrics(model(t, h), e), "LMMSE": L.metrics(L.rx_lmmse(t, B, h, snr)[0], e),
               "MF": L.metrics(L.rx_mf(t), e)}
        I = torch.eye(U, device=L.DEVICE)
        out["OMA"] = L.metrics(L.rx_mf(L.transmit_receive(e, I, h, snr, g)), e)
        accumulate(acc, out, {"SC_offdiag_mean": (offd.sum() / (CHUNK * U * (U - 1))).item()})
    return finalize(acc)


def main():
    t0 = time.time()
    res = {}
    which = VARIANTS2 if len(sys.argv) > 1 and sys.argv[1] == "2" else VARIANTS
    outname = "variants2.json" if which is VARIANTS2 else "variants.json"
    for name, v in which.items():
        model = L.GramSC(U, v.get("hidden", L.CFG["hidden"]), feat=v["feat"], use_h=v["use_h"])
        model = L.train_receiver(model, U, d, v["iters"], 256, 2e-3, 1, snr_set=v["snr"],
                                 log=None, tail=v["tail"], kind=v.get("kind", "lin"))
        torch.save(model.state_dict(), CKPT / f"variant_{name}.pt")
        res[name] = {}
        for snr, beta in POINTS:
            r = eval_point(model, snr, beta, L.gen(int(1e4 + 100 * snr + 1000 * beta)))
            res[name][f"{snr:.0f}dB_b{beta}"] = r
            print(f"[{name:12s} {snr:4.0f} dB b={beta}] SC {r['SC']['ser']:.2e}/{r['SC']['cos']:.3f} "
                  f"LMMSE {r['LMMSE']['ser']:.2e}/{r['LMMSE']['cos']:.3f} MF {r['MF']['ser']:.2e}/{r['MF']['cos']:.3f} "
                  f"OMA {r['OMA']['ser']:.2e} W {r['_extra']['SC_offdiag_mean']:+.3f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(DATA / outname, "w"), indent=1)


if __name__ == "__main__":
    main()
