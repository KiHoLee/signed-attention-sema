"""Reviewer check (M7): does the attention receiver leave the matched-filter
curve when trained for twice the steps? Evaluates att_U4_d64_s1 (20,000 steps)
and att_U4_d64_s1_long (40,000 steps) against MF and SC on the beta = 0.64
SNR sweep. Output: data/att_long.csv"""
from __future__ import annotations
import csv, math, os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *

U, d, BETA = 4, 64, 0.64
N = n_blocks(int(os.environ.get("SEMA_N", "50000"))); CHUNK = min(10000, N)


def load_att_tag(tag):
    m = L.ProjAttention(U, d)
    m.load_state_dict(torch.load(CKPT / f"{tag}.pt", map_location=L.DEVICE, weights_only=True))
    return m.to(L.DEVICE).eval()


@torch.no_grad()
def main():
    att, attl, sc = load_att(U), load_att_tag("att_U4_d64_s1_long"), load_sc(U)
    B = L.affinity_matrix(torch.full((U,), math.sqrt(BETA), device=L.DEVICE)); rows = []; t0 = time.time()
    for snr in range(-6, 25, 2):
        g = L.gen(8000 + snr); acc = {}
        for _ in range(N // CHUNK):
            a = torch.full((CHUNK, U), math.sqrt(BETA), device=L.DEVICE); e, _ = L.sample_embeddings(CHUNK, a, d, g)
            h = L.sample_gains(CHUNK, U, g); t = L.transmit_receive(e, L.signatures(B), h, snr, g)
            accumulate(acc, {"ATT": L.metrics(att(t), e), "ATTlong": L.metrics(attl(t), e), "MF": L.metrics(L.rx_mf(t), e), "SC": L.metrics(sc(t, h), e)})
        res = finalize(acc); row = {"snr": snr}
        for m in ["ATT", "ATTlong", "MF", "SC"]:
            row[f"ser_{m}"] = res[m]["ser"]; row[f"cos_{m}"] = res[m]["cos"]
        rows.append(row); print(f"[{snr:3d} dB] " + " ".join(f"{m}: ser {res[m]['ser']:.2e} cos {res[m]['cos']:.4f}" for m in ["ATT", "ATTlong", "MF", "SC"]) + f" ({time.time()-t0:.0f}s)", flush=True)
    with open(DATA / "att_long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
