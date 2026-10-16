"""E0b: Monte Carlo verification of the closed-form MSE of the linear
receivers (Prop. 1 and the general forms), for the validation figure.
Unnormalized estimates. Sweeps: SNR at beta = 0.64, and beta at 0 and
10 dB (U = 4, d = 64, h = 1). Also the general asymmetric case (Table).
Output: data/theory_snr.csv, data/theory_beta.csv, data/theory_asym.csv
"""
from __future__ import annotations
import csv, math, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L

DATA = Path(__file__).resolve().parents[1] / "data"
U, d, N = 4, 64, 20000


def mc(beta, snr, g, Bsig_beta=None):
    rho = L.snr_lin(snr)
    a = torch.full((N, U), math.sqrt(beta), device=L.DEVICE)
    e, _ = L.sample_embeddings(N, a, d, g)
    B = L.affinity_matrix(a[0])
    Bs = B if Bsig_beta is None else L.affinity_matrix(torch.full((U,), math.sqrt(Bsig_beta), device=L.DEVICE))
    h = torch.ones(N, U, device=L.DEVICE)
    t = L.transmit_receive(e, L.signatures(Bs), h, snr, g)
    Bb, Bsb = B.expand(N, U, U), Bs.expand(N, U, U)
    W = L.lmmse_weights(Bsb, h, snr, d, Bb)
    mse_l = ((torch.einsum('buv,bvd->bud', W, t) - e) ** 2).sum(-1).mean().item()
    Gam = Bsb * (h.unsqueeze(1) / h.unsqueeze(2))
    mse_sr = ((torch.linalg.solve(Gam, t) - e) ** 2).sum(-1).mean().item()
    I = torch.eye(U, device=L.DEVICE)
    to = L.transmit_receive(e, I, h, snr, g)
    mse_w = ((to / (1 + 1 / rho) - e) ** 2).sum(-1).mean().item()
    Wo = L.lmmse_weights(I.expand(N, U, U), h, snr, d, Bb)
    mse_ol = ((torch.einsum('buv,bvd->bud', Wo, t * 0 + to) - e) ** 2).sum(-1).mean().item()
    bs = beta if Bsig_beta is None else Bsig_beta
    return dict(mc_lmmse=mse_l, cf_lmmse=L.mse_lmmse_sym(U, beta, rho, beta_s=bs),
                mc_sr=mse_sr, cf_sr=L.mse_sr_sym(U, bs, rho),
                mc_oma=mse_w, cf_oma=1 / (1 + rho),
                mc_omal=mse_ol, cf_omal=L.mse_lmmse_sym(U, beta, rho, orthogonal_sig=True))


def main():
    g = L.gen(31415)
    rows = []
    for snr in range(-10, 31, 2):
        r = mc(0.64, snr, g); r["snr"] = snr; rows.append(r)
        print(f"snr {snr:3d} L {r['mc_lmmse']:.4f}/{r['cf_lmmse']:.4f} SR {r['mc_sr']:.4f}/{r['cf_sr']:.4f} "
              f"OMA {r['mc_oma']:.4f}/{r['cf_oma']:.4f} OMAL {r['mc_omal']:.4f}/{r['cf_omal']:.4f}")
    with open(DATA / "theory_snr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    rows = []
    for snr in [0, 10]:
        for beta in np.round(np.arange(0, 0.96, 0.05), 2):
            r = mc(float(beta), snr, g); r["snr"] = snr; r["beta"] = float(beta); rows.append(r)
    with open(DATA / "theory_beta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # general asymmetric check
    a1 = torch.tensor([0.9, 0.75, 0.55, 0.3], device=L.DEVICE)
    h1 = torch.tensor([1.0, 0.7, 1.4, 0.5], device=L.DEVICE)
    rows = []
    for snr in [0, 10]:
        rho = L.snr_lin(snr)
        e, _ = L.sample_embeddings(N, a1.expand(N, U), d, g)
        B = L.affinity_matrix(a1.expand(N, U)); h = h1.expand(N, U)
        t = L.transmit_receive(e, L.signatures(B), h, snr, g)
        W = L.lmmse_weights(B, h, snr, d)
        mc_l = ((torch.einsum('buv,bvd->bud', W, t) - e) ** 2).sum(-1).mean(0)
        cf_l = L.mse_lmmse_general(B[0], h1, rho, d)
        Gam = B * (h.unsqueeze(1) / h.unsqueeze(2))
        mc_s = ((torch.linalg.solve(Gam, t) - e) ** 2).sum(-1).mean(0)
        cf_s = L.mse_sr_general(B[0], h1, rho, d)
        for u in range(U):
            rows.append(dict(snr=snr, user=u + 1, a=a1[u].item(), h=h1[u].item(),
                             mc_lmmse=mc_l[u].item(), cf_lmmse=cf_l[u].item(),
                             mc_sr=mc_s[u].item(), cf_sr=cf_s[u].item()))
    with open(DATA / "theory_asym.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main()
