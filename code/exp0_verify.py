"""E0: verification of the library against the closed forms of the paper.

 (a) the literal physical chain (spread, superpose, noise, despread) equals
     the reduced model in distribution: compare the first two moments of the
     outputs over many blocks;
 (b) Monte Carlo per-user MSE of the SR and LMMSE receivers (unnormalized)
     agrees with the closed forms mse_sr_sym / mse_lmmse_sym (Prop. 1) and
     with mse_*_general for asymmetric gains;
 (c) the LMMSE off-diagonal weight changes sign at the closed-form boundary
     snr_boundary_sym (Prop. 2);
 (d) the OMA (orthogonal-signature) chain reproduces the single-user Wiener
     MSE and the OMA+LMMSE closed form.
Writes data/verify.csv (one row per check). Seeded by its own generator.
"""
from __future__ import annotations
import csv, math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)
U, d = 4, 64
rows = []


def unnormalized_lmmse(t, B, h, snr):
    W = L.lmmse_weights(B, h, snr, d)
    return torch.einsum('buv,bvd->bud', W, t)


def unnormalized_sr(t, B, h):
    Gam = B * (h.unsqueeze(1) / h.unsqueeze(2))
    return torch.linalg.solve(Gam, t)


def main():
    global rows
    g = L.gen(12345)
    n = 20000
    # ---------- (a) chain equivalence ----------
    a = torch.full((n, U), math.sqrt(0.64), device=L.DEVICE)
    e, s = L.sample_embeddings(n, a, d, g)
    B = L.affinity_matrix(a)
    h = L.sample_gains(n, U, g)
    snr = 6.0
    t_phys = L.transmit_receive(e, L.signatures(B), h, snr, g)
    t_red = L.transmit_receive_reduced(e, B, h, snr, g)
    sig = torch.einsum('buv,bvd->bud', B * (h.unsqueeze(1) / h.unsqueeze(2)), e)
    # noise cross-covariance per coordinate, averaged over blocks, scaled by h_u h_v
    def noise_cov(t):
        nz = (t - sig) * h.unsqueeze(-1)
        return torch.einsum('bud,bvd->uv', nz, nz) / (n * d)
    C1, C2 = noise_cov(t_phys), noise_cov(t_red)
    sigma2 = L.sigma2_of(snr, d)
    dev = ((C1 - sigma2 * B[0]).abs().max() / sigma2).item()
    dev2 = ((C2 - sigma2 * B[0]).abs().max() / sigma2).item()
    print(f"(a) noise cov vs sigma^2 B: physical dev {dev:.4f}, reduced dev {dev2:.4f}")
    rows.append(["chain_noise_cov_dev_physical", dev]); rows.append(["chain_noise_cov_dev_reduced", dev2])
    # signal part is deterministic given e, h: check exactly
    t_clean = L.transmit_receive(e, L.signatures(B), h, 200.0, g)
    err = (t_clean - sig).abs().max().item()
    print(f"(a) signal part exact (max abs dev) {err:.2e}")
    rows.append(["chain_signal_exact_maxdev", err])

    # ---------- (b) closed-form MSE, symmetric h=1 ----------
    for beta in [0.0, 0.3, 0.64, 0.9]:
        for snr in [0.0, 10.0, 20.0]:
            rho_c = L.snr_lin(snr)
            a = torch.full((n, U), math.sqrt(beta), device=L.DEVICE)
            e, _ = L.sample_embeddings(n, a, d, g)
            B = L.affinity_matrix(a)
            h = torch.ones(n, U, device=L.DEVICE)
            t = L.transmit_receive(e, L.signatures(B), h, snr, g)
            mc_l = ((unnormalized_lmmse(t, B, h, snr) - e) ** 2).sum(-1).mean().item()
            cf_l = L.mse_lmmse_sym(U, beta, rho_c)
            if beta < 0.99:
                mc_s = ((unnormalized_sr(t, B, h) - e) ** 2).sum(-1).mean().item()
                cf_s = L.mse_sr_sym(U, beta, rho_c)
            else:
                mc_s = cf_s = float('nan')
            print(f"(b) beta={beta:.2f} snr={snr:4.1f}  LMMSE mc {mc_l:.5f} cf {cf_l:.5f} | SR mc {mc_s:.5f} cf {cf_s:.5f}")
            rows += [[f"mse_lmmse_b{beta}_s{snr}_mc", mc_l], [f"mse_lmmse_b{beta}_s{snr}_cf", cf_l],
                     [f"mse_sr_b{beta}_s{snr}_mc", mc_s], [f"mse_sr_b{beta}_s{snr}_cf", cf_s]]
    # asymmetric gains and profile: general closed forms
    a1 = torch.tensor([0.9, 0.75, 0.55, 0.3], device=L.DEVICE)
    h1 = torch.tensor([1.0, 0.7, 1.4, 0.5], device=L.DEVICE)
    snr = 10.0; rho_c = L.snr_lin(snr)
    e, _ = L.sample_embeddings(n, a1.expand(n, U), d, g)
    B = L.affinity_matrix(a1.expand(n, U))
    h = h1.expand(n, U)
    t = L.transmit_receive(e, L.signatures(B), h, snr, g)
    mc_l = ((unnormalized_lmmse(t, B, h, snr) - e) ** 2).sum(-1).mean(0)
    cf_l = L.mse_lmmse_general(B[0], h1, rho_c, d)
    mc_s = ((unnormalized_sr(t, B, h) - e) ** 2).sum(-1).mean(0)
    cf_s = L.mse_sr_general(B[0], h1, rho_c, d)
    print("(b) asym LMMSE mc", mc_l.cpu().numpy().round(4), "cf", cf_l.cpu().numpy().round(4))
    print("(b) asym SR    mc", mc_s.cpu().numpy().round(4), "cf", cf_s.cpu().numpy().round(4))
    for u in range(U):
        rows += [[f"asym_lmmse_u{u}_mc", mc_l[u].item()], [f"asym_lmmse_u{u}_cf", cf_l[u].item()],
                 [f"asym_sr_u{u}_mc", mc_s[u].item()], [f"asym_sr_u{u}_cf", cf_s[u].item()]]

    # ---------- (c) sign boundary ----------
    for beta in [0.2, 0.64, 0.9]:
        b_db = L.snr_boundary_sym(U, beta)
        a = torch.full((1, U), math.sqrt(beta), device=L.DEVICE)
        B = L.affinity_matrix(a); h = torch.ones(1, U, device=L.DEVICE)
        offs = []
        for snr in [b_db - 3, b_db - 0.5, b_db + 0.5, b_db + 3]:
            W = L.lmmse_weights(B, h, snr, d)[0]
            offs.append(W[0, 1].item())
        print(f"(c) beta={beta:.2f} boundary {b_db:5.2f} dB; W_12 at -3,-0.5,+0.5,+3 dB: "
              + " ".join(f"{o:+.4f}" for o in offs))
        rows.append([f"boundary_db_b{beta}", b_db])
        for k, o in zip(["m3", "m05", "p05", "p3"], offs):
            rows.append([f"W12_b{beta}_{k}", o])

    # ---------- (d) OMA chain ----------
    beta, snr = 0.64, 10.0
    rho_c = L.snr_lin(snr)
    a = torch.full((n, U), math.sqrt(beta), device=L.DEVICE)
    e, _ = L.sample_embeddings(n, a, d, g)
    B = L.affinity_matrix(a); h = torch.ones(n, U, device=L.DEVICE)
    I = torch.eye(U, device=L.DEVICE)
    t = L.transmit_receive(e, I, h, snr, g)                 # orthogonal signatures
    wiener = t / (1 + 1 / rho_c)                             # single-user Wiener scaling
    mc_su = ((wiener - e) ** 2).sum(-1).mean().item()
    cf_su = 1 / (1 + rho_c)
    W = L.lmmse_weights(I.expand(n, U, U), h, snr, d, Bprior=B)
    mc_ol = ((torch.einsum('buv,bvd->bud', W, t) - e) ** 2).sum(-1).mean().item()
    cf_ol = L.mse_lmmse_sym(U, beta, rho_c, orthogonal_sig=True)
    print(f"(d) OMA single-user mc {mc_su:.5f} cf {cf_su:.5f} | OMA+LMMSE mc {mc_ol:.5f} cf {cf_ol:.5f}")
    rows += [["oma_su_mc", mc_su], ["oma_su_cf", cf_su], ["oma_lmmse_mc", mc_ol], ["oma_lmmse_cf", cf_ol]]

    with open(DATA / "verify.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["check", "value"]); w.writerows(rows)
    print("wrote", DATA / "verify.csv", "device", L.DEVICE)


if __name__ == "__main__":
    main()
