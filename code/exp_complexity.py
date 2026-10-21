"""Per-block complexity of every compared receiver (Table VI).

Two measurements per receiver:
  1. Multiply-accumulates (MACs) per block from explicit operation counts,
     U users, d coordinates, n = U d. Conventions:
       - every receiver ends with the per-user normalization nrm(.), U d MACs;
       - a U x U linear solve with k right-hand sides costs U^3/3 + k U^2
         (LU factorization plus forward and backward substitution);
       - matrix products are counted as written in the receiver, without
         exploiting the symmetry of the Gram matrix;
       - the empirical LMMSE receiver is counted with the Kronecker structure
         of Gamma (x) I_d exploited (U n^2 per product with it), which is the
         cheaper of the two ways to form its n x n system; the dense count is
         reported as well.
  2. Wall-clock time per block of the reference implementation in
     sema_lib.py / exp_plugin.py on the CPU of the machine that runs the
     script, batched, median of repeated runs (the device and processor are
     written to the output).

Output: data/complexity_receivers.csv (one row per receiver) and
        data/complexity_receivers.json (the same plus the machine record).
"""
from __future__ import annotations
import csv, json, math, os, platform, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *
from exp_plugin import rx_plugin

U, d = 4, 64
n = U * d
H, DK = 4, 32                 # attention receiver: heads x width
SNR = 10.0


def solve(Um, k):
    return Um ** 3 / 3 + k * Um ** 2


def macs():
    norm = U * d
    apply_w = U * U * d
    lmmse_form = 2 * U ** 3 + U ** 3 + solve(U, U)          # Gamma C Gamma^T, C Gamma^T, solve for W
    sc_params = L.n_params(L.GramSC(U, L.CFG["hidden"], feat=L.CFG["feat"]))
    att_params = L.n_params(L.ProjAttention(U, d))
    feat = U * U + U                                         # correlation form and log diagonal
    rows = {
        "Genie-aided LMMSE": (lmmse_form + apply_w + norm, "4.33U^3 + U^2 d + U d", 0),
        "Plug-in LMMSE": (solve(U, d) + U * d + U ** 3 + lmmse_form + apply_w + norm,
                          "noise estimate (U^3/3 + U^2 d + U d + U^3) + LMMSE (4.33U^3 + U^2 d) + U d", 0),
        "SR": (solve(U, d) + norm, "U^3/3 + U^2 d + U d", 0),
        "MF": (norm, "U d", 0),
        "Moment LMMSE": (U * U * d + 2 * U ** 3 + solve(U, 0) + U + 9 * U ** 3 + lmmse_form + apply_w + norm,
                         "Gram U^2 d + whitening 2.33U^3 + eigendecomposition ~9U^3 + LMMSE 4.33U^3 + U^2 d + U d", 0),
        "Proposed SC": (U * U * d + feat + sc_params + apply_w + norm, "Gram U^2 d + features + parameters + U^2 d + U d", sc_params),
        "Softmax SC": (U * U * d + feat + sc_params + U * U + apply_w + norm, "as SC + U^2 softmax", sc_params),
        "SC-a": (U * U * d + feat + sc_params + apply_w + norm, "as SC", sc_params),
        "Attention receiver": (2 * U * d * H * DK + 2 * U * U * H * DK + U * H * DK * d + U * d + norm,
                               "K,V projections 2 U d H d_k + scores and context 2 U^2 H d_k + output U H d_k d + residual + U d", att_params),
        "TLC": (apply_w + norm, "U^2 d + U d", U * U),
        "TLC-g": (apply_w + 2 * U * d + norm, "U^2 d + 2 U d + U d", U * U),
        "Empirical LMMSE": (2 * U * n * n + solve(n, n) + n * n + norm, "2 U n^2 + n^3/3 + n^3 + n^2 + U d, n = U d (structured)", 0),
    }
    dense_emp = 2 * n ** 3 + solve(n, n) + n * n + norm
    return rows, dense_emp


@torch.no_grad()
def timing(reps=7):
    dev = torch.device("cpu")
    g = L.gen(1)
    nb = 4096
    beta = 0.64
    a = torch.full((nb, U), math.sqrt(beta)); e, _ = L.sample_embeddings(nb, a, d, g)
    B = L.affinity_matrix(a[0]); h = L.sample_gains(nb, U, g)
    t = L.transmit_receive(e, L.signatures(B), h, SNR, g)
    sc, scs = load_sc(U), load_sc(U, softmax=True)
    att = load_att(U)
    tlc, tlcg = L.TLC(U).eval(), L.TLCg(U).eval()
    Bb = B.unsqueeze(0).expand(nb, U, U)
    Cx = torch.eye(n) / d
    fns = {
        "Genie-aided LMMSE": (lambda: L.rx_lmmse(t, B, h, SNR, B), nb),
        "Plug-in LMMSE": (lambda: rx_plugin(t, Bb, h, L.noise_estimate(t, Bb, h)), nb),
        "SR": (lambda: L.rx_sr(t, B, h), nb),
        "MF": (lambda: L.rx_mf(t), nb),
        "Moment LMMSE": (lambda: L.rx_lmmse_mom(t, B, h), nb),
        "Proposed SC": (lambda: sc(t, h), nb),
        "Softmax SC": (lambda: scs(t, h), nb),
        "SC-a": (lambda: sc(t, h), nb),
        "Attention receiver": (lambda: att(t), nb),
        "TLC": (lambda: tlc(t), nb),
        "TLC-g": (lambda: tlcg(t, h), nb),
        "Empirical LMMSE": (lambda: L.rx_lmmse_empirical(t[:64], Cx.unsqueeze(0).expand(64, n, n), B, h[:64], SNR), 64),
    }
    out = {}
    for k, (f, m) in fns.items():
        f()                                    # warm-up
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
        ts.sort(); out[k] = 1e6 * ts[len(ts) // 2] / m
        print(f"  {k:20s} {out[k]:10.3f} us/block", flush=True)
    return out


def main():
    os.environ.setdefault("SEMA_DEVICE", "cpu")
    rows, dense_emp = macs()
    tm = timing()
    rec = []
    for k, (m, formula, p) in rows.items():
        rec.append(dict(receiver=k, macs_per_block=int(round(m)), parameters=p, cpu_us_per_block=round(tm[k], 4), mac_count=formula))
        print(f"{k:20s} MACs {m:12.0f}  params {p:6d}  cpu {tm[k]:9.3f} us/block", flush=True)
    with open(DATA / "complexity_receivers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec[0].keys())); w.writeheader(); w.writerows(rec)
    meta = dict(U=U, d=d, n=n, empirical_lmmse_dense_macs=int(round(dense_emp)),
                torch=torch.__version__, threads=torch.get_num_threads(),
                machine=platform.machine(), processor=platform.processor(), system=platform.platform(),
                note="CPU wall-clock of the batched reference implementations; MACs from operation counts")
    json.dump(dict(meta=meta, receivers=rec), open(DATA / "complexity_receivers.json", "w"), indent=1)
    print("wrote", DATA / "complexity_receivers.csv")


if __name__ == "__main__":
    main()
