"""Plug-in LMMSE without side information (reviewer item M1/M3): the receiver
substitutes the signature Gram B_s for the embedding Gram B, which it knows
through the signatures, and estimates the noise level from its own outputs,
either on the current block alone (PLG, one-block estimate of 1/rho by
sema_lib.noise_estimate) or as a long-term average over K = 1000 preceding
blocks (PLGL). No training, no signaling. On the matched front end
(B_s = B) the plug-in receiver has the same information as the proposed
receiver, so it is the model-based reference the proposed receiver must be
measured against.

Modes (argv[1]):
  snr      SNR sweep at beta 0.64            -> data/plugin/snr_b0.64.csv
  beta     similarity sweep at 0 dB           -> data/plugin/beta_0dB.csv
  profiles the seven profiles of Table IV     -> data/plugin/profiles.json
  varmix   block-varying pair (signatures = long-term Gram) -> data/plugin/varmix.json
  scaling  user count and dimension           -> data/plugin/scaling_U.csv, scaling_d.csv
  real     AG News BERT SNR sweep             -> data/plugin/real_ag_news_bert-base-uncased_snr.csv
Env: SEMA_N (blocks per point, default 2e5), SEMA_NFLOOR (default 2e6 for the
low-SER points), SEMA_CHUNK, SEMA_REP (real data channel draws per group).
Every row also carries the proposed receiver (SC), the genie-aided LMMSE,
MF, and SR evaluated on the same blocks, and the mean and standard
deviation of the one-block estimate of 1/rho relative to the truth.
"""
from __future__ import annotations
import csv, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *

OUT = DATA / "plugin"; OUT.mkdir(exist_ok=True)
N0 = n_blocks(int(os.environ.get("SEMA_N", "200000"))); NF = n_blocks(int(os.environ.get("SEMA_NFLOOR", "2000000")))
CHUNK0 = int(os.environ.get("SEMA_CHUNK", "10000"))
K_LONG = 1000
RHOINV_MIN = 1e-4          # clamp of the noise estimate (-40 dB): a nonpositive estimate means "noise below resolution"


def rx_plugin(t, Bsig, h, rho_inv):
    """LMMSE (12) with B := B_s and sigma^2 := rho_inv/d per block. rho_inv (batch,)."""
    batch, U, d = t.shape
    if Bsig.dim() == 2:
        Bsig = Bsig.unsqueeze(0).expand(batch, U, U)
    sigma2 = (rho_inv.clamp(min=RHOINV_MIN) / d).view(-1, 1, 1)
    Gam = Bsig * (h.unsqueeze(1) / h.unsqueeze(2))
    Cx = Bsig / d
    Cn = sigma2 * Bsig / (h.unsqueeze(1) * h.unsqueeze(2))
    S = Gam @ Cx @ Gam.transpose(1, 2) + Cn
    W = torch.linalg.solve(S.transpose(1, 2), (Cx @ Gam.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)
    return L._nrm(torch.einsum('buv,bvd->bud', W, t))


def long_term(Bsig, snr, g, sampler):
    """Long-term 1/rho estimate from K_LONG separate blocks drawn by sampler(n, g)."""
    e, h = sampler(K_LONG, g)
    t = L.transmit_receive(e, L.signatures(Bsig), h, snr, g)
    return L.noise_estimate(t, Bsig, h).mean()


@torch.no_grad()
def point(sampler, Bsig, snr, g, N, sc=None, Bprior_fn=None, s_fn=None, tau=None, probe=None, labels=None):
    """sampler(n, g) -> (e, h[, extra]); Bsig (U,U) signature Gram. Bprior_fn(extra) -> per-block
    true Gram for the genie-aided LMMSE (defaults to Bsig). Returns metrics per scheme."""
    tau = tau or L.CFG["tau"]
    CHUNK = min(CHUNK0, N)
    rl = long_term(Bsig, snr, g, lambda n, gg: sampler(n, gg)[:2])
    acc = {}; rel = []
    for _ in range(max(N // CHUNK, 1)):
        smp = sampler(CHUNK, g)
        e, h = smp[0], smp[1]; extra = smp[2] if len(smp) > 2 else None
        t = L.transmit_receive(e, L.signatures(Bsig), h, snr, g)
        ri = L.noise_estimate(t, Bsig, h)
        rel.append(((ri / (1.0 / L.snr_lin(snr))) - 1.0).cpu())
        s = s_fn(extra) if s_fn else None
        out = {}
        out["PLG"] = L.metrics(rx_plugin(t, Bsig, h, ri), e, tau, s)
        out["PLGL"] = L.metrics(rx_plugin(t, Bsig, h, rl.expand(CHUNK)), e, tau, s)
        Bp = Bprior_fn(extra) if Bprior_fn else None
        out["LMMSE"] = L.metrics(L.rx_lmmse(t, Bsig, h, snr, Bp)[0], e, tau, s)
        out["MF"] = L.metrics(L.rx_mf(t), e, tau, s)
        out["SR"] = L.metrics(L.rx_sr(t, Bsig, h), e, tau, s)
        if sc is not None:
            out["SC"] = L.metrics(sc(t, h), e, tau, s)
        if probe is not None and labels is not None:
            y = labels(extra)
            for m in out:
                eh = {"PLG": rx_plugin(t, Bsig, h, ri), "PLGL": rx_plugin(t, Bsig, h, rl.expand(CHUNK)),
                      "LMMSE": L.rx_lmmse(t, Bsig, h, snr, Bp)[0], "MF": L.rx_mf(t), "SR": L.rx_sr(t, Bsig, h),
                      "SC": sc(t, h) if sc is not None else None}[m]
                out[m]["acc"] = (probe(eh.reshape(-1, e.shape[-1])).argmax(1) == y.reshape(-1)).float().mean().item()
        accumulate(acc, out)
    res = finalize(acc)
    rel = torch.cat(rel)
    res["_extra"] = dict(rhoinv_rel_mean=rel.mean().item(), rhoinv_rel_std=rel.std().item(),
                         rhoinv_long_rel=(rl.item() * L.snr_lin(snr) - 1.0))
    return res


def sym_sampler(U, d, beta):
    a = math.sqrt(beta)
    def f(n, g):
        e, s = L.sample_embeddings(n, torch.full((n, U), a, device=L.DEVICE), d, g)
        return e, L.sample_gains(n, U, g)
    return f


def write_csv(rows, name):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", OUT / name, flush=True)


def row_of(res, head):
    row = dict(head)
    for m in ["PLG", "PLGL", "LMMSE", "MF", "SR", "SC"]:
        if m in res:
            for k in ["ser", "cos", "acc"]:
                if k in res[m]:
                    row[f"{k}_{m}"] = res[m][k]
    row.update(res["_extra"]); row["n_decisions"] = res["PLG"]["n"]
    return row


def fmt(res):
    return " ".join(f"{m}:{res[m]['ser']:.2e}" for m in ["PLG", "PLGL", "LMMSE", "MF", "SR", "SC"] if m in res) \
        + f" | 1/rho rel {res['_extra']['rhoinv_rel_mean']:+.2f}+-{res['_extra']['rhoinv_rel_std']:.2f}"


def main(mode):
    U, d = 4, 64; t0 = time.time()
    if mode == "snr":
        sc = load_sc(U); beta = 0.64
        B = L.affinity_matrix(torch.full((U,), math.sqrt(beta), device=L.DEVICE)); rows = []
        for snr in range(-6, 25, 2):
            res = point(sym_sampler(U, d, beta), B, float(snr), L.gen(7000 + snr), NF if snr >= 4 else N0, sc=sc)
            rows.append(row_of(res, {"snr": snr})); print(f"[{snr:3d} dB] {fmt(res)} ({time.time()-t0:.0f}s)", flush=True)
        write_csv(rows, "snr_b0.64.csv")
    elif mode == "beta":
        sc = load_sc(U); rows = []
        for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
            B = L.affinity_matrix(torch.full((U,), math.sqrt(beta), device=L.DEVICE))
            res = point(sym_sampler(U, d, beta), B, 0.0, L.gen(7100 + int(beta * 100)), NF if beta >= 0.75 else N0, sc=sc)
            rows.append(row_of(res, {"beta": beta})); print(f"[beta {beta:.2f}] {fmt(res)} ({time.time()-t0:.0f}s)", flush=True)
        write_csv(rows, "beta_0dB.csv")
    elif mode == "profiles":
        import exp3_profiles as E3
        sc = load_sc(U); table = {}
        for snr in [0.0, 10.0]:
            for prof in ["sym", "asym", "nearfar", "twoclust", "negcorr"]:
                def sampler(n, g, prof=prof):
                    e, s, B, h = E3.sample(prof, n, g); return e, h, (s, B)
                e0, s0, B0, h0 = E3.sample(prof, 2, L.gen(1))
                res = point(sampler, B0, snr, L.gen(3000 + int(snr) + hash(prof) % 100), N0, sc=sc,
                            s_fn=lambda x: x[0])
                table[f"{prof}_{snr:.0f}dB"] = res; print(f"[{snr:.0f} dB] {prof:9s} {fmt(res)} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(table, open(OUT / "profiles.json", "w"), indent=1)
    elif mode == "varmix":
        import exp4_varmix as E4
        sc = load_sc(U); Bbar = E4.avg_B(); table = {}
        for snr in [0.0, 10.0]:
            def sampler(n, g):
                e, s, a = E4.sample_varmix(n, g); return e, L.sample_gains(n, U, g), (s, a)
            res = point(sampler, Bbar, snr, L.gen(4000 + int(snr)), N0, sc=sc, s_fn=lambda x: x[0],
                        Bprior_fn=lambda x: L.affinity_matrix(x[1]))
            table[f"{snr:.0f}dB"] = res; print(f"[varmix {snr:.0f} dB] {fmt(res)} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(table, open(OUT / "varmix.json", "w"), indent=1)
    elif mode == "scaling":
        rows = []
        for Uu in [2, 3, 4, 5, 6, 8]:
            sc = load_sc(Uu, 64)
            for snr in [0.0, 10.0]:
                # symmetric beta = 0.64 profile at every U, as in exp5_scaling.sweep
                acc = {}; g = L.gen(5000 + Uu + int(snr)); rel = []
                for _ in range(N0 // CHUNK0):
                    a = torch.full((CHUNK0, Uu), math.sqrt(0.64), device=L.DEVICE); e, _ = L.sample_embeddings(CHUNK0, a, 64, g)
                    h = L.sample_gains(CHUNK0, Uu, g); Bb = L.affinity_matrix(a[0])
                    t = L.transmit_receive(e, L.signatures(Bb), h, snr, g); ri = L.noise_estimate(t, Bb, h)
                    rel.append((ri * L.snr_lin(snr) - 1).cpu())
                    out = {"PLG": L.metrics(rx_plugin(t, Bb, h, ri), e), "LMMSE": L.metrics(L.rx_lmmse(t, Bb, h, snr)[0], e),
                           "MF": L.metrics(L.rx_mf(t), e), "SC": L.metrics(sc(t, h), e)}
                    accumulate(acc, out)
                res = finalize(acc); rel = torch.cat(rel)
                res["_extra"] = dict(rhoinv_rel_mean=rel.mean().item(), rhoinv_rel_std=rel.std().item(), rhoinv_long_rel=float("nan"))
                rows.append(row_of(res, {"U": Uu, "snr": snr})); print(f"[U={Uu} {snr:.0f} dB] {fmt(res)} ({time.time()-t0:.0f}s)", flush=True)
        write_csv(rows, "scaling_U.csv")
        rows = []; sc64 = load_sc(4, 64)
        for dd in [32, 64, 128, 256]:
            for snr in [0.0, 10.0]:
                acc = {}; g = L.gen(6000 + dd + int(snr)); rel = []
                for _ in range(N0 // CHUNK0):
                    a = torch.full((CHUNK0, 4), math.sqrt(0.64), device=L.DEVICE); e, _ = L.sample_embeddings(CHUNK0, a, dd, g); h = L.sample_gains(CHUNK0, 4, g)
                    Bb = L.affinity_matrix(a[0]); t = L.transmit_receive(e, L.signatures(Bb), h, snr, g); ri = L.noise_estimate(t, Bb, h)
                    rel.append((ri * L.snr_lin(snr) - 1).cpu())
                    out = {"PLG": L.metrics(rx_plugin(t, Bb, h, ri), e), "LMMSE": L.metrics(L.rx_lmmse(t, Bb, h, snr)[0], e),
                           "MF": L.metrics(L.rx_mf(t), e), "SC": L.metrics(sc64(t, h), e)}
                    accumulate(acc, out)
                res = finalize(acc); rel = torch.cat(rel)
                res["_extra"] = dict(rhoinv_rel_mean=rel.mean().item(), rhoinv_rel_std=rel.std().item(), rhoinv_long_rel=float("nan"))
                rows.append(row_of(res, {"d": dd, "snr": snr})); print(f"[d={dd} {snr:.0f} dB] {fmt(res)} ({time.time()-t0:.0f}s)", flush=True)
        write_csv(rows, "scaling_d.csv")
    elif mode == "real":
        import exp8_real as E8
        sc = load_sc(U); REP = int(os.environ.get("SEMA_REP", "100"))
        Ztr, Zte, ytr, yte, ncls = E8.load_corpus("ag_news", "bert-base-uncased")
        g = L.gen(9000); n_sup = 200 if QUICK else 4000; n_eval = 200 if QUICK else 2000
        grp_sup = E8.nn_groups(Ztr, n_sup, g); grp_eval = E8.nn_groups(Zte, n_eval, g)
        beta_bar = E8.group_stats(Ztr, grp_sup)["beta_mean"]
        Bsig = L.affinity_matrix(torch.full((U,), math.sqrt(max(beta_bar, 0.0)), device=L.DEVICE))
        probe = E8.train_probe(Ztr, ytr, ncls); yg = yte[grp_eval]; rows = []
        snrs = [int(x) for x in os.environ.get("SEMA_SNRS", "-4,-2,0,2,4,6,8,10,12,14,16,18,20").split(",")]
        for snr in snrs:
            gg = L.gen(9100 + snr)
            # the evaluation groups are fixed; sampler cycles through them with fresh channel draws
            state = {"i": 0}
            def sampler(n, g2):
                idx = torch.arange(state["i"], state["i"] + n) % n_eval; state["i"] += n
                return Zte[grp_eval[idx]], L.sample_gains(n, U, g2), yg[idx]
            def sup_sampler(n, g2):
                idx = torch.randint(0, len(grp_sup), (n,), generator=g2, device=L.DEVICE)
                return Ztr[grp_sup[idx]], L.sample_gains(n, U, g2)
            tau = L.CFG["tau"]; CHUNK = min(CHUNK0, n_eval)
            rl = long_term(Bsig, snr, gg, sup_sampler); acc = {}; rel = []
            for r in range(REP):
                for i in range(0, n_eval, CHUNK):
                    e, h, y = sampler(min(CHUNK, n_eval - i), gg)
                    t = L.transmit_receive(e, L.signatures(Bsig), h, snr, gg); ri = L.noise_estimate(t, Bsig, h)
                    rel.append((ri * L.snr_lin(snr) - 1).cpu())
                    ehs = {"PLG": rx_plugin(t, Bsig, h, ri), "PLGL": rx_plugin(t, Bsig, h, rl.expand(len(e))),
                           "LMMSE": L.rx_lmmse(t, Bsig, h, snr)[0], "MF": L.rx_mf(t), "SR": L.rx_sr(t, Bsig, h), "SC": sc(t, h)}
                    out = {}
                    for m, eh in ehs.items():
                        mt = L.metrics(eh, e, tau); mt["acc"] = (probe(eh.reshape(-1, d)).argmax(1) == y.reshape(-1)).float().mean().item(); out[m] = mt
                    accumulate(acc, out)
            res = finalize(acc); rel = torch.cat(rel)
            res["_extra"] = dict(rhoinv_rel_mean=rel.mean().item(), rhoinv_rel_std=rel.std().item(), rhoinv_long_rel=(rl.item() * L.snr_lin(snr) - 1))
            rows.append(row_of(res, {"snr": snr})); print(f"[real {snr:3d} dB] {fmt(res)} acc PLG {res['PLG']['acc']:.3f} SC {res['SC']['acc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        write_csv(rows, "real_ag_news_bert-base-uncased_snr.csv")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "snr")
