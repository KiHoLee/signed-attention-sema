"""E8: real foundation-model embeddings through the exact front end.

Corpora (cached mean-pooled encoder outputs, data/real/emb_<corpus>_<model>.pt
with train 8000 / test 2000 sentences and topic labels):
  ag_news  / bert-base-uncased       (4 topics)   main corpus, SNR sweep
  sst2     / bert-base-uncased       (2 classes)  second corpus
  ag_news  / distilbert-base-uncased (4 topics)   second encoder
Preparation: center and unit-normalize with the train mean, PCA to d = 64
fitted on the train pool, renormalize. Correlated user groups: one test
sentence and its U-1 nearest neighbors (user order permuted). Independent
groups: U random test sentences. The signatures use one similarity value
estimated on nearest-neighbor groups of the train pool (not per group).
Receivers: SC (trained on the synthetic model only, applied unchanged),
SCsoft, ATT, TLC (trained on train-pool groups through the same chain at
each SNR), LMMSE-e (empirical Ud x Ud prior from train groups), LMMSE
(isotropic prior with the estimated similarity), SR, MF, OMA, OMAL.
Metrics: SER (tau), cosine, and topic accuracy of a linear probe trained on
clean train embeddings and applied to the recovered embeddings.
Each test group is sent through REP independent channel draws.
Outputs: data/real_<corpus>_<model>_snr.csv, data/real_indep.csv,
         data/real_tau.csv, data/real_stats.json
"""
from __future__ import annotations
import csv, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *

U, d = 4, 64
REP = 5 if QUICK else int(os.environ.get("SEMA_REP", "100"))   # channel draws per group (2000 groups -> 2e5 blocks)
SNRS = [int(x) for x in os.environ["SEMA_SNRS"].split(",")] if os.environ.get("SEMA_SNRS") else list(range(-4, 21, 2))
SUFFIX = os.environ.get("SEMA_SUFFIX", "")   # rerun of the main corpus only, written to real_*{SUFFIX}.csv
WHICH = ["SC", "SCsoft", "ATT", "TLC", "LMMSE", "MOM", "SR", "MF", "OMA", "OMAL"]
REAL = DATA / "real"
CORPORA = [("ag_news", "bert-base-uncased"), ("sst2", "bert-base-uncased"),
           ("ag_news", "distilbert-base-uncased")]
if SUFFIX:
    CORPORA = CORPORA[:1]


def load_corpus(corpus, model):
    o = torch.load(REAL / f"emb_{corpus}_{model}.pt", map_location="cpu", weights_only=False)
    Xtr, Xte = o["train"].double(), o["test"].double()
    mu = Xtr.mean(0, keepdim=True)
    Xtr, Xte = Xtr - mu, Xte - mu
    Xtr = Xtr / Xtr.norm(dim=1, keepdim=True); Xte = Xte / Xte.norm(dim=1, keepdim=True)
    _, _, Vt = torch.linalg.svd(Xtr, full_matrices=False)
    P = Vt[:d].T
    Ztr, Zte = Xtr @ P, Xte @ P
    Ztr = Ztr / Ztr.norm(dim=1, keepdim=True); Zte = Zte / Zte.norm(dim=1, keepdim=True)
    return (Ztr.float().to(L.DEVICE), Zte.float().to(L.DEVICE),
            o["train_labels"].to(L.DEVICE), o["test_labels"].to(L.DEVICE), int(o["n_classes"]))


def nn_groups(Z, n_groups, g):
    """Anchor + U-1 nearest neighbors (cosine), order permuted. One group per
    anchor; anchors drawn without replacement."""
    G = Z @ Z.T
    G.fill_diagonal_(-2.0)
    N = Z.shape[0]
    anchors = torch.randperm(N, generator=g, device=L.DEVICE)[:n_groups]
    nb = G[anchors].topk(U - 1, dim=1).indices
    grp = torch.cat([anchors.unsqueeze(1), nb], dim=1)
    perm = torch.rand(n_groups, U, generator=g, device=L.DEVICE).argsort(dim=1)
    return torch.gather(grp, 1, perm)


def random_groups(Z, n_groups, g):
    return torch.stack([torch.randperm(Z.shape[0], generator=g, device=L.DEVICE)[:U] for _ in range(n_groups)])


def group_stats(Z, grp):
    E = Z[grp]                                    # (n,U,d)
    G = torch.einsum('nud,nvd->nuv', E, E)
    iu = torch.triu_indices(U, U, 1)
    cosv = G[:, iu[0], iu[1]].mean(1)
    sv = torch.linalg.svdvals(E)
    r1 = sv[:, 0] ** 2 / (sv ** 2).sum(1)
    return dict(beta_mean=cosv.mean().item(), beta_std=cosv.std().item(),
                beta_p10=cosv.quantile(0.1).item(), beta_p90=cosv.quantile(0.9).item(),
                rank1_mean=r1.mean().item(), rank1_std=r1.std().item(),
                frac_neg_pair=(G[:, iu[0], iu[1]] < 0).float().mean().item())


def train_probe(Ztr, ytr, n_classes, seed=0):
    torch.manual_seed(seed)
    W = torch.nn.Linear(d, n_classes).to(L.DEVICE)
    opt = torch.optim.Adam(W.parameters(), lr=1e-2)
    for _ in range(300):
        loss = torch.nn.functional.cross_entropy(W(Ztr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    return W.eval()


def joint_cov(Z, grp):
    S = Z[grp].reshape(len(grp), -1)
    return (S.T @ S) / len(S)


@torch.no_grad()
def evaluate(Ztr, Zte, ytr, yte, n_classes, grp_sup, grp_eval, Bsig, snr, models, Cx, probe, g, tau):
    """Full suite on grp_eval x REP channel draws. Returns metrics dict."""
    acc = {}
    yg = yte[grp_eval]                            # (n,U)
    n = len(grp_eval)
    chunk = int(os.environ.get("SEMA_EVAL_CHUNK", "2000"))
    rho_inv = None
    if "SCrho" in models:                         # long-term noise estimate from K training-group blocks at this SNR
        K = models["SCrho"].K
        idx = torch.randint(0, len(grp_sup), (K,), generator=g, device=L.DEVICE)
        E2 = Ztr[grp_sup[idx]]; h2 = L.sample_gains(K, U, g)
        t2 = L.transmit_receive(E2, L.signatures(Bsig), h2, snr, g)
        rho_inv = L.noise_estimate(t2, Bsig, h2).mean()
    for r in range(REP):
        for i in range(0, n, chunk):
            E = Zte[grp_eval[i:i + chunk]]
            y = yg[i:i + chunk]
            h = L.sample_gains(len(E), U, g)
            C = L.signatures(Bsig)
            t = L.transmit_receive(E, C, h, snr, g)
            out = {}
            out["SC"] = models["SC"](t, h); out["SCsoft"] = models["SCsoft"](t, h); out["ATT"] = models["ATT"](t)
            out["TLC"] = models["TLC"](t)
            if "SCrho" in models:
                out["SCrho"] = models["SCrho"](t, h, rho_inv.expand(len(E)))
            if "TLCg" in models:
                out["TLCg"] = models["TLCg"](t, h)
            out["LMMSE"] = L.rx_lmmse(t, Bsig, h, snr)[0]
            out["MOM"] = L.rx_lmmse_mom(t, Bsig, h)[0]
            out["SR"] = L.rx_sr(t, Bsig, h); out["MF"] = L.rx_mf(t)
            I = torch.eye(U, device=L.DEVICE)
            to = L.transmit_receive(E, I, h, snr, g)
            out["OMA"] = L.rx_mf(to)
            out["OMAL"] = L.rx_lmmse(to, I, h, snr, Bsig)[0]
            if Cx is not None and not os.environ.get("SEMA_SKIP_EMP"):
                out["LMMSEe"] = L.rx_lmmse_empirical(t, Cx.unsqueeze(0).expand(len(E), U * d, U * d), Bsig, h, snr)
            res = {}
            for m, eh in out.items():
                mt = L.metrics(eh, E, tau)
                mt["acc"] = (probe(eh.reshape(-1, d)).argmax(1) == y.reshape(-1)).float().mean().item()
                res[m] = mt
            accumulate(acc, res)
    # clean-embedding probe accuracy on the same groups (reference)
    return finalize(acc)


def main():
    t0 = time.time()
    tau = L.CFG["tau"]
    models = dict(SC=load_sc(U), SCsoft=load_sc(U, softmax=True), ATT=load_att(U))
    stats = {}
    for corpus, model in CORPORA:
        key = f"{corpus}_{model}"
        Ztr, Zte, ytr, yte, ncls = load_corpus(corpus, model)
        g = L.gen(9000)
        n_sup = 200 if QUICK else 4000
        n_eval = 200 if QUICK else 2000
        grp_sup = nn_groups(Ztr, n_sup, g)
        grp_eval = nn_groups(Zte, n_eval, g)
        st = dict(nn_sup=group_stats(Ztr, grp_sup), nn_eval=group_stats(Zte, grp_eval))
        beta_bar = st["nn_sup"]["beta_mean"]
        Bsig = L.affinity_matrix(torch.full((U,), math.sqrt(max(beta_bar, 0.0)), device=L.DEVICE))
        Cx = joint_cov(Ztr, grp_sup)
        probe = train_probe(Ztr, ytr, ncls)
        st["probe_acc_clean_test"] = (probe(Zte).argmax(1) == yte).float().mean().item()
        st["probe_acc_clean_eval_groups"] = (probe(Zte[grp_eval].reshape(-1, d)).argmax(1) == yte[grp_eval].reshape(-1)).float().mean().item()
        st["n_classes"] = ncls
        stats[key] = st
        print(f"== {key}: beta_bar {beta_bar:.3f} (eval {st['nn_eval']['beta_mean']:.3f}), rank1 {st['nn_eval']['rank1_mean']:.3f}, "
              f"probe clean {st['probe_acc_clean_test']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        snrs = SNRS if corpus == "ag_news" and model == "bert-base-uncased" else [0, 10]
        rows = []
        for snr in snrs:
            def data_fn(batch, gg, snr=snr):
                idx = torch.randint(0, len(grp_sup), (batch,), generator=gg, device=L.DEVICE)
                return Ztr[grp_sup[idx]], Bsig, L.sample_gains(batch, U, gg), snr
            models["TLC"] = train_tlc(U, d, data_fn)
            if os.environ.get("SEMA_VARIANTS"):
                models["SCrho"] = load_sc(U, tag="gramsc_U4_d64_s1_rho", use_rho=True, K=1000)
                models["TLCg"] = train_tlc_g(U, d, data_fn)
            res = evaluate(Ztr, Zte, ytr, yte, ncls, grp_sup, grp_eval, Bsig, snr, models, Cx, probe,
                           L.gen(9100 + snr), tau)
            row = {"snr": snr, "n_decisions": res["SC"]["n"]}
            for m in WHICH + ["LMMSEe", "SCrho", "TLCg"]:
                if m not in res:
                    continue
                for k in ["ser", "cos", "acc"]:
                    row[f"{k}_{m}"] = res[m][k]
            rows.append(row)
            print(f"[{key} {snr:3d} dB] " + " ".join(f"{m}:{res[m]['ser']:.2e}" for m in WHICH + ["LMMSEe", "SCrho", "TLCg"] if m in res)
                  + f" acc SC {res['SC']['acc']:.3f} OMA {res['OMA']['acc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        with open(DATA / f"real_{key}_snr{SUFFIX}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

        if corpus == "ag_news" and model == "bert-base-uncased" and not SUFFIX:
            # independent groups (no correlation): signatures orthogonal (beta = 0)
            grp_ind = random_groups(Zte, n_eval, g)
            grp_sup_i = random_groups(Ztr, n_sup, g)
            st["indep"] = group_stats(Zte, grp_ind)
            Bi = torch.eye(U, device=L.DEVICE)
            Cxi = joint_cov(Ztr, grp_sup_i)
            rows = []
            for snr in [0, 10]:
                def data_fn(batch, gg, snr=snr):
                    idx = torch.randint(0, len(grp_sup_i), (batch,), generator=gg, device=L.DEVICE)
                    return Ztr[grp_sup_i[idx]], Bi, L.sample_gains(batch, U, gg), snr
                models["TLC"] = train_tlc(U, d, data_fn)
                res = evaluate(Ztr, Zte, ytr, yte, ncls, grp_sup_i, grp_ind, Bi, snr, models, Cxi, probe,
                               L.gen(9200 + snr), tau)
                row = {"snr": snr, "n_decisions": res["SC"]["n"]}
                for m in WHICH + ["LMMSEe"]:
                    for k in ["ser", "cos", "acc"]:
                        row[f"{k}_{m}"] = res[m][k]
                rows.append(row)
                print(f"[indep {snr:3d} dB] " + " ".join(f"{m}:{res[m]['ser']:.2e}" for m in WHICH + ["LMMSEe"]), flush=True)
            with open(DATA / "real_indep.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
            # threshold sweep at 0 and 10 dB (nearest-neighbor groups)
            rows = []
            for snr in [0, 10]:
                def data_fn(batch, gg, snr=snr):
                    idx = torch.randint(0, len(grp_sup), (batch,), generator=gg, device=L.DEVICE)
                    return Ztr[grp_sup[idx]], Bsig, L.sample_gains(batch, U, gg), snr
                models["TLC"] = train_tlc(U, d, data_fn)
                for tt in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                    res = evaluate(Ztr, Zte, ytr, yte, ncls, grp_sup, grp_eval, Bsig, snr, models, None, probe,
                                   L.gen(9300 + snr), tt)
                    row = {"snr": snr, "tau": tt}
                    for m in WHICH:
                        row[f"ser_{m}"] = res[m]["ser"]
                    rows.append(row)
                    print(f"[tau {snr} dB {tt:.2f}] " + " ".join(f"{m}:{res[m]['ser']:.2e}" for m in WHICH), flush=True)
            with open(DATA / "real_tau.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(stats, open(DATA / f"real_stats{SUFFIX}.json", "w"), indent=1)
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
