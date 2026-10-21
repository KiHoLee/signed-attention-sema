"""Signatures that are NOT matched to the per-block similarity.

Question 1 (reviewer M1, option (a)): when the transmitters agree only on a
long-term similarity, the signatures carry B_bar, not the per-block B, so the
receiver no longer knows B through the signatures. Does the proposed receiver,
which reads the observed Gram matrix, then outperform the plug-in receiver,
which commits to B_s?
Question 2: does the proposed receiver need similarity-matched signatures at
all? With orthogonal signatures (C = I, the OMA front end) the plug-in prior
B_s / d = I / d says the users are independent, whereas the observed Gram
still reveals the similarity.

Receivers (all read the signatures and the gains, the front-end knowledge):
  SC     proposed receiver as published (trained on per-block matched signatures)
  SClt   same architecture, retrained on signature-agnostic blocks (below)
  SCltb  as SClt plus the signature Gram B_s as input (front-end knowledge)
  PLG    plug-in LMMSE: (12) with B := B_s and the one-block noise estimate (22)
  LMMSE  genie-aided: coupling from B_s, prior = true per-block B, true sigma^2
  SR, MF, OMA (C = I, MF), OMAL (C = I, genie prior)
Training of SClt / SCltb, per block: beta_bar ~ U[0, 0.9]; with probability 3/4
share coefficients jittered around sqrt(beta_bar) by +-Delta, Delta ~ U[0, 0.25],
otherwise the mixed training profile; signatures: per-block matched, long-term
symmetric beta_bar, orthogonal, or symmetric with an unrelated beta_s ~ U[0, 0.9],
each with probability 1/4. Same SNR set, loss, optimizer, and steps as SC.

Modes (argv[1]):
  train  -> data/ckpt/gramsc_U4_d64_s1_lt.pt, _ltb.pt
  synth  -> data/longterm/synth_{matched,lt,orth,ltmix}.csv
  real   -> data/longterm/real_{lt,orth}.csv (AG News, BERT)
Env: SEMA_N (blocks per point, default 2e5), SEMA_NFLOOR (default 4e5 for SNR >= 4 dB),
SEMA_REP (real data channel draws per group, default 100), SEMA_ITERS.
"""
from __future__ import annotations
import csv, math, os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *
from exp_plugin import rx_plugin

U, d = 4, 64
OUT = DATA / "longterm"; OUT.mkdir(exist_ok=True)
N0 = n_blocks(int(os.environ.get("SEMA_N", "200000"))); NF = n_blocks(int(os.environ.get("SEMA_NFLOOR", "400000")))
CHUNK0 = int(os.environ.get("SEMA_CHUNK", "10000"))
BETA_BAR, DELTA = 0.64, 0.15


def sym(beta):
    """Symmetric Gram with off-diagonal beta; beta (n,) -> (n,U,U)."""
    a = torch.sqrt(beta.clamp(min=0.0)).unsqueeze(1).expand(-1, U)
    return L.affinity_matrix(a)


def jitter(n, g, beta_bar, delta):
    """Share coefficients jittered independently around sqrt(beta_bar)."""
    base = torch.sqrt(torch.as_tensor(beta_bar, device=L.DEVICE, dtype=L.DT)).reshape(-1, 1)
    dl = torch.as_tensor(delta, device=L.DEVICE, dtype=L.DT).reshape(-1, 1)
    u = 2 * torch.rand(n, U, generator=g, device=L.DEVICE) - 1
    return (base + dl * u).clamp(0.0, 0.97)


# ----------------------------------------------------------------- training
def train(tag, use_bs, iters, seed=1):
    C = L.CFG
    g = L.gen(seed); torch.manual_seed(seed)
    m = L.GramSC(U, C["hidden"], feat=C["feat"], use_h=C["use_h"], use_bs=use_bs).to(L.DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=C["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    n = C["batch"]; t0 = time.time()
    for it in range(iters):
        bbar = torch.rand(n, generator=g, device=L.DEVICE) * 0.9
        a = jitter(n, g, bbar, torch.rand(n, generator=g, device=L.DEVICE) * 0.25)
        mix = torch.rand(n, 1, generator=g, device=L.DEVICE) < 0.25
        a = torch.where(mix, L.sample_profile(n, U, g), a)
        e, _ = L.sample_embeddings(n, a, d, g); B = L.affinity_matrix(a)
        mode = torch.randint(0, 4, (n,), generator=g, device=L.DEVICE)
        bs = torch.where(mode == 1, bbar, torch.where(mode == 2, torch.zeros_like(bbar), torch.rand(n, generator=g, device=L.DEVICE) * 0.9))
        Bs = torch.where((mode == 0).view(-1, 1, 1), B, sym(bs))
        h = L.sample_gains(n, U, g)
        snr = C["snr_train"][torch.randint(0, len(C["snr_train"]), (1,), generator=g, device=L.DEVICE).item()]
        t = L.transmit_receive(e, L.signatures(Bs), h, snr, g)
        out = m(t, h, bs=Bs) if use_bs else m(t, h)
        loss = L.cosine_loss(out, e)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 2000 == 0 or it == iters - 1:
            print(f"  {tag} it {it:5d}/{iters} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    m.eval(); torch.save(m.state_dict(), CKPT / f"{tag}.pt"); return m


def load(tag, use_bs):
    m = L.GramSC(U, L.CFG["hidden"], feat=L.CFG["feat"], use_h=L.CFG["use_h"], use_bs=use_bs)
    m.load_state_dict(torch.load(CKPT / f"{tag}.pt", map_location=L.DEVICE, weights_only=True))
    return m.to(L.DEVICE).eval()


# --------------------------------------------------------------- evaluation
SCHEMES = ["SC", "SClt", "SCltb", "PLG", "LMMSE", "SR", "MF", "OMA", "OMAL"]


@torch.no_grad()
def run_block(models, e, B, Bs, h, snr, g, tau=None, probe=None, y=None):
    tau = tau or L.CFG["tau"]
    n = e.shape[0]
    Bs = Bs if Bs.dim() == 3 else Bs.unsqueeze(0).expand(n, U, U)
    t = L.transmit_receive(e, L.signatures(Bs), h, snr, g)
    ri = L.noise_estimate(t, Bs, h)
    eh = {"SC": models["SC"](t, h), "SClt": models["SClt"](t, h), "SCltb": models["SCltb"](t, h, bs=Bs),
          "PLG": rx_plugin(t, Bs, h, ri), "LMMSE": L.rx_lmmse(t, Bs, h, snr, B)[0],
          "SR": L.rx_sr(t, Bs, h), "MF": L.rx_mf(t)}
    I = torch.eye(U, device=L.DEVICE).expand(n, U, U)
    to = L.transmit_receive(e, I, h, snr, g)
    eh["OMA"] = L.rx_mf(to); eh["OMAL"] = L.rx_lmmse(to, I, h, snr, B)[0]
    out = {}
    for k, v in eh.items():
        mt = L.metrics(v, e, tau)
        if probe is not None:
            mt["acc"] = (probe(v.reshape(-1, d)).argmax(1) == y.reshape(-1)).float().mean().item()
        out[k] = mt
    return out


def write(rows, name):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", OUT / name, flush=True)


def row_of(res, head):
    row = dict(head)
    for k in SCHEMES:
        for m in ["ser", "cos", "acc"]:
            if m in res[k]:
                row[f"{m}_{k}"] = res[k][m]
    row["n_decisions"] = res["SC"]["n"]
    return row


def fmt(res):
    return " ".join(f"{k}:{res[k]['ser']:.2e}" for k in SCHEMES)


def synth(models):
    g0 = L.gen(11)
    amix = L.sample_profile(100000, U, g0); Bm = L.affinity_matrix(amix)
    off = ~torch.eye(U, dtype=torch.bool, device=L.DEVICE)
    beta_mix = Bm[:, off].mean().item()
    print(f"mixed-profile long-term similarity beta_bar = {beta_mix:.3f}", flush=True)
    conds = {
        "matched": lambda n, g: (lambda a: (a, L.affinity_matrix(a)))(jitter(n, g, BETA_BAR, DELTA)),
        "lt": lambda n, g: (jitter(n, g, BETA_BAR, DELTA), sym(torch.full((n,), BETA_BAR, device=L.DEVICE))),
        "orth": lambda n, g: (jitter(n, g, BETA_BAR, DELTA), torch.eye(U, device=L.DEVICE).expand(n, U, U)),
        "ltmix": lambda n, g: (L.sample_profile(n, U, g), sym(torch.full((n,), beta_mix, device=L.DEVICE))),
    }
    t0 = time.time()
    for name, fn in conds.items():
        rows = []
        for snr in range(-6, 25, 2):
            g = L.gen(12000 + snr + 100 * list(conds).index(name)); N = NF if snr >= 4 else N0; acc = {}
            for _ in range(N // CHUNK0):
                a, Bs = fn(CHUNK0, g)
                e, _ = L.sample_embeddings(CHUNK0, a, d, g); B = L.affinity_matrix(a)
                if name == "matched":
                    Bs = B
                accumulate(acc, run_block(models, e, B, Bs, L.sample_gains(CHUNK0, U, g), float(snr), g))
            res = finalize(acc); rows.append(row_of(res, {"snr": snr}))
            print(f"[{name:7s} {snr:3d} dB] {fmt(res)} ({time.time()-t0:.0f}s)", flush=True)
        write(rows, f"synth_{name}.csv")


def real(models):
    import exp8_real as E8
    REP = int(os.environ.get("SEMA_REP", "100"))
    Ztr, Zte, ytr, yte, ncls = E8.load_corpus("ag_news", "bert-base-uncased")
    g = L.gen(9000); n_sup = 200 if QUICK else 4000; n_eval = 200 if QUICK else 2000
    grp_sup = E8.nn_groups(Ztr, n_sup, g); grp_eval = E8.nn_groups(Zte, n_eval, g)
    beta_bar = E8.group_stats(Ztr, grp_sup)["beta_mean"]
    Bbar = L.affinity_matrix(torch.full((U,), math.sqrt(max(beta_bar, 0.0)), device=L.DEVICE))
    probe = E8.train_probe(Ztr, ytr, ncls); yg = yte[grp_eval]
    print(f"AG News BERT: long-term beta_bar = {beta_bar:.3f}", flush=True)
    t0 = time.time()
    for name, Bs0 in [("lt", Bbar), ("orth", torch.eye(U, device=L.DEVICE))]:
        rows = []
        for snr in range(-4, 21, 2):
            gg = L.gen(9100 + snr); acc = {}; CH = min(2000, n_eval)
            for r in range(REP):
                for i in range(0, n_eval, CH):
                    E = Zte[grp_eval[i:i + CH]]; y = yg[i:i + CH]; n = len(E)
                    # genie prior: the long-term Gram (the per-group Gram is not part of the model)
                    accumulate(acc, run_block(models, E, Bbar.expand(n, U, U), Bs0, L.sample_gains(n, U, gg), float(snr), gg,
                                              probe=probe, y=y))
            res = finalize(acc); rows.append(row_of(res, {"snr": snr}))
            print(f"[real {name:4s} {snr:3d} dB] {fmt(res)} | acc SC {res['SC']['acc']:.3f} SClt {res['SClt']['acc']:.3f} "
                  f"PLG {res['PLG']['acc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        write(rows, f"real_{name}.csv")


def main(mode):
    iters = int(os.environ.get("SEMA_ITERS", str(L.CFG["iters"]))) if not QUICK else 60
    if mode == "train":
        train("gramsc_U4_d64_s1_lt", False, iters)
        train("gramsc_U4_d64_s1_ltb", True, iters)
        return
    models = dict(SC=load_sc(U), SClt=load("gramsc_U4_d64_s1_lt", False), SCltb=load("gramsc_U4_d64_s1_ltb", True))
    {"synth": synth, "real": real}[mode](models)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synth")
