"""E4: block-varying correlation. In every block a random pair of the four
users shares a scene (a = 0.9) and the other two are nearly independent
(a = 0.1). The signatures and the side-information receivers only know
the long-term average Gram; LMMSE-genie knows the pair of every block.
SNR 0 and 10 dB. Output: data/varmix.json
"""
from __future__ import annotations
import itertools, json, math, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *

U, d = 4, 64
N = n_blocks(200_000); CHUNK = min(25_000, N)
A_HI, A_LO = 0.9, 0.1
PAIRS = list(itertools.combinations(range(U), 2))


def sample_varmix(n, g):
    a = torch.full((n, U), A_LO, device=L.DEVICE)
    pick = torch.randint(0, len(PAIRS), (n,), generator=g, device=L.DEVICE)
    P = torch.tensor(PAIRS, device=L.DEVICE)[pick]          # (n,2)
    a.scatter_(1, P, A_HI)
    e, s = L.sample_embeddings(n, a, d, g)
    return e, s, a


def avg_B():
    Bs = torch.zeros(U, U, device=L.DEVICE)
    for i, j in PAIRS:
        a = torch.full((U,), A_LO, device=L.DEVICE); a[i] = a[j] = A_HI
        Bs += L.affinity_matrix(a)
    return Bs / len(PAIRS)


def main():
    models = dict(SC=load_sc(U), SCsoft=load_sc(U, softmax=True), ATT=load_att(U))
    Bbar = avg_B()
    table = {}
    t0 = time.time()
    for snr in [0.0, 10.0]:
        def data_fn(batch, gg, snr=snr):
            e, _, _ = sample_varmix(batch, gg)
            return e, Bbar, L.sample_gains(batch, U, gg), snr
        models["TLC"] = train_tlc(U, d, data_fn)
        g = L.gen(4000 + int(snr))
        acc, acc_g = {}, {}
        for _ in range(N // CHUNK):
            e, s, a = sample_varmix(CHUNK, g)
            h = L.sample_gains(CHUNK, U, g)
            out, extra = run_suite(e, Bbar, Bbar, h, snr, g, models, s=s)
            accumulate(acc, out, extra)
            # genie LMMSE: per-block true embedding Gram as prior, same signatures
            t = L.transmit_receive(e, L.signatures(Bbar), h, snr, g)
            eh, _ = L.rx_lmmse(t, Bbar.expand(CHUNK, U, U), h, snr, L.affinity_matrix(a))
            accumulate(acc_g, {"LMMSEg": L.metrics(eh, e, L.CFG["tau"], s)})
        res = finalize(acc); res.update(finalize(acc_g))
        table[f"{snr:.0f}dB"] = res
        print(f"[{snr:.0f} dB] " + " ".join(f"{m}:{res[m]['ser']:.2e}" for m in ORDER + ["LMMSEg"])
              + f" ({time.time()-t0:.0f}s)", flush=True)
    table["Bbar_offdiag"] = Bbar[0, 1].item()
    json.dump(table, open(DATA / "varmix.json", "w"), indent=1)


if __name__ == "__main__":
    main()
