"""E3: heterogeneous and out-of-distribution similarity structures at 0 and
10 dB (U = 4, d = 64). Profiles:
  sym        a_u = 0.8 (beta = 0.64)
  asym       a = (0.9, 0.75, 0.55, 0.3)
  nearfar    a_u = 0.8, mean-gain offsets (9, 6, 3, 0) dB
  both       asymmetric a and offsets (0, 3, 6, 9) dB
  indep      a_u = 0 (orthogonal embeddings)
  twoclust   two scenes, users {1,2} and {3,4} each share one (OOD)
  negcorr    a = (0.8, 0.8, -0.8, -0.8): negative cross-cluster similarity (OOD)
The masks (signatures) use the true Gram of the profile (two-cluster and
negative-correlation Grams are PSD, so the Cholesky construction exists).
Output: data/profiles.json
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import *

U, d = 4, 64
N = n_blocks(200_000); CHUNK = min(25_000, N)
A_HI = 0.8


def sample(profile, n, g):
    """Returns e, s(or None), Bsig(U,U), h."""
    if profile == "twoclust":
        e = L.sample_two_cluster(n, U, d, A_HI, g)
        B = L.gram_of(e[:1])[0]                   # exact block Gram (same for all)
        return e, None, B, L.sample_gains(n, U, g)
    a = {"sym": [A_HI] * 4, "asym": [0.9, 0.75, 0.55, 0.3], "nearfar": [A_HI] * 4,
         "both": [0.9, 0.75, 0.55, 0.3], "indep": [0.0] * 4,
         "negcorr": [A_HI, A_HI, -A_HI, -A_HI]}[profile]
    a = torch.tensor(a, device=L.DEVICE).expand(n, U)
    e, s = L.sample_embeddings(n, a, d, g)
    B = L.affinity_matrix(a[0])
    off = {"nearfar": [9, 6, 3, 0], "both": [0, 3, 6, 9]}.get(profile)
    h = L.sample_gains(n, U, g, offsets_db=off)
    return e, s, B, h


def main():
    models = dict(SC=load_sc(U), SCsoft=load_sc(U, softmax=True), ATT=load_att(U))
    sc_h = load_sc(U, tag="gramsc_U4_d64_s1_h", use_h=True)
    sc_sgn = load_sc(U, tag="gramsc_U4_d64_s1_sgn")
    table = {}
    t0 = time.time()
    for snr in [0.0, 10.0]:
        def data_fn(batch, gg, snr=snr):
            ap = L.sample_profile(batch, U, gg)
            e, _ = L.sample_embeddings(batch, ap, d, gg)
            return e, L.affinity_matrix(ap), L.sample_gains(batch, U, gg), snr
        models["TLC"] = train_tlc(U, d, data_fn)
        for prof in ["sym", "asym", "nearfar", "both", "indep", "twoclust", "negcorr"]:
            g = L.gen(3000 + int(snr) + hash(prof) % 100)
            acc = {}
            for _ in range(N // CHUNK):
                e, s, B, h = sample(prof, CHUNK, g)
                out, extra = run_suite(e, B, B, h, snr, g, models, s=s)
                t = L.transmit_receive(e, L.signatures(B), h, snr, g)
                with torch.no_grad():
                    out["SCh"] = L.metrics(sc_h(t, h), e, L.CFG["tau"], s)
                    out["SCsgn"] = L.metrics(sc_sgn(t, h), e, L.CFG["tau"], s)
                accumulate(acc, out, extra)
            res = finalize(acc)
            table[f"{prof}_{snr:.0f}dB"] = res
            print(f"[{snr:.0f} dB] {prof:9s} " + " ".join(f"{m}:{res[m]['ser']:.2e}" for m in ORDER + ["SCh", "SCsgn"])
                  + f" ({time.time()-t0:.0f}s)", flush=True)
    json.dump(table, open(DATA / "profiles.json", "w"), indent=1)


if __name__ == "__main__":
    main()
