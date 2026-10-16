"""Train every learned receiver used in the paper and store the
checkpoints in data/ckpt/. One run per line of RUNS; `python exp_train.py
<name>` trains one, no argument trains all that are missing.

  gramsc_U{U}_d64_s1      proposed SC, U in {2,3,4,5,6,8}, seed 1
  gramsc_U4_d64_s{2,3}    seed spread of the main configuration
  gramsc_U4_d64_s1_lam    decorrelation-penalty ablation (lam = 0.1)
  gramsc_U4_d64_s1_soft   softmax twin (convex weights)
  gramsc_U4_d256_s1       trained at d = 256 (scaling check)
  att_U4_d64_s1           projection-based attention receiver
  gramsc_U4_d64_s1_h      variant that also reads the front-end gains
  gramsc_U4_d64_s1_sgn    variant trained with signed share coefficients
Training: Adam, cosine-annealed lr, batch 256, 6,000 iterations, mixed
similarity profiles, SNR drawn from {0,2,...,24} dB, Rayleigh gains,
exact signature front end. Loss: mean cosine distortion.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sema_lib as L
from suite import CKPT, QUICK

CKPT.mkdir(parents=True, exist_ok=True)
C = L.CFG
RUNS = {}
for U in [2, 3, 4, 5, 6, 8]:
    RUNS[f"gramsc_U{U}_d64_s1"] = dict(kind="sc", U=U, d=64, seed=1)
for s in [2, 3]:
    RUNS[f"gramsc_U4_d64_s{s}"] = dict(kind="sc", U=4, d=64, seed=s)
RUNS["gramsc_U4_d64_s1_lam"] = dict(kind="sc", U=4, d=64, seed=1, lam=0.1)
RUNS["gramsc_U4_d64_s1_soft"] = dict(kind="sc", U=4, d=64, seed=1, softmax=True)
RUNS["gramsc_U4_d256_s1"] = dict(kind="sc", U=4, d=256, seed=1)
RUNS["att_U4_d64_s1"] = dict(kind="att", U=4, d=64, seed=1)
RUNS["gramsc_U4_d64_s1_h"] = dict(kind="sc", U=4, d=64, seed=1, use_h=True)          # gain-conditioned variant
RUNS["gramsc_U4_d64_s1_sgn"] = dict(kind="sc", U=4, d=64, seed=1, profile="mixed_signed")  # signed-coefficient training


def train_one(name):
    r = RUNS[name]
    U, d, seed = r["U"], r["d"], r["seed"]
    iters = 60 if QUICK else C["iters"]
    if r["kind"] == "sc":
        model = L.GramSC(U, C["hidden"], softmax=r.get("softmax", False),
                         feat=C["feat"], use_h=r.get("use_h", C["use_h"]))
    else:
        model = L.ProjAttention(U, d)
    t0 = time.time()
    print(f"== {name}: {L.n_params(model)} parameters, device {L.DEVICE}", flush=True)
    model = L.train_receiver(model, U, d, iters, C["batch"], C["lr"], seed,
                             lam=r.get("lam", 0.0), snr_set=C["snr_train"],
                             log=lambda m: print(m, flush=True), kind=C["kind"],
                             profile=r.get("profile", "mixed"))
    torch.save(model.state_dict(), CKPT / f"{name}.pt")
    print(f"   saved {name}.pt ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in RUNS if not (CKPT / f"{n}.pt").exists()]
    for n in names:
        train_one(n)
