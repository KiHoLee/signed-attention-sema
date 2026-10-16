"""Prints every quantity that Section VI quotes, computed from data/, so the
text can be checked against the final run (writing standard 12.7, 12.10).
python3 code/check_claims.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"


def rows(name):
    with open(DATA / name) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def col(rs, k):
    return np.array([r[k] for r in rs])


def first_cross(x, a, b, cond):
    """first x where cond(a,b) fails (a<=b) -> crossover"""
    for xi, ai, bi in zip(x, a, b):
        if not cond(ai, bi):
            return xi
    return None


print("=== synthetic SNR sweep beta 0.64 (claims in Sec. VI-D)")
rs = rows("snr_sweep_b0.64.csv"); snr = col(rs, "snr")
for k in ["SC", "LMMSE", "MOM", "SCsoft", "ATT", "TLC", "MF", "OMA", "OMAL"]:
    print(f"  {k:7s} ser " + " ".join(f"{v:.1e}" for v in col(rs, 'ser_' + k)[::2]))
dsc = 1 - col(rs, "cos_SC"); dl = 1 - col(rs, "cos_LMMSE")
print("  distortion gap SC-LMMSE:", " ".join(f"{s:.0f}:{g:.4f}" for s, g in zip(snr, dsc - dl)))
print("  SC below OMA in SER up to:", first_cross(snr, col(rs, "ser_SC"), col(rs, "ser_OMA"), lambda a, b: a <= b), "dB")
print("  SC below OMA in distortion up to:", first_cross(snr, dsc, 1 - col(rs, "cos_OMA"), lambda a, b: a <= b), "dB")
print("  softmax distortion from 8 dB:", " ".join(f"{1-v:.3f}" for v in col(rs, 'cos_SCsoft')[7:]))
print("  TLC SER at 22-24 dB:", col(rs, "ser_TLC")[-2:])
print("  MOM/LMMSE SER ratio below 6 dB:", [f"{m/l if l > 0 else np.inf:.1f}" for m, l in zip(col(rs, 'ser_MOM')[:7], col(rs, 'ser_LMMSE')[:7])])
print("  SC offdiag:", " ".join(f"{s:.0f}:{w:+.2f}" for s, w in zip(snr, col(rs, 'SC_offdiag_mean'))))
for b in [0.3, 0.9]:
    r2 = rows(f"snr_sweep_b{b}.csv"); s2 = col(r2, "snr")
    print(f"  beta {b}: SC below OMA in SER up to", first_cross(s2, col(r2, "ser_SC"), col(r2, "ser_OMA"), lambda a, b_: a <= b_), "dB; distortion up to",
          first_cross(s2, 1 - col(r2, "cos_SC"), 1 - col(r2, "cos_OMA"), lambda a, b_: a <= b_), "dB")

print("=== beta sweep 0 dB")
rb = rows("beta_sweep_0dB.csv"); beta = col(rb, "beta")
for k in ["SC", "LMMSE", "SR", "OMA", "OMAL", "MF"]:
    print(f"  {k:6s} " + " ".join(f"{b:.1f}:{v:.1e}" for b, v in zip(beta, col(rb, 'ser_' + k)) if round(b * 20) % 2 == 0))
i3, i8 = int(np.argmin(abs(beta - 0.3))), int(np.argmin(abs(beta - 0.8)))
print("  SC ser 0.3 -> 0.8:", col(rb, "ser_SC")[i3], col(rb, "ser_SC")[i8], " LMMSE:", col(rb, "ser_LMMSE")[i3], col(rb, "ser_LMMSE")[i8])
print("  OMAL/SC ratio for beta>0.4:", [f"{o/s if s > 0 else np.inf:.0f}" for b, o, s in zip(beta, col(rb, 'ser_OMAL'), col(rb, 'ser_SC')) if b > 0.4])

print("=== seeds / penalty")
try:
    with open(DATA / "seeds.csv") as f:
        sd = list(csv.DictReader(f))
    for data in ["synthetic", "agnews"]:
        for snr_ in sorted(set(float(r["snr"]) for r in sd if r["data"] == data)):
            sub = {r["model"]: r for r in sd if r["data"] == data and float(r["snr"]) == snr_}
            cosv = [float(sub[m]["cos"]) for m in ["seed1", "seed2", "seed3"]]
            serv = [float(sub[m]["ser"]) for m in ["seed1", "seed2", "seed3"]]
            print(f"  {data:9s} {snr_:4.0f} dB cos spread {max(cosv)-min(cosv):.4f} ser {min(serv):.1e}-{max(serv):.1e} | lam0.1 cos {float(sub['lam0.1']['cos']):.4f} (seed1 {cosv[0]:.4f}) ser {float(sub['lam0.1']['ser']):.1e}")
except Exception as e:
    print("  seeds:", e)

print("=== mismatch")
for r in rows("mismatch.csv"):
    if r["snr"] == 0.0 and abs(abs(r["delta"]) - 0.2) < 1e-6:
        print(f"  delta {r['delta']:+.1f}: SR {r['ser_SR']:.3f} LMMSE {r['ser_LMMSE']:.1e} SC {r['ser_SC']:.1e}")

print("=== scaling U (cos)")
for r in rows("scaling_U.csv"):
    print(f"  U={r['U']:.0f} {r['snr']:.0f}dB SC {r['cos_SC']:.3f} LMMSE {r['cos_LMMSE']:.3f} MF {r['cos_MF']:.3f} OMA {r['cos_OMA']:.3f} SR {r['cos_SR']:.3f} | ser SC {r['ser_SC']:.1e} LMMSE {r['ser_LMMSE']:.1e}")
print("=== scaling d")
for r in rows("scaling_d.csv"):
    print(f"  d={r['d']:.0f} {r['snr']:.0f}dB SC64 {r['ser_SC']:.1e}/{r['cos_SC']:.3f} SC256 {r['ser_SC256']:.1e}/{r['cos_SC256']:.3f} LMMSE {r['ser_LMMSE']:.1e}/{r['cos_LMMSE']:.3f} MOM {r['ser_MOM']:.1e}/{r['cos_MOM']:.3f} MF {r['ser_MF']:.1e}/{r['cos_MF']:.3f}")

print("=== profiles")
P = json.load(open(DATA / "profiles.json"))
for k, v in P.items():
    print(f"  {k:14s} " + " ".join(f"{m}:{v[m]['ser']:.1e}/{v[m]['cos']:.3f}" for m in ["SC", "SCh", "SCsgn", "LMMSE", "MOM", "MF", "OMA", "TLC"] if m in v))
V = json.load(open(DATA / "varmix.json"))
for k in ["0dB", "10dB"]:
    v = V[k]; print(f"  varmix {k}: " + " ".join(f"{m}:{v[m]['ser']:.1e}" for m in ["SC", "SCsoft", "TLC", "MF", "LMMSE", "LMMSEg", "OMA", "OMAL"]))

print("=== real AG News")
rr = rows("real_ag_news_bert-base-uncased_snr.csv"); s3 = col(rr, "snr")
for k in ["SC", "SCsoft", "ATT", "TLC", "LMMSE", "LMMSEe", "MOM", "SR", "MF", "OMA"]:
    print(f"  {k:7s} ser " + " ".join(f"{s:.0f}:{v:.1e}" for s, v in zip(s3, col(rr, 'ser_' + k))))
print("  SC/LMMSE ser ratio:", [f"{s:.0f}:{a/b if b > 0 else np.inf:.1f}" for s, a, b in zip(s3, col(rr, 'ser_SC'), col(rr, 'ser_LMMSE'))])
print("  SC/LMMSEe ser ratio:", [f"{s:.0f}:{a/b if b > 0 else np.inf:.1f}" for s, a, b in zip(s3, col(rr, 'ser_SC'), col(rr, 'ser_LMMSEe'))])
print("  OMA below SC from:", first_cross(s3, col(rr, "ser_SC"), col(rr, "ser_OMA"), lambda a, b: a <= b), "dB")
S = json.load(open(DATA / "real_stats.json"))["ag_news_bert-base-uncased"]
print("  clean acc eval groups:", S["probe_acc_clean_eval_groups"])
for k in ["SC", "LMMSE", "LMMSEe", "OMA", "SR", "MF"]:
    a = col(rr, "acc_" + k); print(f"  acc {k:6s} 0dB {a[2]:.3f} 10dB {a[7]:.3f}")
print("=== tau sweep ordering (SC vs MF vs LMMSE vs OMA)")
for r in rows("real_tau.csv"):
    print(f"  {r['snr']:.0f}dB tau {r['tau']:.2f}: SC {r['ser_SC']:.1e} LMMSE {r['ser_LMMSE']:.1e} MF {r['ser_MF']:.1e} OMA {r['ser_OMA']:.1e}")
print("=== complexity", json.load(open(DATA / "complexity.json")))
