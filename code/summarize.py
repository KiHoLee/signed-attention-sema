"""Prints the key numbers of every result file in data/ in a compact form
for writing the evaluation section (no figure, no computation beyond
ratios). Run after the queue: python3 code/summarize.py
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
ORDER = ["SC", "SCsoft", "ATT", "TLC", "LMMSE", "MOM", "LMMSEe", "SR", "MF", "OMA", "OMAL", "LMMSEg"]


def rows(name):
    p = DATA / name
    if not p.exists():
        print(f"[missing] {name}"); return []
    with open(p) as f:
        return [{k: (float(v) if v not in ("",) else np.nan) for k, v in r.items()} for r in csv.DictReader(f)]


def fmt(v):
    return f"{v:.1e}" if v < 1e-2 else f"{v:.3f}"


def table(rs, xkey, metric="ser", keys=None, every=1):
    if not rs:
        return
    keys = keys or [m for m in ORDER if f"{metric}_{m}" in rs[0]]
    print(f"{xkey:>6s} | " + " ".join(f"{m:>9s}" for m in keys))
    for r in rs[::every]:
        print(f"{r[xkey]:6.2f} | " + " ".join(f"{fmt(r[f'{metric}_{m}']):>9s}" for m in keys))


def main():
    for snr in [0, 10]:
        rs = rows(f"beta_sweep_{snr}dB.csv")
        print(f"\n=== beta sweep {snr} dB: SER"); table(rs, "beta", "ser", every=2)
        print(f"=== beta sweep {snr} dB: cosine"); table(rs, "beta", "cos", every=3)
    for b in [0.3, 0.64, 0.9]:
        rs = rows(f"snr_sweep_b{b}.csv")
        print(f"\n=== SNR sweep beta {b}: SER"); table(rs, "snr", "ser", every=2)
        print(f"=== SNR sweep beta {b}: cosine"); table(rs, "snr", "cos", every=2)
        if rs:
            print("    SC offdiag:", " ".join(f"{r['snr']:.0f}:{r['SC_offdiag_mean']:+.3f}" for r in rs[::2]))
            print("    LMMSE offd:", " ".join(f"{r['snr']:.0f}:{r['LMMSE_offdiag_mean']:+.3f}" for r in rs[::2]))
    p = DATA / "profiles.json"
    if p.exists():
        print("\n=== profiles (SER / cos)")
        for k, v in json.load(open(p)).items():
            print(f"{k:14s} " + " ".join(f"{m}:{fmt(v[m]['ser'])}/{v[m]['cos']:.3f}" for m in ORDER if m in v))
    p = DATA / "varmix.json"
    if p.exists():
        d = json.load(open(p)); print("\n=== block-varying (SER / cos), Bbar offdiag", d.get("Bbar_offdiag"))
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{k:6s} " + " ".join(f"{m}:{fmt(v[m]['ser'])}/{v[m]['cos']:.3f}" for m in ORDER if m in v))
    rs = rows("scaling_U.csv")
    if rs:
        print("\n=== scaling U (SER)")
        for r in rs: print(f"U={r['U']:.0f} {r['snr']:.0f}dB " + " ".join(f"{m}:{fmt(r['ser_'+m])}/{r['cos_'+m]:.3f}" for m in ORDER if f"ser_{m}" in r))
    rs = rows("scaling_d.csv")
    if rs:
        print("\n=== scaling d (SER)")
        for r in rs: print(f"d={r['d']:.0f} {r['snr']:.0f}dB " + " ".join(f"{m}:{fmt(r['ser_'+m])}/{r['cos_'+m]:.3f}" for m in ORDER if f"ser_{m}" in r) + f" SC256:{fmt(r['ser_SC256'])}/{r['cos_SC256']:.3f}")
    for snr in [0, 10]:
        rs = rows(f"sigdesign_{snr}dB.csv")
        if rs:
            print(f"\n=== signature design {snr} dB");
            for r in rs: print(f"bs={r['beta_s']:.2f} mseL {r['mse_cf_lmmse']:.4f} mseSR {r['mse_cf_sr']:.4f} " + " ".join(f"{m}:{fmt(r['ser_'+m])}/{r['cos_'+m]:.3f}" for m in ["SC","LMMSE","MOM","SR","MF"] if f"ser_{m}" in r))
    for name in ["real_ag_news_bert-base-uncased_snr.csv", "real_sst2_bert-base-uncased_snr.csv", "real_ag_news_distilbert-base-uncased_snr.csv", "real_indep.csv"]:
        rs = rows(name)
        if rs:
            print(f"\n=== {name}: SER"); table(rs, "snr", "ser")
            print(f"=== {name}: cosine"); table(rs, "snr", "cos")
            print(f"=== {name}: accuracy"); table(rs, "snr", "acc")
    rs = rows("real_tau.csv")
    if rs:
        print("\n=== tau sweep (real)")
        for r in rs: print(f"{r['snr']:.0f}dB tau {r['tau']:.2f} " + " ".join(f"{m}:{fmt(r['ser_'+m])}" for m in ORDER if f"ser_{m}" in r))
    p = DATA / "real_stats.json"
    if p.exists():
        print("\n=== real stats"); print(json.dumps(json.load(open(p)), indent=0)[:1500])
    rs = rows("mismatch.csv")
    if rs:
        print("\n=== mismatch");
        for r in rs: print(f"{r['snr']:.0f}dB d={r['delta']:+.2f} SR {fmt(r['ser_SR'])} LMMSE {fmt(r['ser_LMMSE'])} SC {fmt(r['ser_SC'])}")
    p = DATA / "complexity.json"
    if p.exists():
        print("\n=== complexity", json.load(open(p)))


if __name__ == "__main__":
    main()
