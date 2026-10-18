"""Adds the SC-rho and TLC-g columns of the real-data variant runs to the base
AG News result file: all SNRs from real_*_snr_var.csv (REP = 100) and the
12-20 dB points from real_*_snr_varfloor.csv (REP = 1000).
python3 code/merge_real_variants.py"""
from __future__ import annotations
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
BASE = "real_ag_news_bert-base-uncased_snr.csv"
COLS = [f"{k}_{m}" for m in ("SCrho", "TLCg") for k in ("ser", "cos", "acc")]


def load(name):
    with open(DATA / name) as f:
        r = csv.DictReader(f); return list(r), r.fieldnames


def main():
    rows, fields = load(BASE)
    var, _ = load(BASE.replace(".csv", "_var.csv"))
    floor = {}
    if (DATA / BASE.replace(".csv", "_varfloor.csv")).exists():
        fl, _ = load(BASE.replace(".csv", "_varfloor.csv")); floor = {float(r["snr"]): r for r in fl}
    src = {float(r["snr"]): r for r in var}
    for c in COLS:
        if c not in fields: fields.append(c)
    for r in rows:
        k = float(r["snr"]); s = floor.get(k, src.get(k))
        if s is None: continue
        for c in COLS: r[c] = s[c]
    with open(DATA / BASE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("merged", len(rows), "rows; floor points:", sorted(floor))


if __name__ == "__main__":
    main()
