"""Replaces the low-SER points of the base result files by the ten-times-larger
reruns of code/run_floor.sh (same seeds, same receivers), keeping the rerun
files in data/ for provenance. python3 code/merge_floor.py
"""
from __future__ import annotations
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
PAIRS = [("snr_sweep_b0.64.csv", "snr_sweep_b0.64_floor.csv", "snr"),
         ("beta_sweep_0dB.csv", "beta_sweep_0dB_floor.csv", "beta"),
         ("real_ag_news_bert-base-uncased_snr.csv", "real_ag_news_bert-base-uncased_snr_floor.csv", "snr")]


def load(name):
    with open(DATA / name) as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames


def main():
    for base, floor, key in PAIRS:
        if not (DATA / floor).exists():
            print("missing", floor); continue
        rows, fields = load(base); new, _ = load(floor)
        repl = {float(r[key]): r for r in new}
        n = 0
        for i, r in enumerate(rows):
            k = float(r[key])
            if k in repl:
                rows[i] = {c: repl[k].get(c, r[c]) for c in fields}; n += 1
        with open(DATA / base, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        print(f"{base}: replaced {n} rows from {floor}")


if __name__ == "__main__":
    main()
