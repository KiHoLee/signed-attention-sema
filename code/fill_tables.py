"""Generates tables_gen.tex (input by main.tex) from the result files in
data/, so that no number in a table is typed by hand. Run after the
experiment queue: python3 code/fill_tables.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "tables_gen.tex"

LAB = {"SC": "SC", "SCh": "SC-g", "SCsgn": "SC-s", "SCsoft": "SC-soft", "ATT": "Attention", "TLC": "TLC", "LMMSE": "LMMSE",
       "LMMSEe": "LMMSE-emp", "LMMSEg": "Genie LMMSE", "MOM": "Moment LMMSE", "SR": "SR", "MF": "MF",
       "OMA": "OMA", "OMAL": "OMA + LMMSE"}


def rows(name):
    with open(DATA / name) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def g(r, c, k):
    """entry r[c][k] or r[f"{k}_{c}"], or None when the scheme is absent."""
    if isinstance(r.get(c), dict):
        return r[c].get(k)
    return r.get(f"{k}_{c}")


def ser(v):
    """SER cell: three decimals above 1e-3, otherwise scientific with one digit;
    exact zero over the run is printed as 0 (the text states the decision count)."""
    if v is None:
        return "--"
    if v == 0:
        return "$0$"
    if v >= 1e-3:
        return f"${v:.3f}$"
    m, e = f"{v:.1e}".split("e")
    return f"${m}\\!\\times\\!10^{{{int(e)}}}$"


def ser4(v):
    """SER scaled by 1e4 with one decimal (for the dimension block)."""
    return "--" if v is None else f"${v * 1e4:.1f}$"


def cs(v):
    return "--" if v is None else f"${v:.3f}$"


def main():
    L = []
    # ------------------------------------------------ profiles / OOD (one row per profile)
    P = json.load(open(DATA / "profiles.json"))
    V = json.load(open(DATA / "varmix.json"))
    c0 = ["SC", "SCh", "SCsgn", "TLC", "LMMSE", "LMMSEg", "MOM", "MF", "OMA"]
    c1 = ["SC", "TLC", "LMMSE", "MF", "OMA"]
    names = [("sym", "Symmetric"), ("asym", "Asymmetric"), ("nearfar", "Near-far"),
             ("twoclust", "Two scenes"), ("negcorr", "Negative correlation")]
    L.append("% ---- Table: profiles (generated) ----")
    L.append("\\begin{table*}[!t]\n\\centering\n\\caption{SER at $0$~dB and Cosine Recovery at $10$~dB Under Heterogeneous, Out-of-Distribution, and Block-Varying Similarity Profiles ($U=4$, $d=64$). The Block-Varying Rows Report Both Metrics at the Stated SNR}\n\\label{tab:profiles}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.2pt}\n\\renewcommand{\\arraystretch}{1.0}")
    L.append("\\begin{tabular}{l" + "c" * (len(c0) + len(c1)) + "}\n\\toprule")
    L.append(" & \\multicolumn{%d}{c}{SER $(\\times10^{-4})$ at $0$~dB} & \\multicolumn{%d}{c}{Cosine recovery at $10$~dB} \\\\" % (len(c0), len(c1)))
    L.append("\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}" % (len(c0) + 1, len(c0) + 2, len(c0) + len(c1) + 1))
    L.append("Profile & " + " & ".join(LAB[c] for c in c0) + " & " + " & ".join(LAB[c] for c in c1) + " \\\\\n\\midrule")
    for k, nm in names:
        r0, r1 = P[f"{k}_0dB"], P[f"{k}_10dB"]
        L.append(nm + " & " + " & ".join(ser4(g(r0, c, "ser")) for c in c0) + " & " + " & ".join(cs(g(r1, c, "cos")) for c in c1) + " \\\\")
    for snr in ["0dB", "10dB"]:
        L.append(f"Block varying, {snr[:-2]}~dB & " + " & ".join(ser4(g(V[snr], c, "ser")) for c in c0) + " & " + " & ".join(cs(g(V[snr], c, "cos")) for c in c1) + " \\\\")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    # ------------------------------------------------ scaling in U and d (one table)
    D = rows("scaling_d.csv"); Ur = rows("scaling_U.csv")
    L.append("% ---- Table: scaling (generated) ----")
    L.append("\\begin{table}[!t]\n\\centering\n\\caption{Cosine Recovery Versus User Count ($d=64$) and SER and Cosine Recovery Versus Embedding Dimension ($U=4$) at $\\beta=0.64$, With SC$_{64}$ and SC$_{256}$ Trained at $d=64$ and $d=256$ and Applied Unchanged}\n\\label{tab:dim}\n\\scriptsize\n\\setlength{\\tabcolsep}{0.9pt}\n\\renewcommand{\\arraystretch}{1.0}")
    cols = ["SC", "LMMSE", "MOM", "SR", "MF", "OMA"]
    L.append("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule")
    L.append(" & " + " & ".join(LAB[c] for c in cols) + " \\\\\n\\midrule")
    for snr in [0.0, 10.0]:
        L.append("\\multicolumn{%d}{l}{\\emph{Cosine recovery versus $U$ at $%d$~dB}} \\\\" % (len(cols) + 1, int(snr)))
        for r in [x for x in Ur if x["snr"] == snr and int(x["U"]) in (2, 4, 6, 8)]:
            L.append(f"$U={int(r['U'])}$ & " + " & ".join(cs(g(r, c, "cos")) for c in cols) + " \\\\")
    cols2 = ["SC", "SC256", "LMMSE", "MOM", "MF", "OMA"]
    lab = {"SC": "SC$_{64}$", "SC256": "SC$_{256}$"}
    L.append("\\midrule\n" + " & " + " & ".join(lab.get(c, LAB.get(c, c)) for c in cols2) + " \\\\")
    L.append("\\multicolumn{%d}{l}{\\emph{SER $(\\times10^{-4})$ at $0$~dB versus $d$}} \\\\" % (len(cols) + 1))
    for r in [x for x in D if x["snr"] == 0.0]:
        L.append(f"$d={int(r['d'])}$ & " + " & ".join(ser4(g(r, c, "ser")) for c in cols2) + " \\\\")
    L.append("\\multicolumn{%d}{l}{\\emph{Cosine recovery at $10$~dB versus $d$}} \\\\" % (len(cols) + 1))
    for r in [x for x in D if x["snr"] == 10.0]:
        L.append(f"$d={int(r['d'])}$ & " + " & ".join(cs(g(r, c, "cos")) for c in cols2) + " \\\\")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    # ------------------------------------------------ real corpora
    S = json.load(open(DATA / "real_stats.json"))
    corp = [("ag_news_bert-base-uncased", "AG News, BERT"), ("sst2_bert-base-uncased", "SST-2, BERT"),
            ("ag_news_distilbert-base-uncased", "AG News, DistilBERT")]
    cols = ["SC", "LMMSE", "LMMSEe", "MOM", "SR", "MF", "OMA"]
    L.append("% ---- Table: real corpora (generated) ----")
    L.append("\\begin{table*}[!t]\n\\centering\n\\caption{SER and Classification Accuracy at $0$ and $10$~dB on Real Embedding Groups With Their Group Statistics ($U=4$, $d=64$)}\n\\label{tab:real}\n\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.0}")
    L.append("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule")
    L.append(" & " + " & ".join(LAB[c] for c in cols) + " \\\\")
    for key, nm in corp:
        st = S[key]["nn_eval"]
        R = {int(r["snr"]): r for r in rows(f"real_{key}_snr.csv")}
        L.append("\\midrule\n\\multicolumn{%d}{l}{\\emph{%s}: $\\bar\\beta=%.2f\\pm%.2f$, leading direction $%.0f$ percent, clean accuracy $%.3f$} \\\\"
                 % (len(cols) + 1, nm, st["beta_mean"], st["beta_std"], 100 * st["rank1_mean"], S[key]["probe_acc_clean_eval_groups"]))
        for snr in [0, 10]:
            L.append(f"SER, {snr}~dB & " + " & ".join(ser(g(R[snr], c, "ser")) for c in cols) + " \\\\")
        for snr in ([0, 10] if key.startswith("ag_news_bert") else ([10] if key.startswith("sst2") else [])):
            L.append(f"accuracy, {snr}~dB & " + " & ".join(cs(g(R[snr], c, "acc")) for c in cols) + " \\\\")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
