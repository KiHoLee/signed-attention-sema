#!/usr/bin/env bash
# Deterministic build chain (pdflatex, bibtex, pdflatex x2) into out/.
cd "$(dirname "$0")/.."
rm -rf out && mkdir out
export TEXINPUTS=.:$TEXINPUTS BIBINPUTS=.:$BIBINPUTS BSTINPUTS=.:$BSTINPUTS
P="pdflatex -interaction=nonstopmode -file-line-error -output-directory=out main.tex"
$P >/dev/null 2>&1; bibtex out/main >/dev/null 2>&1; $P >/dev/null 2>&1; $P >/dev/null 2>&1
grep -E "^!|^\./[a-z_]+\.tex:[0-9]+:" out/main.log | head -5
grep -E "undefined" out/main.log | head -3
grep "Overfull" out/main.log
python3 -c "import fitz;d=fitz.open('out/main.pdf');print('pages',len(d)); print([len(p.get_text().split()) for p in d])"
