#!/usr/bin/env bash
# Builds the submission package (sources + figures + PDF) and the Overleaf
# package. Run after `bash code/gate.sh` passes.
set -e
cd "$(dirname "$0")/.."
STAMP=$(date +%Y%m%d)
rm -rf pkg && mkdir -p pkg/TSP_submission
cp main.tex tables_gen.tex references.bib IEEEtran.cls fig1_system.tex pkg/TSP_submission/
mkdir -p pkg/TSP_submission/fig && cp fig/fig1_system.pdf fig/fig_analysis.pdf fig/fig_synth.pdf fig/fig_real.pdf pkg/TSP_submission/fig/
cp out/main.pdf pkg/TSP_submission/main.pdf
cp out/main.bbl pkg/TSP_submission/main.bbl
( cd pkg && zip -qr "TSP_SignedAttention_submission_$STAMP.zip" TSP_submission )
# standalone build check of the package
( cd pkg/TSP_submission && rm -f main.pdf && mkdir -p out && export TEXINPUTS=.:$TEXINPUTS BIBINPUTS=.:$BIBINPUTS BSTINPUTS=.:$BSTINPUTS \
  && pdflatex -interaction=nonstopmode -output-directory=out main.tex >/dev/null && bibtex out/main >/dev/null && pdflatex -interaction=nonstopmode -output-directory=out main.tex >/dev/null && pdflatex -interaction=nonstopmode -output-directory=out main.tex >/dev/null \
  && python3 -c "import fitz;print('package builds:',len(fitz.open('out/main.pdf')),'pages')" )
ls -la pkg/*.zip
