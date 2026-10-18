# Signed user-wise attention for semantic multiple access

Simulation code, result data, and figures for

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Embedding Multiplexing Gain from
> Semantic Similarity via Signed User-Wise Attention," submitted to
> *IEEE Transactions on Signal Processing*, 2026.

## Contents

- `code/sema_lib.py` shared library: shared-scene embeddings, the exact
  signature-spread front end (literal transmit, superpose, noise, despread),
  closed-form receivers (SR, LMMSE, moment-matched LMMSE, orthogonal
  references), learned receivers (proposed signed user-wise attention,
  softmax twin, projection attention, trainable linear combiner), losses,
  metrics, and the closed-form MSE expressions of the paper.
- `code/exp*.py` one script per experiment; `code/replot.py` regenerates
  every figure from `data/`; `code/fill_tables.py` regenerates every
  table; `code/summarize.py` prints the key numbers.
- `data/` raw results (CSV/JSON) behind every figure and table, and the
  trained parameters in `data/ckpt/` (GPU training is not bit-reproducible,
  so the released checkpoints are the reported models).
- `fig/` the figure PDFs of the paper.

## Figure and table map

| Paper item | Script | Data | Figure file |
|---|---|---|---|
| Fig. 2 closed forms vs Monte Carlo | `exp0_theory.py` | `theory_snr.csv`, `theory_beta.csv`, `theory_asym.csv` | `fig/fig_theory.pdf` |
| Fig. 3 LMMSE and SC weights | `exp2_snr.py` | `snr_sweep_b{0.3,0.64,0.9}.csv` | `fig/fig_weights.pdf` |
| Fig. 4 signature design | `exp6_sigdesign.py` | `sigopt.csv`, `sigdesign_{0,10}dB.csv` | `fig/fig_sigdesign.pdf` |
| Fig. 5 SER vs SNR, Fig. 6 cosine distortion vs SNR | `exp2_snr.py` | `snr_sweep_b0.64.csv` | `fig/fig_snr_ser.pdf`, `fig/fig_snr_cos.pdf` |
| Fig. 7 SER vs similarity | `exp1_beta.py` | `beta_sweep_0dB.csv` | `fig/fig_beta_0dB_ser.pdf` |
| Fig. 8, Table V real embeddings | `exp8_real.py` | `real_*.csv`, `real_stats.json` | `fig/fig_real.pdf` |
| Table III profiles and block-varying rows | `exp3_profiles.py`, `exp4_varmix.py` | `profiles.json`, `varmix.json` | |
| Table IV user count and dimension | `exp5_scaling.py` | `scaling_U.csv`, `scaling_d.csv` | |
| Text: Haar-mask floor, mismatch, complexity | `exp7_misc.py` | `haar_masks.csv`, `mismatch.csv`, `complexity.json` | |
| Text: seed spread, decorrelation penalty | `exp9_seeds.py` | `seeds.csv` | |
| Floor points of Figs. 5, 7, 8 at 10x blocks (merged into the base files) | `run_floor.sh`, `merge_floor.py` | `*_floor.csv` | |
| SC-rho and TLC-g variants (overlays in Figs. 5-7) | `exp_train.py gramsc_U4_d64_s1_rho`, `exp_floor_variants.py snr` / `beta` | `variants_floor/*.csv`, `ckpt/gramsc_U4_d64_s1_rho.pt` | |
| Text: receiver design study | `exp_variants.py` | `variants*.json` | |
| Trained receivers | `exp_train.py` | `ckpt/*.pt` | |

`python3 code/replot.py` regenerates the seven result figures from `data/`
(set `SEMA_DEVICE=cpu` on machines without CUDA); every figure is drawn on a
2.975 in canvas (0.85 column width) with an 8:6 axes box and 8 pt Times text (STIX math). `python3 code/fill_tables.py`
regenerates the LaTeX source of Tables III to V. `fig/fig1_system.pdf` is the
system diagram of Fig. 1. The experiments were run on one NVIDIA RTX A4500
(CUDA) and the closed-form checks of `exp0_theory.py` on CPU.

## Requirements and conventions

Python 3.10+, PyTorch 2.x (CUDA optional; the Mac fallback is CPU because
MPS lacks batched QR), NumPy, Matplotlib. Seeds are fixed inside every
script (`sema_lib.gen(seed)`). Energy and channel conventions: unit-energy
embeddings, unit-norm signatures of length U, per-user SNR
rho = 1/(d sigma^2) with sigma^2 the per-chip noise variance, block
Rayleigh gains |CN(0,1)| floored at 0.2 after phase compensation.

The real-embedding caches (`data/real/emb_*.pt`, mean-pooled BERT and
DistilBERT outputs of AG News and SST-2 sentences with labels) are not
included because of their size; `exp8_real.py` rebuilds them from the
Hugging Face datasets and models named in the script on first run.

## License

MIT for the code (`LICENSE`); result data and figures CC BY 4.0.
