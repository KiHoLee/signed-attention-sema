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

| Paper item | Script | Data |
|---|---|---|
| Fig. 2 (closed form vs Monte Carlo) | `exp0_theory.py` | `theory_snr.csv` |
| Fig. 3 (weights), Figs. 4-5 (SNR sweeps) | `exp2_snr.py` | `snr_sweep_b*.csv` |
| Fig. 6 (similarity sweep) | `exp1_beta.py` | `beta_sweep_*dB.csv` |
| Fig. 7 (signature design) | `exp6_sigdesign.py` | `sigopt.csv`, `sigdesign_*dB.csv` |
| Fig. 8 (user scaling), Table dim | `exp5_scaling.py` | `scaling_U.csv`, `scaling_d.csv` |
| Table profiles | `exp3_profiles.py` | `profiles.json` |
| Table block-varying | `exp4_varmix.py` | `varmix.json` |
| Table masks, mismatch, complexity | `exp7_misc.py` | `haar_masks.csv`, `mismatch.csv`, `complexity.json` |
| Figs. 9-10, Table real | `exp8_real.py` | `real_*.csv`, `real_stats.json` |
| receivers | `exp_train.py` | `ckpt/*.pt` |

## Requirements and conventions

Python 3.10+, PyTorch 2.x (CUDA optional; the Mac fallback is CPU because
MPS lacks batched QR), NumPy, Matplotlib. Seeds are fixed inside every
script (`sema_lib.gen(seed)`). Energy and channel conventions: unit-energy
embeddings, unit-norm signatures of length U, per-user SNR
rho = 1/(d sigma^2) with sigma^2 the per-chip noise variance, block
Rayleigh gains |CN(0,1)| floored at 0.2 after phase compensation.

The real-embedding caches (`data/real/emb_*.pt`, mean-pooled BERT and
DistilBERT outputs of AG News and SST-2 sentences with labels) are
produced by the encoder script of the companion repository and are
included for convenience.

## License

MIT for the code; result data and figures CC BY 4.0.
