#!/usr/bin/env bash
# Reruns the low-SER points of Figs. 5, 7, and 8 with ten times the blocks, so
# that the residual floors rest on about a hundred errors per point instead
# of ten. Run on the GPU server from the project folder; merge with
# python3 code/merge_floor.py afterwards.
set -e
P=${PYTHON:-$HOME/tr_env/bin/python}
mkdir -p logs
SEMA_N=2000000 SEMA_CHUNK=10000 SEMA_BETAS=0.64 SEMA_SNRS=4,6,8,10,12,14,16,18,20,22,24 SEMA_SUFFIX=_floor $P code/exp2_snr.py > logs/exp2_floor.log 2>&1
SEMA_N=2000000 SEMA_CHUNK=10000 SEMA_BETAS=0.75,0.8,0.85,0.9 SEMA_SNRS=0 SEMA_SUFFIX=_floor $P code/exp1_beta.py > logs/exp1_floor.log 2>&1
SEMA_REP=1000 SEMA_SNRS=12,14,16,18,20 SEMA_SUFFIX=_floor $P code/exp8_real.py > logs/exp8_floor.log 2>&1
echo "floor reruns done $(date)"
