#!/usr/bin/env bash
# Trains SC-rho and runs the floor study (SNR sweep, then beta sweep) on the GPU server.
set -e
P=${PYTHON:-$HOME/tr_env/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
$P code/exp_train.py gramsc_U4_d64_s1_rho > logs/train_rho.log 2>&1
SEMA_CHUNK=10000 $P code/exp_floor_variants.py snr  > logs/floor_variants_snr.log 2>&1
SEMA_CHUNK=10000 $P code/exp_floor_variants.py beta > logs/floor_variants_beta.log 2>&1
echo "floor variants done $(date)"
