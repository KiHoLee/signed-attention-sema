#!/usr/bin/env bash
# Full queue on AINS: training, then every experiment, sequentially (GPU is
# shared). Logs in logs/<name>.log; rerun any single step by name.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=code
PY=${PY:-$HOME/tr_env/bin/python}
mkdir -p logs data/ckpt
for s in ${@:-exp_train exp1_beta exp2_snr exp3_profiles exp4_varmix exp5_scaling exp6_sigdesign exp7_misc exp8_real}; do
  echo "=== $s $(date)" | tee -a logs/queue.log
  $PY -I code/$s.py > logs/$s.log 2>&1
  echo "=== $s exit $? $(date)" | tee -a logs/queue.log
done
