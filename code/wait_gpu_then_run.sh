#!/usr/bin/env bash
# Waits until at least $1 MiB of GPU memory is free, then runs the given command.
need=${1:-2500}; shift
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  free=$((total - used))
  if [ "$free" -ge "$need" ]; then echo "GPU free ${free} MiB at $(date), starting"; break; fi
  sleep 60
done
exec "$@"
