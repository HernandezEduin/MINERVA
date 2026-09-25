#!/bin/bash

config="$1"
export PYTHONPATH="."
gpu_id="${2:-}"   # optional (e.g., 0). If empty -> CPU.

# ---- Make sure conda-provided CUDA/cuDNN are used (TF 2.11 expects CUDA 11.x + cuDNN 8.x) ----
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

if command -v python >/dev/null 2>&1; then
  python_cmd=(python)
elif command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "minerva_tf2"; then
  python_cmd=(conda run -n minerva_tf2 python)
else
  python_cmd=(python3)
fi

cmd=("${python_cmd[@]}" code/model/evaluation.py --config_yaml "$config")

# If no GPU id given, or GPU doesn't exist, run on CPU
if [[ -z "$gpu_id" ]] || ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -i "$gpu_id" >/dev/null 2>&1; then
  echo "Executing (CPU): ${cmd[*]}"
  CUDA_VISIBLE_DEVICES="" "${cmd[@]}"
else
  echo "Executing: CUDA_VISIBLE_DEVICES=$gpu_id ${cmd[*]}"
  CUDA_VISIBLE_DEVICES=$gpu_id "${cmd[@]}"
fi