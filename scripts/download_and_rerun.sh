#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data2/home/qrh/data/code/LayerSkip"
modelscope_bin="/home/qrh/miniconda3/envs/layer/bin/modelscope"
log_dir="$repo_dir/logs/rerun_20260814"
mkdir -p "$log_dir"

retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if "$@"; then
      return 0
    fi
    echo "Attempt $attempt failed; retrying in 20 seconds..." >&2
    sleep 20
  done
  return 1
}

# Only Hugging Face weights are needed. ModelScope's original/*.pth copies
# duplicate roughly 15 GB for Llama-3-8B and are intentionally excluded.
retry "$modelscope_bin" download --model LLM-Research/Meta-Llama-3-8B-Instruct \
  --local_dir /data/meta-llama/Meta-Llama-3-8B-Instruct \
  --exclude 'original/*' --max-workers 4
retry "$modelscope_bin" download --model LLM-Research/Llama-3.2-1B-Instruct \
  --local_dir /data/meta-llama/Llama-3.2-1B-Instruct \
  --exclude 'original/*' --max-workers 4

retry "$modelscope_bin" download --dataset cais/mmlu --local_dir /data/cais/mmlu
retry "$modelscope_bin" download --dataset evalscope/hellaswag --local_dir /data/Rowan/hellaswag
retry "$modelscope_bin" download --dataset allenai/winogrande --local_dir /data/allenai/winogrande
retry "$modelscope_bin" download --dataset openai-mirror/gsm8k --local_dir /data/openai/gsm8k
retry "$modelscope_bin" download --dataset openai-mirror/openai_humaneval \
  --local_dir /data/openai/openai_humaneval

cd "$repo_dir"

# Each worker sees exactly one of the permitted physical GPUs. GPU 4-7 are
# never exposed to these processes.
CUDA_VISIBLE_DEVICES=0 conda run -n layer python eval.py \
  --model meta-llama/Llama-3.2-1B-Instruct --strategy none \
  --tasks mmlu hellaswag winogrande gsm8k humaneval --local --device cuda \
  >"$log_dir/gpu0_1b_none.log" 2>&1 &
echo $! >"$log_dir/gpu0_1b_none.pid"

CUDA_VISIBLE_DEVICES=1 conda run -n layer python eval.py \
  --model meta-llama/Llama-3.2-1B-Instruct --strategy calibratedskip \
  --calibratedskip_metrics activation_ratio gradient_value gradient_trace shapley_value \
  --calibration_max_samples 4096 --tasks mmlu hellaswag winogrande gsm8k \
  --local --device cuda >"$log_dir/gpu1_1b_calibration.log" 2>&1 &
echo $! >"$log_dir/gpu1_1b_calibration.pid"

CUDA_VISIBLE_DEVICES=2 conda run -n layer python eval.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct --strategy none \
  --tasks mmlu hellaswag winogrande gsm8k humaneval --local --device cuda \
  >"$log_dir/gpu2_8b_none.log" 2>&1 &
echo $! >"$log_dir/gpu2_8b_none.pid"

CUDA_VISIBLE_DEVICES=3 conda run -n layer python eval.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct --strategy calibratedskip \
  --calibratedskip_metrics activation_ratio gradient_value gradient_trace shapley_value \
  --calibration_max_samples 4096 --tasks mmlu hellaswag winogrande gsm8k \
  --local --device cuda >"$log_dir/gpu3_8b_calibration.log" 2>&1 &
echo $! >"$log_dir/gpu3_8b_calibration.pid"

wait
