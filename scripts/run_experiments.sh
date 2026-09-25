#!/usr/bin/env bash
# Trains the three regimes (full, frozen, lora) with identical settings.
#
# Usage:
#   bash scripts/run_experiments.sh [--smoke] [--gpus 0,1] [extra main.py args...]
#
# With two GPUs, full fine-tuning (the slowest regime) runs on the first GPU while
# frozen and then lora run on the second. With one GPU, all three run one after another.
# --smoke stacks configs/project/smoke.yaml for a quick end-to-end check.
# Extra arguments are passed to main.py, e.g. --data.path /path/to/ade20k
set -euo pipefail
cd "$(dirname "$0")/.."

configs=(-c configs/project/base_ade20k_eomt_small_512.yaml)
smoke=()
gpus="0"
extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) smoke=(-c configs/project/smoke.yaml); shift ;;
    --gpus) gpus="$2"; shift 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done

mkdir -p logs
train() {  # train <gpu> <regime>
  echo "[$(date '+%F %T')] Training $2 on GPU $1 (log: logs/$2.out)"
  CUDA_VISIBLE_DEVICES="$1" python main.py fit "${configs[@]}" \
    -c "configs/project/$2.yaml" ${smoke[@]+"${smoke[@]}"} ${extra[@]+"${extra[@]}"} \
    > "logs/$2.out" 2>&1
}

IFS=, read -r -a gpu_list <<< "$gpus"
if [[ ${#gpu_list[@]} -ge 2 ]]; then
  train "${gpu_list[0]}" full &
  full_pid=$!
  train "${gpu_list[1]}" frozen && train "${gpu_list[1]}" lora
  wait "$full_pid"
else
  for regime in full frozen lora; do
    train "${gpu_list[0]}" "$regime"
  done
fi

echo "[$(date '+%F %T')] Done. Summarize with: python scripts/summarize_results.py"
