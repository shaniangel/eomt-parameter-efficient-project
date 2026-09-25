#!/usr/bin/env bash
# Trains the given regimes one after another on this machine's GPU, with identical settings.
#
# Usage:
#   bash scripts/run_experiments.sh [full] [frozen] [lora] [--smoke] [extra main.py args...]
#   bash scripts/run_experiments.sh <regime> --resume logs/<regime>/<run folder>
#
# With no regime given, all three are run. To use one GPU machine per regime, run e.g.:
#   machine A:  bash scripts/run_experiments.sh full
#   machine B:  bash scripts/run_experiments.sh frozen
#   machine C:  bash scripts/run_experiments.sh lora
# then copy the logs/<regime>/ folders into logs/ on one machine before summarizing.
#
# --smoke stacks configs/project/smoke.yaml for a quick end-to-end check; its runs go to
#   logs_smoke/ so they never mix with the real runs in logs/.
# --resume continues an interrupted run from its latest checkpoint (saved at the end of every
#   epoch), in the same run folder.
# Extra arguments are passed to main.py, e.g. --data.path /path/to/ade20k
set -euo pipefail
cd "$(dirname "$0")/.."

regimes=()
smoke=()
resume_dir=""
extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    full|frozen|lora) regimes+=("$1"); shift ;;
    --smoke) smoke=(-c configs/project/smoke.yaml); shift ;;
    --resume) resume_dir="${2%/}"; shift 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done
[[ ${#regimes[@]} -eq 0 ]] && regimes=(full frozen lora)

if [[ -n "$resume_dir" ]]; then
  if [[ ${#regimes[@]} -ne 1 ]]; then
    echo "--resume needs exactly one regime, e.g.: bash scripts/run_experiments.sh lora --resume $resume_dir"
    exit 1
  fi
  ckpt=$(ls -t "$resume_dir"/checkpoints/*.ckpt 2>/dev/null | head -1 || true)
  if [[ -z "$ckpt" ]]; then
    echo "No checkpoint found in $resume_dir/checkpoints/ (one is saved at the end of every epoch)"
    exit 1
  fi
  echo "Resuming from $ckpt"
  # The run folder is <save_dir>/<regime>/<version>: log into that same folder again
  extra+=(--ckpt_path "$ckpt"
    --trainer.logger.init_args.save_dir "$(dirname "$(dirname "$resume_dir")")"
    --trainer.logger.init_args.version "$(basename "$resume_dir")")
fi

# Lets PyTorch reuse freed GPU memory between training and validation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

out_dir=logs
[[ ${#smoke[@]} -gt 0 ]] && out_dir=logs_smoke
mkdir -p "$out_dir"
for regime in "${regimes[@]}"; do
  out="$out_dir/${regime}_$(date '+%F_%H-%M-%S').out"
  echo "[$(date '+%F %T')] Training $regime (console output: $out)"
  python main.py fit \
    -c configs/project/base_ade20k_eomt_small_512.yaml \
    -c "configs/project/$regime.yaml" \
    ${smoke[@]+"${smoke[@]}"} ${extra[@]+"${extra[@]}"} \
    > "$out" 2>&1 || { echo "Training $regime failed, see the end of $out"; exit 1; }
done

echo "[$(date '+%F %T')] Done. Summarize with: python scripts/summarize_results.py"
