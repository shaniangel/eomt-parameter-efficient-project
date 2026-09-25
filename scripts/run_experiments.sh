#!/usr/bin/env bash
# Trains the given regimes one after another on this machine's GPU, with identical settings.
#
# Usage:
#   bash scripts/run_experiments.sh [full] [frozen] [lora] [--smoke] [extra main.py args...]
#
# With no regime given, all three are run. With two machines, split the work, e.g.:
#   machine A:  bash scripts/run_experiments.sh full
#   machine B:  bash scripts/run_experiments.sh frozen lora
# then copy logs/<regime>/ from machine B into logs/ on machine A before summarizing.
#
# --smoke stacks configs/project/smoke.yaml for a quick end-to-end check.
# Extra arguments are passed to main.py, e.g. --data.path /path/to/ade20k
set -euo pipefail
cd "$(dirname "$0")/.."

regimes=()
smoke=()
extra=()
for arg in "$@"; do
  case "$arg" in
    full|frozen|lora) regimes+=("$arg") ;;
    --smoke) smoke=(-c configs/project/smoke.yaml) ;;
    *) extra+=("$arg") ;;
  esac
done
[[ ${#regimes[@]} -eq 0 ]] && regimes=(full frozen lora)

mkdir -p logs
for regime in "${regimes[@]}"; do
  out="logs/${regime}_$(date '+%F_%H-%M-%S').out"
  echo "[$(date '+%F %T')] Training $regime (console output: $out)"
  python main.py fit \
    -c configs/project/base_ade20k_eomt_small_512.yaml \
    -c "configs/project/$regime.yaml" \
    ${smoke[@]+"${smoke[@]}"} ${extra[@]+"${extra[@]}"} \
    > "$out" 2>&1 || { echo "Training $regime failed, see the end of $out"; exit 1; }
done

echo "[$(date '+%F %T')] Done. Summarize with: python scripts/summarize_results.py"
