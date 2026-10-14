#!/bin/bash
#SBATCH --job-name=egb-ff
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@90

set -euo pipefail
task_root=$1
task_python=$2
task_campaign=$3
task_plan=$4
task_offset=$5
cd "$task_root"
export PYTHONPATH="$task_root/src:$task_root/benchmarks:$task_root/outputs/referee/.precision_deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$task_root/outputs/referee/.mplconfig"
export XDG_CACHE_HOME="$task_root/outputs/referee/.cache"
export PYTHONUNBUFFERED=1
exec "$task_python" "$task_root/benchmarks/referee_task_2_slurm.py" \
    --campaign "$task_campaign" run --plan "$task_plan" \
    --index "$((SLURM_ARRAY_TASK_ID + task_offset))"
