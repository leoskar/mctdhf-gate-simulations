#!/bin/bash
#SBATCH --job-name=legs16
#SBATCH -o logs/%x_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-1%1
# Step 1. Per-leg 16x16 gate matrices at the production step 10dt (0.80 ps)
# for the 'best' and 'fast' pulse sets, including the concurrent (0,1)||(2,3)
# leg. arms_best/legs16.npz is the MCTDH side of the nearest-neighbor gates in
# the comparison with the exact reference; arms_fast/legs16.npz is the input
# of the time-step test (job_dt_test.sh).
# Submit from one_electron/routed_gates/ (mkdir -p logs first).
SETS=(best fast)
export PULSE_SET=${SETS[$SLURM_ARRAY_TASK_ID]}
source slurm/_common.sh
export ARM=legs16
python scripts/run_arm_decomp.py
