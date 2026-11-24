#!/bin/bash
#SBATCH --job-name=legs16_4dt
#SBATCH -o logs/%x_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-1%1
# Step 2. Per-leg matrices at 4dt (0.32 ps), the step of the routed circuits,
# for the 'best' and 'sync' pulse sets. They are the input of
# canonical_frames.py (step 3). The 'best' store is written as
# legs16_legsdt0.25.npz so that the 10dt legs16.npz of step 1 is kept.
# Submit from one_electron/routed_gates/ (mkdir -p logs first).
COMBO=(best:legsdt0.25 sync:)
IFS=: read -r PULSE_SET LEGS_TAG <<< "${COMBO[$SLURM_ARRAY_TASK_ID]}"
export PULSE_SET LEGS_TAG
source slurm/_common.sh
export ARM=legs16 SWEEP_DT_DIV=0.25 ALLOW_DT_OVERRIDE=1
python scripts/run_arm_decomp.py
