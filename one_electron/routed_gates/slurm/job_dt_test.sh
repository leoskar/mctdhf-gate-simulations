#!/bin/bash
#SBATCH --job-name=dt_test
#SBATCH -o logs/%x_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-1%1
# Time-step test of a routed circuit (SM): the parallel routing with the
# 'fast' pulse set at 4dt (0.32 ps) and 2dt (0.16 ps). Needs
# arms_fast/legs16.npz from job_legs16.sh.
# Submit from one_electron/routed_gates/ (mkdir -p logs first).
DTDIV=(0.25 0.5)
export PULSE_SET=fast
source slurm/_common.sh
export ARM=parallel SWEEP_DT_DIV=${DTDIV[$SLURM_ARRAY_TASK_ID]} ALLOW_DT_OVERRIDE=1
python scripts/run_arm_decomp.py
