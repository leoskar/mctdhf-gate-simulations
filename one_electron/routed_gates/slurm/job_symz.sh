#!/bin/bash
#SBATCH --job-name=symz
#SBATCH -o logs/%x_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2%1
# Step 4. The three routed circuits of the paper, at 4dt (0.32 ps), with the
# deterministic Z frames of legs16_symz.npz (step 3, canonical_frames.py):
# the symmetric split of each standalone gate, reused in every circuit, plus
# free evolution for idle time. No leg is re-propagated for the frames.
# Submit from one_electron/routed_gates/ (mkdir -p logs first).
COMBO=(serial:best parallel:best parallel:sync)
IFS=: read -r ARM_ PULSE_SET <<< "${COMBO[$SLURM_ARRAY_TASK_ID]}"
export PULSE_SET
source slurm/_common.sh
export ARM=$ARM_
export SWEEP_DT_DIV=0.25 ALLOW_DT_OVERRIDE=1 LEGS_TAG=symz
echo "[symz] ARM=$ARM PULSE_SET=$PULSE_SET SWEEP_DT_DIV=$SWEEP_DT_DIV LEGS_TAG=$LEGS_TAG"
python scripts/run_arm_decomp.py
