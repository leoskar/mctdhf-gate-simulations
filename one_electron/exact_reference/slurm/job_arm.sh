#!/bin/bash
#SBATCH --job-name=exact_arm
#SBATCH -o logs/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
# The synchronized parallel routing propagated on the untruncated 64^4 grid,
# with the Z frames of the MCTDH circuit (routed_gates/results/arms_sync/
# legs16_symz.npz).  Runs in the paper: DT_FINE = 0.25 and 0.5.
# e.g.  DT_FINE=0.25 sbatch slurm/job_arm.sh
# Submit from one_electron/exact_reference/ (mkdir -p logs first).
export DT_FINE=${DT_FINE:?set DT_FINE}
source slurm/_common.sh
export PULSE_SET=sync ARM=parallel PREC=x64 SPLIT_ORDER=4 LEGS_TAG=symz SNAP_EVERY=5.0
python scripts/propagate_arm.py
