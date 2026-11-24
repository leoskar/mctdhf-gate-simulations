#!/bin/bash
#SBATCH --job-name=exact_leg
#SBATCH -o logs/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
# One gate propagated on the untruncated 64^4 grid: all 16 inputs, double
# precision, fourth-order split operator, the same schedule, anchors and qubit
# orbitals as MCTDH.
#   EXACT_LEG=w0w1|w1w2|w2w3|w0w3   DT_FINE=step in t.u.   DEV=cuda|cpu
# Runs in the paper:
#   w0w1, w1w2, w2w3 at DT_FINE = 0.125, 0.25, 0.5
#   w0w3             at DT_FINE = 0.25, 0.5
# e.g.  EXACT_LEG=w0w1 DT_FINE=0.125 sbatch slurm/job_leg.sh
# Submit from one_electron/exact_reference/ (mkdir -p logs first).
export EXACT_LEG=${EXACT_LEG:?set EXACT_LEG} DT_FINE=${DT_FINE:?set DT_FINE}
source slurm/_common.sh
export PULSE_SET=best ANCHOR=primary PREC=x64 PROPAGATOR=strang SPLIT_ORDER=4
if [ "$EXACT_LEG" = "w0w3" ]; then export SNAP_EVERY=10.0; else export SNAP_EVERY=5.0; fi
python scripts/propagate_leg.py
