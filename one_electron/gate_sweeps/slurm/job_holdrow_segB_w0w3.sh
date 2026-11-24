#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=fl_segB
#SBATCH -o logs/segB_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
# Segment A at norb=[4,4,4,4] for the w0w3 hold row (tau_ramp = 81.3).
# Propagates ramp-up + plateau once, checkpointing on the SNAP_DT = 0.1 grid out
# to th_max = 2865.7, so each subsequent hold costs only a step-back plus
# ramp-down (2*tau_ramp = 162.6 t.u. ~ 14 min) instead of a full ~2000 t.u. leg
# (~190 min).  Measured rate on this leg: 5.25 s/t.u. => segA ~ 4.2 h.
set -uo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset SWEEP_DT JAX_PLATFORMS          # SWEEP_DT is NOT SWEEP_DT_DIV
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SWEEP_INTEG=cmf SWEEP_C_PROP=diagonalize SWEEP_B_PROP=diagonalize
export SWEEP_LEG_TYPE=sqiswap
export MEANFIELD_SPEC=0               # full orbitals -- the whole point
export SWEEP_DEV=cuda
export SWEEP_DT_DIV=0.25              # 4dt: 10dt diverges on this leg
export LEG=w0w3 MODE=segB TAU_RAMP=81.3
export HOLDS="${HOLDS:?set HOLDS}"
echo "[segB] leg=$LEG tau_ramp=$TAU_RAMP holds=$HOLDS dt_div=$SWEEP_DT_DIV"
python scripts/run_segment.py
rc=$?; echo "[segB] exit=$rc"; exit $rc
