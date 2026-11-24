#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=mf_n4g
#SBATCH -o logs/mfgpu_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
# norb=[4,4,4,4] confirmations for the mean-field-surrogate comparison figure.
#   C = w0w3 at five holds across the tau_ramp=81.3 row.  This is the leg whose
#       spectators MEDIATE the exchange, where the surrogate is OPTIMISTIC
#       (-7.8e-3 at the argmax) rather than conservative -- the contrast the
#       figure exists to show.  ~3.4 h per point, so serialise with %1.
# GPU: 4dt (SWEEP_DT_DIV=0.25) is mandatory on this leg -- 10dt diverges.
set -uo pipefail
export LC_ALL=C
export SWEEP_DEV=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset SWEEP_DT JAX_PLATFORMS          # SWEEP_DT is NOT SWEEP_DT_DIV
export SWEEP_INTEG=cmf SWEEP_LEG_TYPE=sqiswap
MAN=scripts/manifest_norb4_rows_gpu.txt
read -r TAG LEG TAU_RAMP TAU_HOLD DTDIV < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MAN")
if [ -z "${LEG:-}" ]; then echo "bad task id $SLURM_ARRAY_TASK_ID"; exit 1; fi
export LEG TAU_RAMP TAU_HOLD
export SWEEP_DT_DIV="$DTDIV"
echo "[mf-gpu] task=$SLURM_ARRAY_TASK_ID [$TAG] leg=$LEG tau=($TAU_RAMP,$TAU_HOLD) dt_div=$DTDIV"
python scripts/confirm_norb4.py
rc=$?; echo "[mf-gpu] exit=$rc"; exit $rc
