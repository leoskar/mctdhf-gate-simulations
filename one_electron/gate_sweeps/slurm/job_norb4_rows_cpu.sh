#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=mf_n4c
#SBATCH -o logs/mfcpu_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
# norb=[4,4,4,4] confirmations for the mean-field-surrogate comparison figure.
#   D = per-ramp row peak, three adjacent legs (does MF pick the same pulse?)
#   A = the full hold row at w1w2 tau_ramp=57.2 (does MF reproduce the shape?)
# CPU only: JAX_PLATFORMS=cpu is set explicitly so these never probe CUDA while
# the long w0w3 job holds the device.
set -uo pipefail
export LC_ALL=C
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
unset SWEEP_DT SWEEP_DEV          # SWEEP_DT is NOT SWEEP_DT_DIV
export SWEEP_INTEG=cmf SWEEP_LEG_TYPE=sqiswap
MAN=scripts/manifest_norb4_rows_cpu.txt
read -r TAG LEG TAU_RAMP TAU_HOLD DTDIV < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MAN")
if [ -z "${LEG:-}" ]; then echo "bad task id $SLURM_ARRAY_TASK_ID"; exit 1; fi
export LEG TAU_RAMP TAU_HOLD
export SWEEP_DT_DIV="$DTDIV"
echo "[mf-cpu] task=$SLURM_ARRAY_TASK_ID [$TAG] leg=$LEG tau=($TAU_RAMP,$TAU_HOLD) dt_div=$DTDIV"
python scripts/confirm_norb4.py
rc=$?; echo "[mf-cpu] exit=$rc"; exit $rc
