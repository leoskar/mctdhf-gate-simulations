#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=rs_k16c
#SBATCH -o logs/k16cpu_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
# K=16 register confirmations for the five adjacent-leg points, in parallel on
# CPU while the GPU carries the long w0w3 run.
set -uo pipefail
export LC_ALL=C
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
unset SWEEP_DT SWEEP_DEV
export SWEEP_INTEG=cmf SWEEP_LEG_TYPE=sqiswap SWEEP_DT_DIV=0.1
case "${SLURM_ARRAY_TASK_ID}" in
  0) export LEG=w0w1 TAU_RAMP=82.3 TAU_HOLD=73.2  ;;   # F4 0.995996
  1) export LEG=w1w2 TAU_RAMP=57.2 TAU_HOLD=95.4  ;;   # F4 0.999270
  2) export LEG=w1w2 TAU_RAMP=8.9  TAU_HOLD=137.1 ;;   # F4 0.998754 (fast)
  3) export LEG=w2w3 TAU_RAMP=34.1 TAU_HOLD=112.8 ;;   # F4 0.997988
  4) export LEG=w2w3 TAU_RAMP=11.9 TAU_HOLD=131.7 ;;   # F4 0.997879 (fast)
  *) echo "bad task id"; exit 1 ;;
esac
echo "[k16-cpu] task=$SLURM_ARRAY_TASK_ID leg=$LEG tau=($TAU_RAMP,$TAU_HOLD)"
python scripts/confirm_k16.py
rc=$?; echo "[k16-cpu] exit=$rc"; exit $rc
