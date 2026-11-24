#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=rs_n4c
#SBATCH -o logs/norb4cpu_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
# norb=4 confirmation of the FASTEST point within each leg's own measured
# MF->norb=4 shift. Runs on CPU (both tasks in parallel) because the GPU is
# occupied by the long w0w3 confirmation. JAX_PLATFORMS=cpu is set explicitly so
# the process never probes CUDA while the GPU job holds the device.
set -uo pipefail
export LC_ALL=C          # a decimal-comma locale breaks awk/printf
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
unset SWEEP_DT SWEEP_DEV
export SWEEP_INTEG=cmf SWEEP_LEG_TYPE=sqiswap SWEEP_DT_DIV=0.1
case "${SLURM_ARRAY_TASK_ID}" in
  0) export LEG=w1w2 TAU_RAMP=8.9  TAU_HOLD=137.1 ;;  # MF 0.998312, 2.47 ns
  1) export LEG=w2w3 TAU_RAMP=11.9 TAU_HOLD=131.7 ;;  # MF 0.997712, 2.47 ns
  *) echo "bad task id"; exit 1 ;;
esac
echo "[norb4-cpu] task=$SLURM_ARRAY_TASK_ID leg=$LEG tau=($TAU_RAMP,$TAU_HOLD)"
python scripts/confirm_norb4.py
rc=$?; echo "[norb4-cpu] exit=$rc"; exit $rc
