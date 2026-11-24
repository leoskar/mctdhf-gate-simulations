#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=rs_n4
#SBATCH -o logs/norb4_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# norb=4 confirmation of each leg's refined argmax. The MF surrogate is a cheap
# parameter OPTIMISER; optima are confirmed at full orbital count before being
# reported. Task 3 (w0w3) is the long one (~3.5 h) and is ordered last so the
# three cheap adjacent legs report first.
set -uo pipefail
export LC_ALL=C          # a decimal-comma locale breaks awk/printf
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset SWEEP_DT JAX_PLATFORMS
export SWEEP_INTEG=cmf SWEEP_LEG_TYPE=sqiswap SWEEP_DEV=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
case "${SLURM_ARRAY_TASK_ID}" in
  0) export LEG=w0w1 TAU_RAMP=82.3 TAU_HOLD=73.2   SWEEP_DT_DIV=0.1  ;;  # MF 0.99599
  1) export LEG=w1w2 TAU_RAMP=57.2 TAU_HOLD=95.4   SWEEP_DT_DIV=0.1  ;;  # MF 0.99866
  2) export LEG=w2w3 TAU_RAMP=34.1 TAU_HOLD=112.8  SWEEP_DT_DIV=0.1  ;;  # MF 0.99780
  3) export LEG=w0w3 TAU_RAMP=81.3 TAU_HOLD=2160.7 SWEEP_DT_DIV=0.25 ;;  # MF 0.99608, 4dt
  *) echo "bad task id"; exit 1 ;;
esac
echo "[norb4] task=$SLURM_ARRAY_TASK_ID leg=$LEG tau=($TAU_RAMP,$TAU_HOLD) dt_div=$SWEEP_DT_DIV"
python scripts/confirm_norb4.py
rc=$?; echo "[norb4] exit=$rc"; exit $rc
