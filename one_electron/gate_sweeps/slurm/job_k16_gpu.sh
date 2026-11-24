#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=rs_k16g
#SBATCH -o logs/k16gpu_%A.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# w0w3 K=16: the long pole (~2323 t.u. at 4dt). 10dt diverges on this leg.
set -uo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset SWEEP_DT JAX_PLATFORMS
export SWEEP_INTEG=cmf SWEEP_LEG_TYPE=sqiswap SWEEP_DT_DIV=0.25 SWEEP_DEV=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LEG=w0w3 TAU_RAMP=81.3 TAU_HOLD=2160.7
echo "[k16-gpu] leg=$LEG tau=($TAU_RAMP,$TAU_HOLD)"
python scripts/confirm_k16.py
rc=$?; echo "[k16-gpu] exit=$rc"; exit $rc
