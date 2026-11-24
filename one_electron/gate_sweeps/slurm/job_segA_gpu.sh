#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=rs_segAg
#SBATCH -o logs/segAg_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# w0w3 at SWEEP_DT_DIV=0.25 (4dt): 10dt diverges on this leg. Device selection
# goes through jax.default_device inside the wrapper -- never JAX_PLATFORMS=cuda,
# which leaves qoct_gate_leg/rte_step (which pin jax_platform_name='cpu') with
# no cpu backend at import.
set -uo pipefail
export LC_ALL=C          # a decimal-comma locale breaks awk/printf
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset SWEEP_DT JAX_PLATFORMS
export SWEEP_INTEG=cmf SWEEP_C_PROP=diagonalize SWEEP_B_PROP=diagonalize
export SWEEP_LEG_TYPE=sqiswap MEANFIELD_SPEC=1 SWEEP_DT_DIV=0.25 SWEEP_DEV=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
MF=scripts/manifest_segA_gpu.txt
read -r LEG R TH <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $MF)"
export MODE=segA LEG="$LEG" TAU_RAMP="$R" TH_MAX="$TH"
echo "[segA-gpu] task=$SLURM_ARRAY_TASK_ID leg=$LEG tau_ramp=$R th_max=$TH"
python scripts/run_segment.py
rc=$?; echo "[segA-gpu] exit=$rc"; exit $rc
