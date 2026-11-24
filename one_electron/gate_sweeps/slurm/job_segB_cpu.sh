#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=rs_segBc
#SBATCH -o logs/segBc_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
set -uo pipefail
export LC_ALL=C          # a decimal-comma locale breaks awk/printf
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
unset SWEEP_DT
export SWEEP_INTEG=cmf SWEEP_C_PROP=diagonalize SWEEP_B_PROP=diagonalize
export SWEEP_LEG_TYPE=sqiswap MEANFIELD_SPEC=1 SWEEP_DT_DIV=0.1 SWEEP_DEV=cpu
MF=${RESWEEP_MANIFEST:-scripts/manifest_segB_cpu.txt}
read -r LEG R H <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $MF)"
export MODE=segB LEG="$LEG" TAU_RAMP="$R" HOLDS="$H"
echo "[segB-cpu] task=$SLURM_ARRAY_TASK_ID leg=$LEG tau_ramp=$R hold=$H"
python scripts/run_segment.py
rc=$?; echo "[segB-cpu] exit=$rc"; exit $rc
