#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=rs_segAc
#SBATCH -o logs/segAc_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
# One segment A per (leg, tau_ramp); checkpoints every SNAP_DT=0.1 serve every
# hold below th_max. SLURM (task/affinity, cons_tres/CR_CORE) binds the bare
# batch step to exactly --cpus-per-task cores, so no taskset / srun wrapper.
set -uo pipefail
export LC_ALL=C          # a decimal-comma locale breaks awk/printf
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
unset SWEEP_DT
export SWEEP_INTEG=cmf SWEEP_C_PROP=diagonalize SWEEP_B_PROP=diagonalize
export SWEEP_LEG_TYPE=sqiswap MEANFIELD_SPEC=1 SWEEP_DT_DIV=0.1 SWEEP_DEV=cpu
MF=scripts/manifest_segA_cpu.txt
read -r LEG R TH <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $MF)"
export MODE=segA LEG="$LEG" TAU_RAMP="$R" TH_MAX="$TH"
echo "[segA-cpu] task=$SLURM_ARRAY_TASK_ID leg=$LEG tau_ramp=$R th_max=$TH cores=$(taskset -cp $$ 2>/dev/null)"
python scripts/run_segment.py
rc=$?; echo "[segA-cpu] exit=$rc"; exit $rc
