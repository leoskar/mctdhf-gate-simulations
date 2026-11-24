#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=rs_segBg
#SBATCH -o logs/segBg_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# One array task per tau_ramp: all 13 holds in one python process, so the JAX
# backend/CUDA init is paid once per ramp rather than 13 times.
set -uo pipefail
export LC_ALL=C          # a decimal-comma locale breaks awk/printf
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset SWEEP_DT JAX_PLATFORMS
export SWEEP_INTEG=cmf SWEEP_C_PROP=diagonalize SWEEP_B_PROP=diagonalize
export SWEEP_LEG_TYPE=sqiswap MEANFIELD_SPEC=1 SWEEP_DT_DIV=0.25 SWEEP_DEV=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
MF=${RESWEEP_MANIFEST:-scripts/manifest_segB_gpu.txt}
LEG=$(awk -v k=$((SLURM_ARRAY_TASK_ID+1)) 'NR==1{n=0} {if(!($1" "$2 in seen)){seen[$1" "$2]=++n}} seen[$1" "$2]==k{print $1; exit}' $MF)
R=$(awk -v k=$((SLURM_ARRAY_TASK_ID+1)) '{if(!($1" "$2 in seen)){seen[$1" "$2]=++n}} seen[$1" "$2]==k{print $2; exit}' $MF)
H=$(awk -v l="$LEG" -v r="$R" '$1==l && $2==r{printf "%s%s",(c++?",":""),$3} END{print ""}' $MF)
export MODE=segB LEG="$LEG" TAU_RAMP="$R" HOLDS="$H"
echo "[segB-gpu] task=$SLURM_ARRAY_TASK_ID leg=$LEG tau_ramp=$R holds=$H"
python scripts/run_segment.py
rc=$?; echo "[segB-gpu] exit=$rc"; exit $rc
