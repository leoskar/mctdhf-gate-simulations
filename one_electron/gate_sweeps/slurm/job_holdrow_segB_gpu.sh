#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=hr_segBg
#SBATCH -o logs/hrsegBg_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
# Segment B for the w0w1 hold row (tau_ramp = 82.3), norb=[4,4,4,4]. Genuine
# ramp-down from the segA checkpoint at t_c = tau_hold; each point costs
# 2*tau_ramp = 164.6 t.u. regardless of the hold. Holds are batched a few per
# task so process/JAX startup is amortised, as in job_holdrow_segB_w0w3.sh.
# Needs job_holdrow_segA.sh task 0 to have finished -- submit with
#   --dependency=afterok:<segA jobid>
# Runs the LONG leg (16 x 164.6 t.u.) because the GPU is ~7x faster per t.u.;
# the cheaper w2w3 row goes to the CPU array in parallel.
# th=73.2 is deliberately absent: already confirmed as a full leg
# (results/norb4_confirm/n4_w0w1_tr82.3_th73.2.npz, F=0.99600), and
# figures/mf_fig_wide.py's n4_points() prefers the full-leg value anyway.
set -uo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset SWEEP_DT JAX_PLATFORMS          # SWEEP_DT is NOT SWEEP_DT_DIV
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SWEEP_INTEG=cmf SWEEP_C_PROP=diagonalize SWEEP_B_PROP=diagonalize
export SWEEP_LEG_TYPE=sqiswap
export MEANFIELD_SPEC=0
export SWEEP_DEV=cuda
MAN=scripts/manifest_holdrow_segB_gpu.txt
read -r LEG TAU_RAMP HOLDS DTDIV < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MAN")
if [ -z "${LEG:-}" ]; then echo "bad task id $SLURM_ARRAY_TASK_ID"; exit 1; fi
export LEG TAU_RAMP HOLDS MODE=segB
export SWEEP_DT_DIV="$DTDIV"
echo "[hr-segB-gpu] task=$SLURM_ARRAY_TASK_ID leg=$LEG tau_ramp=$TAU_RAMP holds=$HOLDS dt_div=$DTDIV"
python scripts/run_segment.py
rc=$?; echo "[hr-segB-gpu] exit=$rc"; exit $rc
