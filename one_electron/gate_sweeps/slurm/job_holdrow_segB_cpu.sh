#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=hr_segBc
#SBATCH -o logs/hrsegBc_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
# Segment B for the w2w3 hold row (tau_ramp = 34.1), norb=[4,4,4,4]. Genuine
# ramp-down from the segA checkpoint at t_c = tau_hold; each point costs
# 2*tau_ramp = 68.2 t.u. regardless of the hold. Holds are batched a few per
# task so process/JAX startup is amortised, as in job_holdrow_segB_w0w3.sh.
# Needs job_holdrow_segA.sh task 1 to have finished -- submit with
#   --dependency=afterok:<segA jobid>
# CPU takes the SHORT leg (17 x 68.2 t.u.) while the GPU works the long w0w1
# row; JAX_PLATFORMS=cpu is set explicitly so these never probe CUDA while the
# GPU array holds the device. Submit with %5: 5 x 6 threads = 30 of 36 cores,
# leaving the GPU task its 2.
# th=112.8 is deliberately absent: already confirmed as a full leg
# (results/norb4_confirm/n4_w2w3_tr34.1_th112.8.npz, F=0.99799), and
# figures/mf_fig_wide.py's n4_points() prefers the full-leg value anyway.
set -uo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
unset SWEEP_DT SWEEP_DEV          # SWEEP_DT is NOT SWEEP_DT_DIV
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SWEEP_INTEG=cmf SWEEP_C_PROP=diagonalize SWEEP_B_PROP=diagonalize
export SWEEP_LEG_TYPE=sqiswap
export MEANFIELD_SPEC=0
export JAX_PLATFORMS=cpu
MAN=scripts/manifest_holdrow_segB_cpu.txt
read -r LEG TAU_RAMP HOLDS DTDIV < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MAN")
if [ -z "${LEG:-}" ]; then echo "bad task id $SLURM_ARRAY_TASK_ID"; exit 1; fi
export LEG TAU_RAMP HOLDS MODE=segB
export SWEEP_DT_DIV="$DTDIV"
echo "[hr-segB-cpu] task=$SLURM_ARRAY_TASK_ID leg=$LEG tau_ramp=$TAU_RAMP holds=$HOLDS dt_div=$DTDIV"
python scripts/run_segment.py
rc=$?; echo "[hr-segB-cpu] exit=$rc"; exit $rc
