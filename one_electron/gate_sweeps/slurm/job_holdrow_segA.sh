#!/bin/bash
# Submit from one_electron/gate_sweeps/ (mkdir -p logs first).
#SBATCH --job-name=hr_segA
#SBATCH -o logs/hrsegA_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# Segment A at norb=[4,4,4,4] for the w0w1 and w2w3 hold rows of Fig. 4 (w0w3:
# job_holdrow_segA_w0w3.sh). Each leg's row sits at its own best ramp:
#   w0w1 tau_ramp = 82.3   (MF maximum, also the norb=4 maximum)
#   w2w3 tau_ramp = 34.1   (ditto, by 2e-5 over 57.9)
# delta = tau_ramp (erfc(9) tail => exact reuse), so segA only has to checkpoint
# out to t_end = th_max: 99.3 t.u. for w0w1, 181.6 for w2w3. Cheap -- the saving
# is on the segB side, where each hold costs 2*tau_ramp instead of a full leg.
# Unlike the w0w3 row these legs run at 10dt (SWEEP_DT_DIV=0.1), same as every
# other adjacent-leg point of the sweep.
set -uo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset SWEEP_DT JAX_PLATFORMS          # SWEEP_DT is NOT SWEEP_DT_DIV
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SWEEP_INTEG=cmf SWEEP_C_PROP=diagonalize SWEEP_B_PROP=diagonalize
export SWEEP_LEG_TYPE=sqiswap
export MEANFIELD_SPEC=0               # full orbitals -- the whole point
export SWEEP_DEV=cuda
MAN=scripts/manifest_holdrow_segA.txt
read -r LEG TAU_RAMP TH_MAX DTDIV < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MAN")
if [ -z "${LEG:-}" ]; then echo "bad task id $SLURM_ARRAY_TASK_ID"; exit 1; fi
export LEG TAU_RAMP TH_MAX MODE=segA
export SWEEP_DT_DIV="$DTDIV"
echo "[hr-segA] task=$SLURM_ARRAY_TASK_ID leg=$LEG tau_ramp=$TAU_RAMP th_max=$TH_MAX dt_div=$DTDIV"
python scripts/run_segment.py
rc=$?; echo "[hr-segA] exit=$rc"; exit $rc
