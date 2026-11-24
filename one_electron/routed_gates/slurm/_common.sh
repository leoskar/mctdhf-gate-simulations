# Shared environment for the routed-gate jobs. Sourced, never executed.
set -uo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# Never JAX_PLATFORMS=cuda: some imported modules pin the default platform to
# cpu, so a cuda-only backend list fails at import. The scripts select the GPU
# themselves (SWEEP_DEV=cuda).
unset JAX_PLATFORMS SWEEP_DT
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SWEEP_DEV=cuda SWEEP_DT_DIV=0.1

# PULSE_SET -> (tau_ramp, tau_hold) of each leg, in time units (1 t.u. = 15.92 ps).
#   best  the individually optimized schedules
#   sync  w0w1 and w2w3 share one schedule, so the two concurrent gates of the
#         parallel routing finish together; w1w2 keeps its best schedule
#   fast  shorter ramps on w1w2 and w2w3; used only for the time-step test
# exact_reference/scripts/exact_common.py parses the three case lines below.
case "${PULSE_SET:?PULSE_SET must be set to best|fast|sync}" in
  best) export TAU_W0W1=82.3,73.2  TAU_W1W2=57.2,95.4  TAU_W2W3=34.1,112.8 ;;
  fast) export TAU_W0W1=82.3,73.2  TAU_W1W2=8.9,137.1  TAU_W2W3=11.9,131.7 ;;
  sync) export TAU_W0W1=57.2,92.7  TAU_W1W2=57.2,95.4  TAU_W2W3=57.2,92.7 ;;
  *)    echo "PULSE_SET=$PULSE_SET is not best|fast|sync" >&2; exit 2 ;;
esac
# Schedule of the direct w0w3 gate. Not used by the routed circuits, but
# run_arm_decomp reads all four.
export TAU_W0W3=81.3,2160.7
export TAU_SQI12="$TAU_W1W2"        # read by run_route_par at import
echo "[env] PULSE_SET=$PULSE_SET w0w1=$TAU_W0W1 w1w2=$TAU_W1W2 w2w3=$TAU_W2W3"
