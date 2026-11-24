# Shared environment for the exact-reference jobs. Sourced, never executed.
set -uo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
unset JAX_PLATFORMS
if [ "${DEV:-cuda}" = "cpu" ]; then
  export JAX_PLATFORMS=cpu EXACT_DEV=cpu
else
  # Never JAX_PLATFORMS=cuda: some imported modules pin the default platform
  # to cpu. The device is selected in the scripts (EXACT_DEV).
  export EXACT_DEV=cuda
  # One preallocated pool; tuned for the 8 GB GPU that was used.
  export XLA_PYTHON_CLIENT_PREALLOCATE=true
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.78
fi
