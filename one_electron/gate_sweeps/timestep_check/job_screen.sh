#!/bin/bash
# Timestep check of the CMF integrator on the sqrt(iSWAP)_12 leg (163.7 t.u.).
# Run from one_electron/gate_sweeps/timestep_check/, e.g.
#   LEG=sqi12 INTEG=cmf N_CHUNKS=1637 SCREEN_DEV=cuda DT_DIV=1   sbatch job_screen.sh
#   LEG=sqi12 INTEG=cmf N_CHUNKS=1637 SCREEN_DEV=cuda DT_DIV=0.1 sbatch job_screen.sh
# DT_DIV multiplies the base step 0.005 t.u. by 1/DT_DIV (0.1 -> 0.05 t.u. = 0.80 ps).
set -euo pipefail
# GPU runs must not set JAX_PLATFORMS: imported modules pin 'cpu', so a cuda-only
# backend list makes them fail. The script selects the device itself.
if [ "${SCREEN_DEV:-cpu}" = "cpu" ]; then export JAX_PLATFORMS=cpu; fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
python diag_integ_screen.py
