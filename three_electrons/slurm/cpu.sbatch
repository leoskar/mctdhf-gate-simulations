#!/bin/bash
#SBATCH --job-name=3e-cpu
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm-%x-%j.out
#
# Run one three_electrons module on the CPU. Submit from three_electrons/
# (mkdir -p logs first). The grid is FOURWELL_NUM_DVR from the submitting
# shell: 80 for gates_80pt, orbital_convergence and sbs_check; unset (64
# points) for operating_points, gate_timing, routed_gates and grid_convergence.
# The 80-point modules assert the grid, so a forgotten variable fails at once.
#
#     FOURWELL_NUM_DVR=80 sbatch slurm/cpu.sbatch -m gates_80pt.mf80 gate --leg w1w2 ...
#     sbatch slurm/cpu.sbatch -m gate_timing.timing
#
# Python: $PY if set, else the first python on PATH (with multi-species-mctdhf
# installed, see the README).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
echo "FOURWELL_NUM_DVR=${FOURWELL_NUM_DVR:-unset (64 points)}"
exec "${PY:-python}" -u "$@"
