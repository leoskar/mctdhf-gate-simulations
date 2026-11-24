#!/bin/bash
#SBATCH --job-name=3e-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%x-%j.out
#
# Run one three_electrons module on one GPU. Submit from three_electrons/
# (mkdir -p logs first). The grid is FOURWELL_NUM_DVR from the submitting
# shell: 80 for gates_80pt, orbital_convergence and sbs_check; unset (64
# points) for operating_points, gate_timing, routed_gates and grid_convergence.
# The 80-point modules assert the grid, so a forgotten variable fails at once.
#
#     FOURWELL_NUM_DVR=80 sbatch slurm/gpu.sbatch -m gates_80pt.mf80 gate --leg w1w2 ...
#     sbatch slurm/gpu.sbatch -m gate_timing.timing
#
# Python: $PY if set, else the first python on PATH (with multi-species-mctdhf
# installed, see the README).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Grow the GPU allocation instead of taking 75% of the card up front.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
echo "FOURWELL_NUM_DVR=${FOURWELL_NUM_DVR:-unset (64 points)}"
exec "${PY:-python}" -u "$@"
