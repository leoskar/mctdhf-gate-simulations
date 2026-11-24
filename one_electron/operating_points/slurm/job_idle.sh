#!/bin/bash
#SBATCH --job-name=new_geometry_step1
#SBATCH -o logs/%x.out
#SBATCH -n 4
#SBATCH --gres=gpu:1
# Idle operating point (Table S1). Submit from one_electron/operating_points/:
#   sbatch slurm/job_idle.sh
# Writes ITE_trajectories/new_geometry_step1_stage1_singles_ground.npz
mkdir -p logs
python optimize_qubit_params.py "$SLURM_JOB_NAME" stage1_singles_ground
