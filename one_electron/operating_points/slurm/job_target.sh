#!/bin/bash
#SBATCH --job-name=target
#SBATCH -o logs/%x.out
#SBATCH -n 4
#SBATCH --gres=gpu:1
# One target operating point (Table S2), spectators frozen at the idle point.
# Arguments: <loss_name> <idle stem>. Submitted by submit_targets.sh.
mkdir -p logs
python optimize_qubit_params_frozen.py "$SLURM_JOB_NAME" "$1" "$2"
