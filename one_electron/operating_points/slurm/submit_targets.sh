#!/usr/bin/env bash
# The four target operating points (Table S2). Each optimizes the two active
# wells with the spectators frozen at the idle point; the active wells are kept
# 2 GHz from the spectators for the nearest-neighbour pairs (stages 2-4) and
# left free for the pair (0,3) (stage 5). The four jobs are independent.
#
# Usage, from one_electron/operating_points/:
#   ./slurm/submit_targets.sh [<base_name>] [<idle_stem>]
# The published runs used base name chain_frozen_20260621_140848.
set -euo pipefail
BASE="${1:-chain_frozen_$(date +%Y%m%d_%H%M%S)}"
IDLE="${2:-new_geometry_step1_stage1_singles_ground}"
step=2
for LOSS in stage2_superpos_w0w1 stage3_superpos_w1w2 stage4_superpos_w2w3 stage5_superpos_w0w3; do
  sbatch -J "${BASE}_step${step}" slurm/job_target.sh "$LOSS" "$IDLE"
  step=$((step + 1))
done
