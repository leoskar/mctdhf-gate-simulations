#!/bin/bash
# Comparison with MCTDH, Richardson extrapolation and the error channels of
# Table I, from the stored M.npz files. CPU only, a few minutes.
# Run from one_electron/exact_reference/.
set -euo pipefail
export JAX_PLATFORMS=cpu EXACT_DEV=cpu PULSE_SET=best
for leg in w0w1 w1w2 w2w3; do
  T=leg_${leg}_best_x64_primary
  TAG_A=${T}_dt0.125-5_strang MCTDH=best:$leg python scripts/compare_M.py
  # three-point check: the limits from (0.5, 0.25) and (0.25, 0.125) agree
  TAG_FINE=${T}_dt0.125-5_strang TAG_COARSE=${T}_dt0.25-5_strang \
    TAG_COARSEST=${T}_dt0.5-5_strang MCTDH=best:$leg python scripts/richardson.py
done
T=leg_w0w3_best_x64_primary
TAG_A=${T}_dt0.25-5_strang MCTDH=k16:w0w3 python scripts/compare_M.py
TAG_FINE=${T}_dt0.25-5_strang TAG_COARSE=${T}_dt0.5-5_strang \
  MCTDH=k16:w0w3 python scripts/richardson.py
T=arm_parallel_sync_x64
TAG_A=${T}_dt0.25_o4_symz MCTDH=sync:arm:parallel:symz python scripts/compare_M.py
TAG_FINE=${T}_dt0.25_o4_symz TAG_COARSE=${T}_dt0.5_o4_symz \
  MCTDH=sync:arm:parallel:symz python scripts/richardson.py
python scripts/exact_channels.py
