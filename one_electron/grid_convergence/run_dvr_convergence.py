"""DVR grid convergence of the two-well FCI reference (App. B 3).

Lowest-6 eigenvalues of the interacting two-well grid Hamiltonian at the idle and
resonant configs, for N in {32, 48, 64, 96}: the production N=64 grid is
converged.

    python run_dvr_convergence.py        (from one_electron/grid_convergence/)
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # numpy/scipy only; keep JAX off GPU

import numpy as np

import grid_lib as gl
import td_fci as tf

N_LIST = [32, 48, 64, 96]
K = 6
X_IDLE = np.asarray(tf.X_IDLE_DEFAULT, dtype=float)
X_RES = np.asarray(tf.X_RES_DEFAULT, dtype=float)
CONFIGS = {"idle": X_IDLE, "res": X_RES}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", tf.GEOM_TAG, "dvr_convergence.npz")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def main():
    print(f"X_IDLE = {np.round(X_IDLE, 4)}")
    print(f"X_RES  = {np.round(X_RES, 4)}\n")

    evals = {cfg: np.zeros((len(N_LIST), K)) for cfg in CONFIGS}
    for cfg, x4 in CONFIGS.items():
        print(f"=== config: {cfg} ===")
        for i, N in enumerate(N_LIST):
            grid = gl.build_grid(N)
            w, _ = gl.fci_eigs(x4, grid, k=K)
            evals[cfg][i] = w
            print(f"  N={N:3d}: {np.round(w, 5)}")
        # convergence relative to finest grid
        ref = evals[cfg][-1]
        print(f"  --- |E_k(N) - E_k(N={N_LIST[-1]})| (lowest 4) ---")
        for i, N in enumerate(N_LIST[:-1]):
            d = np.abs(evals[cfg][i, :4] - ref[:4])
            rel = d / np.abs(ref[:4])
            print(f"  N={N:3d}: abs={np.array2string(d, precision=2, floatmode='maxprec')}"
                  f"  rel={np.array2string(rel, precision=2, floatmode='maxprec')}")
        print()

    np.savez(OUT, N_list=np.array(N_LIST), K=K,
             evals_idle=evals["idle"], evals_res=evals["res"],
             x_idle=X_IDLE, x_res=X_RES)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
