"""Print Tables S1 and S2 of the supplementary material (one electron per well).

Reads the optimizer output in ITE_trajectories/. No propagation: the stored
`best_y` is the converged state at the optimum, already rotated to the
state-averaged natural orbitals, so the qubit populations F[k, n] follow
directly from its configuration coefficients.

    python print_tables.py
"""
import importlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)

import optimize_qubit_params as oqp          # builds the 50-state solver
from utils import qubit_corner_fidelities, iter_bitstrings
from losses._superpos import W_SUPER, WEIGHTS_BY_CFG_IDX

ITE = os.path.join(HERE, "ITE_trajectories")
IDLE = "new_geometry_step1_stage1_singles_ground"
TARGETS = {(0, 1): "chain_frozen_20260621_140848_step2_stage2_superpos_w0w1",
           (1, 2): "chain_frozen_20260621_140848_step3_stage3_superpos_w1w2",
           (2, 3): "chain_frozen_20260621_140848_step4_stage4_superpos_w2w3",
           (0, 3): "chain_frozen_20260621_140848_step5_stage5_superpos_w0w3"}
CONFIGS = iter_bitstrings(4, 2)


def populations(d):
    C, bs = oqp.MCTDHF._y_to_Cb(np.asarray(d["best_y"]))
    return np.asarray(qubit_corner_fidelities(C, 4, n_levels=2))


d = np.load(os.path.join(ITE, IDLE + ".npz"))
x = d["best_x"]
print("Table S1 -- idle operating point")
print("  omega/2pi (GHz):", np.round(x[:4], 2))
print("  beta/2pi  (GHz):", np.round(x[4:], 3))
print(f"  L_idle = {float(d['best_loss']):.3f}   lowest population "
      f"{float(np.min(d['fidelities'][np.argmin(d['losses'])])):.4f}")

print("\nTable S2 -- target operating points (active wells i, j)")
for (i, j), stem in TARGETS.items():
    d = np.load(os.path.join(ITE, stem + ".npz"))
    loss_mod = importlib.import_module("losses." + str(d["loss_name"]))
    x, F = d["best_x"], populations(d)
    A, B = loss_mod.A, loss_mod.B            # columns of |1_i> and |1_j>
    res = loss_mod.compute(C_final=None, bs_final=None, F=F, energies=None,
                           num_wells=4)
    assert abs(res["loss"] - float(d["best_loss"])) < 1e-5
    # k_i, k_j: the two eigenstates the cost function assigns to the active
    # pair. Stages 2-4 order the wells by frequency, so these are the
    # eigenstates at positions A and B. Stage 5 does not; there they are the
    # two eigenstates left after the assignment of the other 14 qubit states,
    # ordered by their |1_i>/|1_j> character, as in its cost function.
    if (i, j) != (0, 3):
        ki, kj = A, B
    else:
        others = [n for n in range(16) if n not in (A, B)]
        used = set(int(k) for k in res["matched_idx"][others])
        left = np.array([k for k in range(F.shape[0]) if k not in used])
        ki, kj = left[np.argsort(-(F[left, A] + F[left, B]))[:2]]
    l_active = W_SUPER * ((F[[ki, kj]][:, [A, B]] - 0.5) ** 2).sum()
    # qubit states with exactly one active well and >= 1 spectator excited
    mixed = [n for n, c in enumerate(CONFIGS)
             if c[i] + c[j] == 1 and sum(c) - c[i] - c[j] >= 1]
    f = res["fidelities"][mixed]
    print(f"  ({i},{j}): omega_i {x[i]:.2f}  omega_j {x[j]:.2f}  beta_i {x[4+i]:.3f}"
          f"  beta_j {x[4+j]:.3f}  L_target {res['loss']:.3f}")
    print(f"         P_i->i {F[ki, A]:.3f}  P_i->j {F[ki, B]:.3f}  "
          f"P_j->i {F[kj, A]:.3f}  P_j->j {F[kj, B]:.3f}")
    share = float((WEIGHTS_BY_CFG_IDX[mixed] * (1 - f)).sum())
    print(f"         L_active {l_active:.3f}  L_spectator {res['loss'] - l_active:.3f}")
    print(f"         the six states with one active and >= 1 spectator excitation"
          f" keep {f.min():.3f}-{f.max():.3f} and contribute {share:.3f} to L_spectator")
