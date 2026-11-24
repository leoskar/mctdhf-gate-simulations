"""Rebuild every rung of the ladder with the SAME single-qubit gates.

The three rungs differ only in how the two-qubit legs are represented.  The
single-qubit gates between them must therefore be treated identically on all
three, and the only treatment the propagation can use is the real one: a
one-body operator U = I - QQ^H + Q u2 Q^H on the well's two qubit orbitals.

The product models were previously built with perfect 2x2 rotations, which made
part of the model-vs-propagation gap an artifact of that idealisation
(+0.027..+0.070 in Fbar16).  This script computes each single-qubit layer's own
16x16 matrix as actually applied -- NO time propagation, the layer only rotates
the shared orbitals -- and rebuilds both product rungs with it.

Self-test: rebuilding the 16x16-leg rung this way must reproduce the stored
Fbar_prod_impl of each arm.

Run once per pulse set used in the paper:
    PULSE_SET=best python scripts/impl_locals.py
    PULSE_SET=sync python scripts/impl_locals.py
Writes results/impl_locals_<set>.json and results/prodmats_<set>_<arm>.npz.
"""
import os, sys, json
import numpy as np

SET = os.environ["PULSE_SET"]
TAUS = {"best": dict(TAU_W0W1="82.3,73.2", TAU_W1W2="57.2,95.4",
                     TAU_W2W3="34.1,112.8", TAU_W0W3="81.3,2160.7"),
        "fast": dict(TAU_W0W1="82.3,73.2", TAU_W1W2="8.9,137.1",
                     TAU_W2W3="11.9,131.7", TAU_W0W3="81.3,2160.7"),
        "sync": dict(TAU_W0W1="57.2,92.7", TAU_W1W2="57.2,95.4",
                     TAU_W2W3="57.2,92.7", TAU_W0W3="81.3,2160.7")}[SET]
os.environ.update(TAUS)
os.environ["PULSE_SET"] = SET
os.environ.setdefault("SWEEP_DEV", "cpu")
os.environ.setdefault("ARM", "legs16")
os.environ.setdefault("SWEEP_DT_DIV", "0.1")   # no propagation here; dt is inert

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_arm_decomp as A          # noqa: E402  (env must be set first)
from circuit_lib import corner4, embed4, score, PAIR   # noqa: E402

R = os.path.abspath(os.path.join(HERE, "..", "results"))
# best and sync use the deterministic frames of scripts/canonical_frames.py
LEGS = {"best": "legs16_symz.npz", "sync": "legs16_symz.npz", "fast": "legs16.npz"}[SET]
STORE = np.load(os.path.join(R, f"arms_{SET}", LEGS), allow_pickle=True)

ARMS_IN_SET = {"best": ["serial", "parallel"], "fast": ["serial", "parallel"],
               "sync": ["parallel"]}[SET]
FILES = {("best", "serial"): "arms_best/arm_serial_dt0.25_symz_20261027_140054.npz",
         ("best", "parallel"): "arms_best/arm_parallel_dt0.25_symz_20261027_163440.npz",
         ("fast", "serial"): "arms_fast/arm_serial_dt0.25_20260820_163140.npz",
         ("fast", "parallel"): "arms_fast/arm_parallel_dt0.25_20260821_010413.npz",
         ("sync", "parallel"): "arms_sync/arm_parallel_dt0.25_symz_20261027_185125.npz"}


def main():
    C16, bs0 = A.load_anchors16()
    QB = {s: A.qubit_basis16(s, C16, bs0)[0] for s in range(A.NUM_WELLS)}
    solver = A.make_solver_single("w1w2")          # any solver: no propagation
    y0 = A.pack_Cb_to_y(A.jnp.stack(C16),
                        [A.jnp.asarray(b, dtype=A.jnp.complex128) for b in bs0])

    M16 = {k: STORE[f"M16_{k}"] for k in ("w0w1", "w1w2", "w2w3", "par")}
    M44 = {k: embed4(corner4(M16[k], PAIR[k]), PAIR[k])
           for k in ("w0w1", "w1w2", "w2w3")}
    # scalar correction for the concurrent leg: its (0,1) corner is taken with
    # wells 2,3 in |00> while those wells are ALSO being driven, so it already
    # carries the (2,3) gate's |00>->|00> amplitude -- and vice versa.  The
    # bare product applies M[0,0] twice; dividing by it restores the correct
    # A(x)B factorisation (verified: corrected[0,0] == M16_par[0,0] exactly).
    M44["par"] = (embed4(corner4(M16["par"], (0, 1)), (0, 1))
                  @ embed4(corner4(M16["par"], (2, 3)), (2, 3))
                  ) / M16["par"][0, 0]

    cache, out = {}, {}
    for arm in ARMS_IN_SET:
        if arm == "serial":
            z = {k: STORE[f"z_{k}_{PAIR[k][0]}{PAIR[k][1]}"] for k in PAIR}
            sched = A.coalesce(A.serial_schedule(z))
        else:
            sched = A.coalesce(A.parallel_schedule(
                STORE["z_par_01"], STORE["z_par_23"], STORE["z_w1w2_12"]))

        P16 = np.eye(16, dtype=complex)     # rung 2: full 16x16 legs
        P44 = np.eye(16, dtype=complex)     # rung 1: 4x4 corners (x) 1
        for kind, val in sched:
            if kind == "local":
                key = tuple(sorted((w, np.round(u, 12).tobytes())
                                   for w, u in val.items()))
                if key not in cache:
                    cache[key] = A.layer16_implemented(solver, y0, val, QB,
                                                       C16, bs0)
                    print(f"  layer {len(cache):2d}: ||L||^2 = "
                          f"{np.vdot(cache[key], cache[key]).real:.5f}",
                          flush=True)
                L = cache[key]
            else:
                L = None
            if L is not None:
                P16, P44 = L @ P16, L @ P44
            else:
                P16, P44 = M16[val] @ P16, M44[val] @ P44

        d = np.load(os.path.join(R, FILES[(SET, arm)]), allow_pickle=True)
        err = float(np.abs(P16 - d["M_prod_impl"]).max())
        F16, _ = score(P16)
        F44, _ = score(P44)
        print(f"\n[{SET}/{arm}] SELF-TEST max|dM| vs stored M_prod_impl = "
              f"{err:.2e}   {'OK' if err < 1e-10 else 'FAIL'}")
        print(f"[{SET}/{arm}] rung1 4x4(x)1 = {F44:.6f}   "
              f"rung2 16x16 = {F16:.6f}   rung3 propagation = "
              f"{float(d['Fbar16']):.6f}")
        out[arm] = dict(F_c4_impl=F44, F_16_impl=F16,
                        F_prop=float(d["Fbar16"]), selftest=err)
        # keep the matrices: downstream figures decompose them by channel
        np.savez(os.path.join(R, f"prodmats_{SET}_{arm}.npz"),
                 M_prod_c4=P44, M_prod_16=P16, M_prop=np.asarray(d["M"]),
                 selftest=err)

    with open(os.path.join(R, f"impl_locals_{SET}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(R, f"impl_locals_{SET}.json"))


if __name__ == "__main__":
    main()
