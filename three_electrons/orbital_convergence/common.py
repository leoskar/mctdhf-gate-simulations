"""Shared pieces of orbital_convergence: the grid guard of gates_80pt, and every
module-level output path of the imported packages redirected into
``orbital_convergence/results``.

Import this module FIRST in every orbital_convergence script. It imports ``gates_80pt.common``
(which imports ``mf_lib``, switching on x64, and sets its own redirections),
then reassigns:

    mf_lib.RESULTS_DIR         -> orbital_convergence/results/_mf_lib/      (catch-all)
    mf_lib.MF_STATE_DIR        -> orbital_convergence/results/mf_states80/
    exact_check.out_dir        -> orbital_convergence/results/_exact_check/<leg>/
    gates_80pt.common.RESULTS       -> orbital_convergence/results/   (mf80's output directory)
    gates_80pt.common.mf_state_path -> :func:`mf_state_path` below

``mf_lib.IDLE_PATH`` stays gates_80pt's ``register80_idle.npz``: it is read only
(``mf_lib.x_vectors`` takes the idle x from it). Inputs are read from
``gates_80pt/results`` by explicit path (:data:`G80_RESULTS`); nothing here
writes there.
"""
import os

from gates_80pt import common as G80  # noqa: I100  (x64 + gates_80pt redirections)

import numpy as np  # noqa: E402

L, EC, fw = G80.L, G80.EC, G80.fw
require_grid = G80.require_grid
save_json, load_json = G80.save_json, G80.load_json

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
G80_RESULTS = G80.RESULTS          # read-only inputs (captured before the
assert G80_RESULTS.endswith(os.path.join("gates_80pt", "results"))  # reassignment)

# ── the redirections ──────────────────────────────────────────────────────
L.RESULTS_DIR = os.path.join(RESULTS, "_mf_lib")
L.MF_STATE_DIR = os.path.join(RESULTS, "mf_states80")


def _ec_out_dir(leg):
    d = os.path.join(RESULTS, "_exact_check", leg)
    os.makedirs(d, exist_ok=True)
    return d


EC.out_dir = _ec_out_dir

# The rungs whose relaxed states gates_80pt already made (read-only inputs).
G80_RUNGS = {"w1w2": (14, 18, 20, 22), "w0w1": (14, 22),
               "w2w3": (14, 20, 22)}


def mf_state_path(leg, norb):
    """gates_80pt's ``mfstates80_<leg>_n<N>.npz`` for its own rungs (read only),
    orbital_convergence/results/ for every new one."""
    if norb in G80_RUNGS.get(leg, ()):
        return os.path.join(G80_RESULTS, f"mfstates80_{leg}_n{norb}.npz")
    return os.path.join(RESULTS, f"mfstates80_{leg}_n{norb}.npz")


G80.RESULTS = RESULTS
G80.mf_state_path = mf_state_path


def assert_writable(path):
    """Every write goes under orbital_convergence/."""
    p = os.path.abspath(path)
    assert p.startswith(HERE + os.sep), f"refusing to write outside orbital_convergence: {p}"
    return p


# ── Delta_80 ──────────────────────────────────────────────────────────────
DELTA80_REF = {"w1w2": 1.175341e-2}


def delta80(leg):
    """Delta_80 from gates_80pt/results/oppoints80.json, checked against the
    stored value."""
    r = load_json(os.path.join(G80_RESULTS, "oppoints80.json"))[leg]
    assert r["status"] == "ok", r["status"]
    D = float(r["final"]["Delta"])
    if leg in DELTA80_REF:
        assert abs(D - DELTA80_REF[leg]) < 5e-9, (D, DELTA80_REF[leg])
    return D


# The state-averaging weights of the four pair states, normalised.
W4 = np.asarray(L.RELAX_WEIGHTS, dtype=np.float64)
W4 = W4 / W4.sum()
