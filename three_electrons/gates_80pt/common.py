"""gates_80pt shared pieces: the grid guard, and every module-level output path of
the imported packages redirected into ``gates_80pt/results``.

Import this module FIRST in every gates_80pt script (after nothing but the
standard library): it imports ``mf_lib`` (which switches on x64) and then
reassigns the paths below, so no imported function can write into another
package's directories.

    mf_lib.IDLE_PATH     -> results/register80_idle.npz
    mf_lib.MF_STATE_DIR  -> results/mf_states80/   (only via ``mf_state_path``)
    mf_lib.RESULTS_DIR   -> results/_mf_lib/       (catch-all)
    exact_check.out_dir  -> results/_exact_check/<leg>/

The grid is set by ``FOURWELL_NUM_DVR`` in the job script, before python
starts; :func:`require_grid` asserts it.
"""
import json
import os

from gate_timing import mf_lib as L  # noqa: E402  (switches on x64)
from gate_timing import exact_check as EC  # noqa: E402
from system import fourwell as fw  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# ── the redirections ──────────────────────────────────────────────────────
_OLD_IDLE = L.IDLE_PATH
REGISTER80 = os.path.join(RESULTS, "register80_idle.npz")
L.IDLE_PATH = REGISTER80
L.MF_STATE_DIR = os.path.join(RESULTS, "mf_states80")
L.RESULTS_DIR = os.path.join(RESULTS, "_mf_lib")


def _ec_out_dir(leg):
    d = os.path.join(RESULTS, "_exact_check", leg)
    os.makedirs(d, exist_ok=True)
    return d


EC.out_dir = _ec_out_dir

# 64-point files: never to be loaded into an 80-point solver.
OLD_IDLE_64 = _OLD_IDLE
LEGS3 = {"w0w1": (0, 1), "w1w2": (1, 2), "w2w3": (2, 3)}

# The 64-point references.
REF64 = dict(
    idle_loss=0.0090, idle_minP=0.9985,
    detunings=[1.711961, 2.540340, 3.166947, 4.412671],
    pA={"w0w1": 0.548, "w1w2": 0.555, "w2w3": 0.548},
    Delta={"w0w1": 1.41593e-2, "w1w2": 1.17467e-2, "w2w3": 1.12536e-2},
    T_eff={"w0w1": 110.9, "w1w2": 135.6, "w2w3": 139.6},
    ramp={"w0w1": 12.6, "w1w2": 9.4, "w2w3": 8.0},
    ladder_F4={14: 0.6721, 18: 0.66198, 20: 0.64290, 22: 0.63590},
    ladder_ret11={14: 0.107, 18: 0.090, 20: 0.049, 22: 0.038},
)


def require_grid(n=80):
    got = fw.NUM_DVR
    env = os.environ.get("FOURWELL_NUM_DVR")
    assert got == n, (f"fourwell.NUM_DVR = {got} (FOURWELL_NUM_DVR={env!r}); "
                      f"this script needs {n}")
    assert all(len(x) == n for x in fw.X_S)
    dx_nm = float(fw.X_S[0][1] - fw.X_S[0][0]) * fw.LENGTH_UNIT_UM * 1e3
    print(f"  grid: {n} sine-DVR points per well, dx = {dx_nm:.3f} nm",
          flush=True)
    return dx_nm


def mf_state_path(leg, norb):
    """results/mfstates80_<leg>_n<N>.npz."""
    return os.path.join(RESULTS, f"mfstates80_{leg}_n{norb}.npz")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, complex):
            return [o.real, o.imag]
        raise TypeError(type(o))
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def x_scaled(leg):
    """(x_idle, x_target) in code units (GHz / 10), from the 64-point files'
    x (grid-independent). Reads the 64-point idle file's x_ghz explicitly so
    it works before the 80-point register exists."""
    import numpy as _np
    with _np.load(OLD_IDLE_64) as d:
        xi = _np.asarray(d["x_ghz"], dtype=_np.float64)
    with _np.load(L.target_path(leg), allow_pickle=True) as d:
        xg = _np.asarray(d["best_x"], dtype=_np.float64)
    return xi / fw.TYPICAL_FREQ_GHZ, xg / fw.TYPICAL_FREQ_GHZ
