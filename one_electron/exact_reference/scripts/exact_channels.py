"""Table I: four-orbital MCTDH against the untruncated reference, per error channel.

For each gate the 16x16 matrices of the reference at two step sizes are
decomposed into leakage, misrouting and internal/collective phase error
(lib/error_channels.py), each scored against the gate's own sqrt(iSWAP) target
over all sixteen inputs. The reference channels are Richardson-extrapolated
for a fourth-order method, X(0) = X(dt) + [X(dt) - X(2dt)] / 15, and compared
with the MCTDH matrix stored in the compare_*.npz file of compare_M.py.

Columns: the extrapolated reference, MCTDH, their difference (the entries of
Table I) and the residual step error of the reference |X(dt) - X(0)|.

    python scripts/exact_channels.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exact_common as ec                     # noqa: E402  (puts lib/ on sys.path)
import error_channels as fpi                  # noqa: E402

RES = ec.RESULTS


def budget(M, U):
    x, fxc = fpi.best_gauge(U, M)
    zl, zr = fpi._zphases(x[:4]), fpi._zphases(x[4:8])
    Mg = (zl[:, None] * M) * zr[None, :]
    o = np.einsum("ij,ij->j", U.conj(), Mg)
    s = (np.abs(M) ** 2).sum(0)
    p = (np.abs(U) * np.abs(M)).sum(0)
    S = float(s.sum())
    F = lambda c: (S + c ** 2) / (16 * 17.0)
    b_lk, b_rt, b_in, Fb = F(np.sqrt(s).sum()), F(p.sum()), F(np.abs(o).sum()), F(abs(o.sum()))
    out = dict(F=Fb, infidelity=1 - Fb, leakage=1 - b_lk, misrouting=b_lk - b_rt,
               internal=b_rt - b_in, collective=b_in - Fb)
    out["phase"] = out["internal"] + out["collective"]
    return out


def exact_M(tag):
    return ec.to_bit_order(np.load(os.path.join(RES, tag, "M.npz"))["M_primary_cfg"])


def cmp_file(fine_tag, mctdh_tag):
    d = np.load(os.path.join(RES, f"compare_{fine_tag}_vs_{mctdh_tag}.npz"))
    return np.asarray(d["A"]), np.asarray(d["B"])


CASES = [  # name, pair, fine tag, coarse tag, MCTDH tag of the compare file
    ("w0w1", (0, 1), "leg_w0w1_best_x64_primary_dt0.125-5_strang", "leg_w0w1_best_x64_primary_dt0.25-5_strang", "best-w0w1"),
    ("w1w2", (1, 2), "leg_w1w2_best_x64_primary_dt0.125-5_strang", "leg_w1w2_best_x64_primary_dt0.25-5_strang", "best-w1w2"),
    ("w2w3", (2, 3), "leg_w2w3_best_x64_primary_dt0.125-5_strang", "leg_w2w3_best_x64_primary_dt0.25-5_strang", "best-w2w3"),
    ("w0w3", (0, 3), "leg_w0w3_best_x64_primary_dt0.25-5_strang", "leg_w0w3_best_x64_primary_dt0.5-5_strang", "k16-w0w3"),
    ("sync", (0, 3), "arm_parallel_sync_x64_dt0.25_o4_symz", "arm_parallel_sync_x64_dt0.5_o4_symz", "sync-arm-parallel-symz"),
]
KEYS = ["infidelity", "leakage", "misrouting", "phase", "internal", "collective"]

for name, pair, fine, coarse, mt in CASES:
    U = ec.target_sqiswap(pair)
    A, B = cmp_file(fine, mt)
    Af = exact_M(fine)
    assert np.allclose(A, Af, atol=1e-12), f"{name}: compare A != fine M.npz"
    Ac = exact_M(coarse)
    bf, bc, bm = budget(Af, U), budget(Ac, U), budget(B, U)
    bx = {k: bf[k] + (bf[k] - bc[k]) / 15 for k in KEYS}
    print(f"\n== {name}  target sqiSWAP{pair}")
    print(f"{'':12s}{'ref(dt->0)':>13s}{'MCTDH':>13s}{'ref-MCTDH':>13s}{'ref err':>11s}")
    for k in KEYS:
        print(f"{k:12s}{bx[k]:13.4e}{bm[k]:13.4e}{bx[k]-bm[k]:+13.2e}{abs(bf[k]-bx[k]):11.1e}")
