"""Helpers for impl_locals.py: scoring, single-qubit layers, circuit schedules
and the 4x4-corner embedding used by the reduced (product) models of the
routed circuits. Nothing here propagates."""
import os, numpy as np


# ── scoring (as run_route_sa.score / run_arm_decomp._score_matrix) ───────────
_S2 = 1.0 / np.sqrt(2.0)
_SQ2 = np.array([[1, 0, 0, 0], [0, _S2, 1j*_S2, 0],
                 [0, 1j*_S2, _S2, 0], [0, 0, 0, 1]], dtype=complex)
BITS = np.array([[(i >> (3 - k)) & 1 for k in range(4)] for i in range(16)])


def target_sqiswap_03():
    U = np.zeros((16, 16), complex)
    for i in range(16):
        q = [(i >> (3 - k)) & 1 for k in range(4)]
        for li in range(4):
            q0o, q3o = (li >> 1) & 1, li & 1
            amp = _SQ2[2*q[0] + q[3], 2*q0o + q3o]
            if amp != 0:
                o = (q0o << 3) | (q[1] << 2) | (q[2] << 1) | q3o
                U[o, i] = amp
    return U


def fxc_4q(U_ref, M, restarts=40, seed=0):
    from scipy.optimize import minimize
    if not np.all(np.isfinite(M)):
        return np.nan
    nref = np.vdot(U_ref, U_ref).real
    nm = np.vdot(M, M).real
    Uc = U_ref.conj().T

    def neg(x):
        zl = np.exp(1j * (BITS @ x[:4]))
        zr = np.exp(1j * (BITS @ x[4:8]))
        t = np.trace(Uc @ ((zl[:, None] * M) * zr[None, :]))
        return -(abs(t) ** 2) / (nref * nm)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        r = minimize(neg, rng.uniform(0, 2*np.pi, 8), method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-11, maxiter=8000))
        if best is None or r.fun < best.fun:
            best = r
    return float(-best.fun)


U_TARGET = target_sqiswap_03()


def score(M):
    fxc = fxc_4q(U_TARGET, M)
    d = 16
    return ((np.vdot(M, M).real
             + fxc * np.vdot(U_TARGET, U_TARGET).real * np.vdot(M, M).real)
            / (d * (d + 1))), fxc


# ── schedule algebra (as in run_arm_decomp) ──────────────────────────────────
HAD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
SDG = np.diag([1.0, -1j]).astype(complex)
BLOCK_U2 = HAD @ SDG


def block_layer(pair):
    return {pair[0]: BLOCK_U2.copy(), pair[1]: BLOCK_U2.copy()}


def zframe(p, pair, side):
    a, b = (p[0], p[1]) if side == "after" else (p[2], p[3])
    return {pair[0]: np.diag([1.0, np.exp(1j * a)]).astype(complex),
            pair[1]: np.diag([1.0, np.exp(1j * b)]).astype(complex)}


def layer16_ideal(assign):
    U = np.eye(1, dtype=complex)
    for s in range(4):
        U = np.kron(U, assign.get(s, np.eye(2, dtype=complex)))
    return U


def swap_block(key, pair, z):
    return [("layer", block_layer(pair)), ("zframe", zframe(z, pair, "before")),
            ("leg", key), ("zframe", zframe(z, pair, "after"))]


def entangler(key, pair, z):
    return [("zframe", zframe(z, pair, "before")), ("leg", key),
            ("zframe", zframe(z, pair, "after"))]


def par_block(z01, z23):
    return [("layer", {**block_layer((0, 1)), **block_layer((2, 3))}),
            ("zframe", {**zframe(z01, (0, 1), "before"),
                        **zframe(z23, (2, 3), "before")}),
            ("leg", "par"),
            ("zframe", {**zframe(z01, (0, 1), "after"),
                        **zframe(z23, (2, 3), "after")})]


def coalesce(sched):
    out = []
    for kind, val in sched:
        if kind in ("layer", "zframe") and out and out[-1][0] == "local":
            merged = dict(out[-1][1])
            for w, u in val.items():
                merged[w] = u @ merged[w] if w in merged else u
            out[-1] = ("local", merged)
        elif kind in ("layer", "zframe"):
            out.append(("local", dict(val)))
        else:
            out.append((kind, val))
    return out


def serial_schedule(z):
    s = []
    for key, pair in (("w2w3", (2, 3)), ("w1w2", (1, 2))):
        for _ in range(3):
            s += swap_block(key, pair, z[key])
    s += entangler("w0w1", (0, 1), z["w0w1"])
    for key, pair in (("w1w2", (1, 2)), ("w2w3", (2, 3))):
        for _ in range(3):
            s += swap_block(key, pair, z[key])
    return s


def parallel_schedule(z01, z23, zent):
    s = []
    for _ in range(3):
        s += par_block(z01, z23)
    s += entangler("w1w2", (1, 2), zent)
    for _ in range(3):
        s += par_block(z01, z23)
    return s


# ── the new rung: 4x4 corner, spectators assumed inert ────────────────────────
def corner4(M16, pair):
    a, b = pair
    idx = []
    for j in range(4):
        na, nb = j & 1, (j >> 1) & 1
        bits = [0, 0, 0, 0]
        bits[a], bits[b] = na, nb
        idx.append((bits[0] << 3) | (bits[1] << 2) | (bits[2] << 1) | bits[3])
    return M16[np.ix_(idx, idx)]


def embed4(M4, pair):
    """M4 (pair basis, index = n_a + 2 n_b) -> 16x16 bit-order, identity on the
    spectators.  This IS the assumption being tested: the gate is taken to act
    on its pair alone, with the spectator occupations carried through unchanged
    and unentangled."""
    a, b = pair
    out = np.zeros((16, 16), complex)
    for i in range(16):
        bi = [(i >> (3 - k)) & 1 for k in range(4)]
        for j in range(16):
            bj = [(j >> (3 - k)) & 1 for k in range(4)]
            if any(bi[s] != bj[s] for s in range(4) if s not in pair):
                continue
            out[i, j] = M4[bi[a] + 2*bi[b], bj[a] + 2*bj[b]]
    return out


PAIR = {"w0w1": (0, 1), "w1w2": (1, 2), "w2w3": (2, 3)}
