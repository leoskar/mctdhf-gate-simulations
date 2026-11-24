"""Error channels of a 16x16 gate matrix scored against sqrt(iSWAP)_03 (x) I.

The analysis functions of the per-input decomposition, used by the figure
scripts (Figs. 4 and 5) and by exact_reference/scripts/exact_channels.py:

  best_gauge        the Fbar-optimal per-qubit Z rotations before and after
  decompose         Fbar_16 split into leakage, misrouting, internal and
                    collective ("inter") phase error
  block_budgets     the same split inside each 4x4 block of fixed spectators
  block_fidelities  Fbar_4 of each spectator block

Bit order: basis index q0<<3 | q1<<2 | q2<<1 | q3.
"""
from __future__ import annotations

import numpy as np

_S2 = 1.0 / np.sqrt(2.0)
_SQ2 = np.array([[1, 0, 0, 0], [0, _S2, 1j * _S2, 0],
                 [0, 1j * _S2, _S2, 0], [0, 0, 0, 1]], dtype=complex)


def target_sqiswap_03():
    """Ideal sqiSWAP(w0,w3) (x) I(w1,w2) in bit order q0<<3|q1<<2|q2<<1|q3."""
    U = np.zeros((16, 16), complex)
    for i in range(16):
        q = [(i >> (3 - k)) & 1 for k in range(4)]
        for li in range(4):
            q0o, q3o = (li >> 1) & 1, li & 1
            amp = _SQ2[2 * q[0] + q[3], 2 * q0o + q3o]
            if amp != 0:
                o = (q0o << 3) | (q[1] << 2) | (q[2] << 1) | q3o
                U[o, i] = amp
    return U


def _zphases(thetas):
    bits = np.array([[(i >> (3 - k)) & 1 for k in range(4)] for i in range(16)])
    return np.exp(1j * (bits @ np.asarray(thetas)))


def best_gauge(U, M, restarts=12, seed=0, sweeps=400, tol=1e-15):
    """The Fbar-optimal per-qubit Z gauge, by closed-form coordinate ascent.

    Writing Tr(U^dag Z_L M Z_R) = sum_ij A_ij z_i w_j with A = conj(U)*M, the
    dependence on ONE angle is affine in its phase: splitting the rows (or
    columns) carrying that bit gives Tr = R0 + exp(i t) R1, maximised in closed
    form at t = arg(R0) - arg(R1) with value |R0| + |R1|. Sweeping the 2*nq
    angles is therefore monotone and converges in a few passes; the restarts
    only guard against local maxima.

    This replaces a 40-restart Nelder-Mead search (which was both slow and not
    guaranteed to find the global maximum). Validated against the pinned
    implementation in lib/gate_fidelity.py: identical to 9 decimals on the
    16x16 register gauge and on all four 4x4 spectator blocks of both routed
    runs, at ~50x the speed (45 ms vs 2.2 s for the 16x16).

    Returns (angles, fxc) with fxc the NORMALISED |Tr|^2/(||U||^2 ||M||^2), the
    same convention the callers below expect.
    """
    d = U.shape[0]
    nq = int(round(np.log2(d)))
    A = U.conj() * M
    bits = np.array([[(i >> (nq - 1 - k)) & 1 for k in range(nq)]
                     for i in range(d)])
    rng = np.random.default_rng(seed)
    best_val, best_x = -1.0, None
    for _ in range(restarts):
        th = rng.uniform(0, 2 * np.pi, nq)
        ph = rng.uniform(0, 2 * np.pi, nq)
        prev, val = -1.0, -1.0
        for _s in range(sweeps):
            z, w = np.exp(1j * (bits @ th)), np.exp(1j * (bits @ ph))
            for k in range(nq):                     # left angles: split rows
                T = (A * w[None, :]).sum(1) * z
                m = bits[:, k] == 1
                R1 = T[m].sum() / np.exp(1j * th[k])
                th[k] = np.angle(T[~m].sum()) - np.angle(R1)
                z = np.exp(1j * (bits @ th))
            for k in range(nq):                     # right angles: split cols
                T = (A * z[:, None]).sum(0) * w
                m = bits[:, k] == 1
                R1 = T[m].sum() / np.exp(1j * ph[k])
                ph[k] = np.angle(T[~m].sum()) - np.angle(R1)
                w = np.exp(1j * (bits @ ph))
            val = abs(np.trace(U.conj().T @ ((z[:, None] * M) * w[None, :])))
            if val - prev < tol:
                break
            prev = val
        if val > best_val:
            best_val, best_x = val, np.concatenate([th, ph])
    fxc = best_val ** 2 / (np.vdot(U, U).real * np.vdot(M, M).real)
    return best_x, float(fxc)


def block_fidelities(M):
    """Fbar_4 in each of the four spectator blocks.

    The ideal gate is sqiSWAP(0,3) (x) I(1,2), so the register splits into four
    4-dimensional blocks labelled by the spectator setting (q1,q2), and the
    target acts IDENTICALLY in each. Scoring every block therefore asks the same
    question four times with different spectators, and the spread across blocks
    is the spectator-conditioned error, which a single register number
    cannot separate from ordinary gate error.

    Each block gets its own 2-qubit local-Z gauge, exactly as the reported
    (0,3)-corner number does, so block (0,0) reproduces it. Population that
    leaves a block is penalised through Tr(M_b M_b^dag) < 4.

    TWO gauges are reported, because they answer different questions:

      "own"      each block re-optimises its own 4 angles. This is the number
                 reported as the (0,3) corner (block (0,0)
                 reproduces it) and it is like-for-like with the direct arm,
                 which was also scored with its own 4-angle gauge. It is NOT
                 comparable to Fbar_16, since four independent 4-angle gauges
                 are strictly more freedom than one 8-angle gauge.
      "register" the block scored inside the SAME 8 angles that produce
                 Fbar_16. This is the gauge panel (a) is drawn in, so it is the
                 one consistent with the figure.

    Basis-order note: rows/cols come out as (q0q3) = 00,01,10,11 while
    gate_fidelity's convention is (00,10,01,11). Fbar is invariant under that
    relabelling because the sqiSWAP reference is symmetric in its two qubits and
    the gauge is maximised over both angles.
    """
    x, _ = best_gauge(target_sqiswap_03(), M)
    zl, zr = _zphases(x[:4]), _zphases(x[4:8])
    Mg = (zl[:, None] * M) * zr[None, :]

    own, reg = {}, {}
    for q1 in (0, 1):
        for q2 in (0, 1):
            idx = [(q0 << 3) | (q1 << 2) | (q2 << 1) | q3
                   for q0 in (0, 1) for q3 in (0, 1)]
            Mb = M[np.ix_(idx, idx)]
            nm = float(np.linalg.norm(Mb) ** 2)
            _, fxc = best_gauge(_SQ2, Mb)
            own[(q1, q2)] = (nm + fxc * 4.0 * nm) / 20.0
            coh = abs(np.trace(_SQ2.conj().T @ Mg[np.ix_(idx, idx)])) ** 2
            reg[(q1, q2)] = (nm + coh) / 20.0
    return own, reg


def block_budgets(M):
    """The same four-term budget applied inside each spectator block.

    Identical construction to decompose(), with two scope changes: d = 4, and
    the survival is the WITHIN-block column norm, so population that flips a
    spectator counts as leaving (it appears in the leak/exit term rather than as
    misrouting). Everything is evaluated in the REGISTER gauge — the same 8
    angles that give Fbar_16 — so the block bars, the register bar and panel (a)
    are all one gauge and are directly comparable.

    The point of the comparison: the inter-input term measured inside a block is
    the phase error with the spectators held in a definite state; measured over
    the register it also contains the phase conditioned on the spectator state.
    The GROWTH from one to the other is the active-spectator conditional phase,
    isolated by scope rather than by fitting a ZZ model to it.
    """
    x, _ = best_gauge(target_sqiswap_03(), M)
    zl, zr = _zphases(x[:4]), _zphases(x[4:8])
    Mg = (zl[:, None] * M) * zr[None, :]

    out = {}
    for q1 in (0, 1):
        for q2 in (0, 1):
            idx = [(q0 << 3) | (q1 << 2) | (q2 << 1) | q3
                   for q0 in (0, 1) for q3 in (0, 1)]
            Mb, Mbg = M[np.ix_(idx, idx)], Mg[np.ix_(idx, idx)]
            s = (np.abs(Mb) ** 2).sum(0)
            p = (np.abs(_SQ2) * np.abs(Mb)).sum(0)
            o = np.einsum("ij,ij->j", _SQ2.conj(), Mbg)
            S = float(s.sum())

            def _F(c, S=S):
                return (S + c ** 2) / 20.0

            b_lk = _F(float(np.sqrt(s).sum()))
            b_rt = _F(float(p.sum()))
            b_in = _F(float(np.abs(o).sum()))
            Fb = _F(abs(o.sum()))
            out[(q1, q2)] = dict(realized=Fb, inter=b_in - Fb,
                                 internal=b_rt - b_in, misrouting=b_lk - b_rt,
                                 leakage=1.0 - b_lk)
    return out


def decompose(M):
    """-> dict of the per-input quantities Fbar is built from."""
    U = target_sqiswap_03()
    x, fxc = best_gauge(U, M)
    zl, zr = _zphases(x[:4]), _zphases(x[4:8])
    Mg = (zl[:, None] * M) * zr[None, :]
    o = np.einsum("ij,ij->j", U.conj(), Mg)        # complex per-input overlap
    s = (np.abs(M) ** 2).sum(0)                    # survival (gauge-invariant)
    nm = np.vdot(M, M).real                        # = sum_j s_j
    Fbar = (nm + fxc * np.vdot(U, U).real * nm) / (16 * 17)
    # canonical orientation: rotate the whole fan so the resultant is on +Re.
    o = o * np.exp(-1j * np.angle(o.sum()))
    # Fit-free error attribution. The identity Fbar = [sum_j s_j + C^2]/(d(d+1))
    # with C = |sum_j o_j|, plus the forced chain
    #
    #   16  >=  sum_j sqrt(s_j)  >=  sum_j p_j  >=  sum_j |o_j|  >=  |sum_j o_j|
    #                                  p_j = sum_i |U_ij| |M_ij|
    #
    # (Cauchy-Schwarz, then the triangle inequality twice) gives four successive
    # UPPER BOUNDS on Fbar computed from the measured data — no free parameters,
    # no extra gauge freedom. Their gaps attribute the shortfall from unity:
    #
    #   leakage        population that left the computational subspace
    #   misrouting     population inside the subspace on the WRONG anchors
    #   internal phase relative phase inside one output superposition; nonzero
    #                  only for the 8 inputs sqiSWAP(0,3) mixes, since for the
    #                  others the ideal output is a single anchor and p_j=|o_j|
    #   inter-input    outputs not in phase with EACH OTHER; the only term that
    #                  is not a property of a single input
    #
    # p_j is the crucial refinement: |o_j| alone conflates amplitude with the
    # internal phase of the produced superposition, so a three-term split
    # mislabels most of the phase error as amplitude.
    #
    # GAUGE CAVEAT. leakage and misrouting are gauge
    # invariant (moduli only), and so is the TOTAL phase error, since sum_j p_j
    # is gauge free and |sum_j o_j| is a maximum. The split of that total into
    # internal and inter-input is NOT invariant: Z_L acts on output rows, so for
    # the eight mixing inputs it rotates the relative phase inside the ideal
    # output superposition, and the optimiser may park error in either bucket.
    # The dependence is one-parameter — the two anchors of a mixing output
    # differ only in bits q0 and q3, so only theta0 - theta3 matters. Quote the
    # total when the split is load-bearing.
    S = float(s.sum())
    p = (np.abs(U) * np.abs(M)).sum(0)

    def _F(c):
        return (S + c ** 2) / (16 * 17.0)

    b_leak = _F(float(np.sqrt(s).sum()))
    b_route = _F(float(p.sum()))
    b_int = _F(float(np.abs(o).sum()))
    budget = dict(leakage=1.0 - b_leak, misrouting=b_leak - b_route,
                  internal=b_route - b_int, inter=b_int - Fbar, realized=Fbar)
    # Exact additive per-state breakdown of Fbar: expanding |sum_j o_j|^2 and
    # splitting each interference term equally between its two states gives
    #     Ftilde_j = [ s_j + Re(o_j conj(O)) ] / (d+1),   O = sum_k o_k
    # whose MEAN over the states is exactly Fbar. No fitting, no extra gauge
    # freedom. A state whose phase agrees with the collective resultant O
    # contributes positively; one that opposes it contributes NEGATIVELY,
    # which no per-input fidelity can express.
    Ft = (s + np.real(o * np.conj(o.sum()))) / 17.0
    return dict(o=o, s=s, p=p, Ft=Ft, Fbar=Fbar, fxc=fxc, budget=budget,
                amp=np.abs(o).mean(), coh=abs(o.sum()) / 16.0,
                circ_sd=float(np.degrees(np.sqrt(-2 * np.log(
                    abs(o.sum()) / np.abs(o).sum())))))
