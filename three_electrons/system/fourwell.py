"""The four-well system with three electrons per well.

Four anharmonic wells, N = 3 electrons each, treated as four distinguishable
species, in the ``new_2um`` geometry.  It is built here from the physical
constants, with the ingredients below; ``check_replica.py`` checks it against
reference energies of an earlier standard-MCTDHF three-electron calculation.

Ingredients
-----------

Geometry (as ``one_electron/lib/geometry_config.py``, GEOMETRY = "new_2um")
    ``NUM_DVR = 64``, well centres at -3, -1, 1, 3 um, well widths 1 um, and
    the oscillator units of omega = 2 pi 10 GHz. KAPPA is the Coulomb
    strength in those units, 811.15.

Intra-well interaction (``INTRA_KERNEL = "veff"``)
    The INTRA-well interaction is NOT a soft Coulomb. It is the derived
    quasi-1D kernel of Sec. S2: the Coulomb interaction averaged
    over the transverse distributions of a harmonic y-channel at
    ``OMEGA_Y_REL = 3`` and the hydrogenic surface state of PRB 107, 035437
    (``r_B = 7.6 nm``). Substituting a soft Coulomb here changes the
    correlation and with it the natural-occupation tail.

Inter-well interaction
    The INTER-well interaction is the BARE Coulomb ``kappa / |x1 - x2|``
    across the two grids — the wells do not overlap, so there is no
    singularity to soften. Each well carries its OWN sine-DVR box
    ``[c - w/2, c + w/2]``, not a shared grid.

Potential (as ``one_electron/lib/utils.anharm_pot``)
    ``V(x) = 0.5 a^2 (x-c)^2 + (b a^2 / 3) (x-c)^4``, with ``(a, b)`` the
    per-well (alpha, beta) control parameters.

``X_IDLE`` is the one-electron idle point,
``one_electron/operating_points/ITE_trajectories/new_geometry_step1_stage1_
singles_ground.npz``'s ``best_x`` divided by ``TYPICAL_FREQ_GHZ = 10``;
``X_GATE_W0W1`` is the w0w1 gate point of the earlier standard-MCTDHF
calculation. Both are inlined below rather than loaded, so this module has
no data dependency.  The operating points of the paper are in
operating_points/.
"""

import os

import numpy as np
from scipy import constants as _const
from scipy.integrate import quad
from scipy.special import erf as _erf, k0e

import jax.numpy as jnp

from mctdhf.sine_dvr import SineDVR

# ── units and geometry ────────────────────────────────────────────────────

NUM_WELLS = 4
NUM_PARTICLES = 3
NUM_DVR = int(os.environ.get("FOURWELL_NUM_DVR", "64"))
TYPICAL_FREQ_GHZ = 10.0

_OMEGA = 2 * np.pi * TYPICAL_FREQ_GHZ * 1e9
LENGTH_UNIT_UM = np.sqrt(_const.hbar / (_const.m_e * _OMEGA)) * 1e6
TIME_UNIT_PS = (1.0 / _OMEGA) * 1e12
KAPPA = float(
    np.sqrt(_const.hbar / (_const.m_e * _OMEGA))
    * _const.m_e * _const.e ** 2
    / (4 * np.pi * _const.epsilon_0 * _const.hbar ** 2)
)

WELL_CENTERS_UM = np.array([-3.0, -1.0, 1.0, 3.0])
WELL_WIDTHS_UM = np.array([1.0, 1.0, 1.0, 1.0])
WELL_CENTERS = WELL_CENTERS_UM / LENGTH_UNIT_UM
WELL_WIDTHS = WELL_WIDTHS_UM / LENGTH_UNIT_UM

# geometry_config computes these with pint; scipy's CODATA agrees to the
# printed precision. Guard the agreement rather than assume it.
assert abs(KAPPA - 811.1525085) < 1e-6, KAPPA
assert abs(LENGTH_UNIT_UM - 0.0429243421775) < 1e-12, LENGTH_UNIT_UM

# ── control points ────────────────────────────────────────────────────────
# best_x / 10, from the step-1 anchor. alpha = x[:4], beta = x[4:].
X_IDLE = np.array([0.768712362, 1.067153578, 1.378836939, 1.495916851,
                   0.029528536, 0.056287096, 0.056162843, 0.197051925])

# The w0w1 gate point of the earlier standard-MCTDHF calculation: wells 0 and
# 1 at a common alpha, the two spectators at idle.
X_GATE_W0W1 = X_IDLE.copy()
X_GATE_W0W1[0] = 1.172761
X_GATE_W0W1[1] = 1.172761
X_GATE_W0W1[4] = 0.143815
X_GATE_W0W1[5] = 0.143241

# ── the quasi-1D intra-well kernel (Sec. S2) ──────────────────────────────

OMEGA_Y_REL = 3.0                      # channel frequency / axial
R_B_UM = 0.0076                        # 7.6 nm, PRB 107, 035437
R_B = R_B_UM / LENGTH_UNIT_UM
SIGMA_U2 = 1.0 / OMEGA_Y_REL


def _veff_scalar(x_abs, kappa, epsrel=1e-11):
    """V_eff at one axial separation.

    Adaptive quadrature, not Gauss-Laguerre: ``k0e`` has a log endpoint at
    x = 0 where a fixed Laguerre rule converges only slowly.
    """
    a = 2.0 / R_B
    pref = kappa / (2.0 * np.sqrt(SIGMA_U2)) * np.sqrt(2.0 / np.pi)

    def f(s):
        q = (x_abs ** 2 + (s / a) ** 2) / (4.0 * SIGMA_U2)
        return (3.0 + 3.0 * s + s ** 2) / 8.0 * np.exp(-s) * k0e(q)

    val, _ = quad(f, 0.0, 40.0, epsabs=0.0, epsrel=epsrel, limit=400)
    return pref * val


def veff(x, kappa=KAPPA):
    """V_eff at axial separation(s), evaluated once per unique |x|."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    uniq, inv = np.unique(np.round(x.ravel(), 12), return_inverse=True)
    vals = np.array([_veff_scalar(xu, kappa) for xu in uniq])
    return vals[inv].reshape(x.shape)


def _veff_interaction(x1, x2):
    return veff(np.asarray(x1) - np.asarray(x2))


# ── grids, kinetic energy, interaction ────────────────────────────────────

DVRS = [SineDVR(x_min=float(WELL_CENTERS[w]) - 0.5 * float(WELL_WIDTHS[w]),
                x_max=float(WELL_CENTERS[w]) + 0.5 * float(WELL_WIDTHS[w]),
                num_points=NUM_DVR)
        for w in range(NUM_WELLS)]

X_S = [np.asarray(d.grid, dtype=np.float64) for d in DVRS]

_T_KIN = None
_G_DICT = None


def t_kin():
    """Kinetic-energy matrix per well (potential-free ``get_h``)."""
    global _T_KIN
    if _T_KIN is None:
        _T_KIN = [jnp.asarray(d.get_h(potential=lambda x: 0.0 * x),
                              dtype=jnp.float64) for d in DVRS]
    return _T_KIN


def g_dict():
    """``{(s, k): g}`` — veff within a well, bare Coulomb between wells.

    The intra-well kernel is built once on well 0's grid and reused for all
    four: the boxes are translates of one another, so ``V(x1 - x2)`` is the
    same matrix. That is what ``leg_lib`` does (``build_well(..., well=0)``
    feeding every diagonal entry) and it saves three 64x64 adaptive
    quadratures.
    """
    global _G_DICT
    if _G_DICT is None:
        g0 = np.asarray(DVRS[0].get_g(interaction=_veff_interaction))
        gd = {(w, w): jnp.asarray(g0, dtype=jnp.float64)
              for w in range(NUM_WELLS)}
        for a in range(NUM_WELLS):
            for b in range(a + 1, NUM_WELLS):
                x1, x2 = np.meshgrid(X_S[a], X_S[b], indexing="ij")
                gd[(a, b)] = jnp.asarray(KAPPA / np.abs(x1 - x2),
                                         dtype=jnp.float64)
        _G_DICT = gd
    return _G_DICT


# ── the one-body Hamiltonian, static and ramped ───────────────────────────

def _v_well(w, alpha, beta):
    dx = X_S[w] - float(WELL_CENTERS[w])
    return 0.5 * alpha ** 2 * dx ** 2 + (beta * alpha ** 2 / 3.0) * dx ** 4


def h_static(x, dtype=jnp.complex128):
    """Frozen ``h_list`` at control vector ``x = [alpha_0..3, beta_0..3]``."""
    T = t_kin()
    return [T[w] + jnp.diag(jnp.asarray(_v_well(w, x[w], x[4 + w]),
                                        dtype=dtype))
            for w in range(NUM_WELLS)]


def make_envelope(tau_ramp, tau_hold):
    """The erf ramp-hold-ramp envelope, and the total time.

    ``env(t) = 1/2 [erf(6(t - tau_r/2)/tau_r) - erf(6(t - t_dn)/tau_r)]``
    with ``t_dn = 3 tau_r / 2 + tau_hold``, so ``env`` is 0 at both ends, 1
    on the hold, and integrates to ``tau_hold + tau_ramp`` exactly.
    """
    t_dn = 1.5 * tau_ramp + tau_hold
    total = 2 * tau_ramp + tau_hold

    def env(t):
        return 0.5 * (jnp.asarray(_erf_j(6 * (t - 0.5 * tau_ramp) / tau_ramp))
                      - jnp.asarray(_erf_j(6 * (t - t_dn) / tau_ramp)))

    return env, total


def make_env_up(tau_ramp):
    """The LEADING EDGE alone: env(0) = 0 rising to 1 at t = tau_ramp.

    The integration tests propagate short windows.  The integration error is
    generated in the ramp edges rather than in the hold, and partly undone
    in the ramp-down, so using the up-ramp alone keeps a short
    window in the regime that actually stresses the integrator, and avoids
    the error cancellation a round trip would introduce.
    """
    def env(t):
        return 0.5 * (1.0 + _erf_j(6 * (t - 0.5 * tau_ramp) / tau_ramp))

    return env


def _erf_j(z):
    from jax.scipy.special import erf as jerf
    return jerf(jnp.asarray(z, dtype=jnp.float64))


def h_ramp(x_idle, x_gate, env, dtype=jnp.complex128):
    """``h_list`` of callables interpolating idle -> gate linearly in ``env``."""
    T = t_kin()
    xi = np.asarray(x_idle, dtype=np.float64)
    xg = np.asarray(x_gate, dtype=np.float64)
    hs = []
    for w in range(NUM_WELLS):
        dx = jnp.asarray(X_S[w] - float(WELL_CENTERS[w]), dtype=jnp.float64)

        def h_of(t, w=w, dx=dx):
            e = env(t)
            a = xi[w] + (xg[w] - xi[w]) * e
            b = xi[4 + w] + (xg[4 + w] - xi[4 + w]) * e
            v = 0.5 * a ** 2 * dx ** 2 + (b * a ** 2 / 3.0) * dx ** 4
            return T[w] + jnp.diag(v.astype(dtype))

        hs.append(h_of)
    return hs


# ── reference energies (check_replica.py) ─────────────────────────────────

E_HARTREE_IDLE = 2586.80203116     # reference, four-well Hartree
E_MCTDHF6_IDLE = 2586.86273224     # reference, norb-6 MCTDHF


def self_check(verbose=True):
    """Cheap structural checks. The energy gate lives in ``check_replica.py``."""
    g = g_dict()
    ok = True
    # intra kernel: finite on the diagonal (veff has no singularity)
    g00 = np.asarray(g[(0, 0)])
    if not np.all(np.isfinite(g00)):
        ok = False
    # eps_eff = kappa / V_eff(0), the recorded equivalent softening
    eps_eff = KAPPA / float(veff(0.0))
    # inter-well: bare Coulomb, so the nearest pair sets the scale
    g01 = np.asarray(g[(0, 1)])
    if verbose:
        print(f"  kappa            {KAPPA:.6f}")
        print(f"  length unit      {LENGTH_UNIT_UM:.8f} um")
        print(f"  box per well     [{DVRS[0].x_min:.4f}, {DVRS[0].x_max:.4f}]"
              f"  dx = {float(DVRS[0].dx):.6f}")
        print(f"  V_eff(0)         {float(veff(0.0)):.6f}  "
              f"(eps_eff = {eps_eff:.6f})")
        print(f"  g_00 range       [{g00.min():.4f}, {g00.max():.4f}]")
        print(f"  g_01 range       [{g01.min():.4f}, {g01.max():.4f}]")
    return ok
