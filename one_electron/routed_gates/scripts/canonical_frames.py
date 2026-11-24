"""Deterministic local-Z frames for the routed arms (LEGS_TAG=symz).

`strip_localZ` maximises |<SQI| z4(aL,bL) M z4(aR,bR)>| by BFGS from random
starts.  SQI commutes with z4(c,c) = exp(icN), so the overlap is exactly flat
along (aL,bL,aR,bR) -> (aL+c, bL+c, aR-c, bR-c) and BFGS lands anywhere on
that line.  The arms are not invariant to c (leaked and spectator amplitude do
not follow a qubit-orbital rotation), so c must be fixed by convention.

Convention, per PHYSICAL two-qubit gate:
  1. fit the STANDALONE leg's corner in closed form (four terms aligned exactly,
     the residual |11> phase shared by one weighted least-squares step);
  2. fix c by the symmetric split z_L = z_R.  The standalone legs are
     transpose-symmetric (symmetric pulse, real H), so the split is exact; of
     the two solutions c, c+pi (Z(x)Z moved across the leg) take the one with
     the smaller total rotation;
  3. the concurrent leg reuses each pair's standalone frame, plus the free
     evolution +omega*tau_idle of a pair that finishes early and idles at rest
     (an after-the-gate phase).

Reads  results/arms_<set>/<source legs16 file>
Writes results/arms_<set>/legs16_symz.npz : the source store with z_* replaced.
CPU only, seconds.  Run: python scripts/canonical_frames.py
"""
import os, sys
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.update(ARM="legs16", PULSE_SET="best", SWEEP_DEV="cpu",
                  TAU_W0W1="82.3,73.2", TAU_W1W2="57.2,95.4",
                  TAU_W2W3="34.1,112.8", TAU_W0W3="81.3,2160.7")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_arm_decomp as A                     # noqa: E402  (env must be set first)
from common import SQI, _z4                    # noqa: E402

R = os.path.join(HERE, "..", "results")
# (pulse set, source store, its dt_div).  The sync store predates the dt_div
# key; logs/rw_sync_5651_0.out shows it was made at SWEEP_DT_DIV=0.25.
SOURCES = [("best", "legs16_legsdt0.25.npz", None),
           ("sync", "legs16.npz", 0.25)]
SOLO = {"01": ("w0w1", (0, 1)), "23": ("w2w3", (2, 3))}
G = np.array([1.0, 1.0, -1.0, -1.0])           # the flat direction


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def overlap(M, p):
    return abs(np.vdot(SQI, _z4(p[0], p[1]) @ M @ _z4(p[2], p[3]))) / (
        np.linalg.norm(SQI) * np.linalg.norm(M))


# term k of <SQI| z_L M z_R>: conj(SQI[j,l]) M[j,l] exp(i(alpha_j + beta_l)),
# alpha = (0, aL, bL, aL+bL), beta likewise; rows of D are d(alpha_j+beta_l)/dp
TERMS = [(0, 0), (1, 1), (2, 2), (1, 2), (2, 1), (3, 3)]
D = np.array([[0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1],
              [1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 1, 1]], float)


def fit_closed_form(M):
    t = np.array([np.conj(SQI[j, l]) * M[j, l] for j, l in TERMS])
    psi, wt = np.angle(t), np.sqrt(np.abs(t))[:, None]
    phi = psi[0]                                # align (00),(11),(22),(12)
    aL = phi - psi[1]
    bR = phi - psi[3] - aL
    bL = phi - psi[2] - bR
    p = np.array([aL, bL, 0.0, bR])
    X = np.hstack([D, -np.ones((6, 1))])
    for _ in range(2):                          # one step converges; two verifies
        r = wrap(psi + D @ p - phi)
        s = np.linalg.lstsq(wt * X, -wt[:, 0] * r, rcond=None)[0]
        p, phi = p + s[:4], phi + s[4]
    return wrap(p)


def symmetric_split(p):
    dA, dB = wrap(p[2] - p[0]) / 2, wrap(p[3] - p[1]) / 2
    # dA, dB are each defined mod pi; take dB's representative nearest dA
    dB = dA + wrap(2 * (dB - dA)) / 2
    c = (dA + dB) / 2
    cands = [wrap(p + cc * G) for cc in (c, c + np.pi)]
    rot = [float(np.sum(q ** 2)) for q in cands]
    i = int(np.argmin(rot))
    q = cands[i]
    asym = max(abs(wrap(q[0] - q[2])), abs(wrap(q[1] - q[3])))
    return q, asym, rot[i], rot[1 - i]


def qubit_omegas():
    """omega_s = E(single excitation in well s) - E(|0000>) at rest."""
    C16, bs0 = A.load_anchors16()
    solver, _ = A.make_solver_par()
    y = A.pack_Cb_to_y(A.jnp.stack(C16),
                       [A.jnp.asarray(b, dtype=A.jnp.complex128) for b in bs0])
    E = {}
    for tt in (5.0e3, 2.0e4):                   # both past every envelope
        e = np.real(np.asarray(solver.calculate_energy(A.jnp.asarray(tt), y)))
        E[tt] = {tuple(c): e[i] for i, c in enumerate(A._B.CONFIGS)}
    e0, e1 = E[5.0e3], E[2.0e4]
    assert max(abs(e0[k] - e1[k]) for k in e0) < 1e-10, "rest energies not static"
    om = np.array([e0[tuple(int(s == w) for w in range(4))] - e0[(0, 0, 0, 0)]
                   for s in range(4)])
    print(f"  omega (rest) = {np.round(om, 6)}", flush=True)
    return om


def main():
    om = qubit_omegas()
    w1w2_frames = []
    for pset, src, dt_src in SOURCES:
        f = os.path.join(R, f"arms_{pset}", src)
        dst = os.path.join(R, f"arms_{pset}", "legs16_symz.npz")
        assert not os.path.exists(dst), f"{dst} exists; refusing to overwrite"
        st = dict(np.load(f))
        dt_div = float(st["dt_div"]) if "dt_div" in st else dt_src
        assert abs(dt_div - 0.25) < 1e-12, f"{src}: expected a 4dt store"
        print(f"\n[{pset}] {src}  (dt_div {dt_div})", flush=True)
        new = {}
        for key, (a, b) in (("w0w1", (0, 1)), ("w1w2", (1, 2)), ("w2w3", (2, 3))):
            M = st[f"corner_{key}_{a}{b}"]
            p0 = fit_closed_form(M)
            q, asym, rot, rot2 = symmetric_split(p0)
            o_old, o_new = overlap(M, st[f"z_{key}_{a}{b}"]), overlap(M, q)
            print(f"  {key}: frame z_L=z_R=({q[0]:+.5f},{q[1]:+.5f})  asym {asym:.1e}"
                  f"  |rot|^2 {rot:.2f} (other branch {rot2:.2f})"
                  f"  overlap {o_new:.9f} vs BFGS {o_old:.9f}", flush=True)
            # guards: same optimum as BFGS; the leg really is symmetric; the
            # branch choice is not a near-tie
            assert abs(o_new - o_old) < 1e-12, "closed form missed the optimum"
            assert asym < 1e-3, f"{key} is not transpose-symmetric ({asym:.1e})"
            assert rot2 - rot > 0.5, f"{key}: c / c+pi branch is a near-tie"
            new[f"z_{key}_{a}{b}"] = q
            if key == "w1w2":
                w1w2_frames.append(q)
        for pr, (solo, (a, b)) in SOLO.items():
            idle = float(st["T_par"]) - float(st[f"T_{solo}"])
            assert idle >= -1e-9, f"pair {pr} outlasts the concurrent leg"
            q = new[f"z_{solo}_{a}{b}"] + np.array([om[a] * idle, om[b] * idle, 0, 0])
            q = wrap(q)
            M = st[f"corner_par_{pr}"]
            o_new, o_own = overlap(M, q), overlap(M, st[f"z_par_{pr}"])
            print(f"  par ({a},{b}): idle {idle:.1f} t.u.  overlap {o_new:.6f}"
                  f" vs own fit {o_own:.6f}  (loss {o_own - o_new:.1e})", flush=True)
            # a sign or wiring error costs ~0.1-0.5; real crosstalk ~5e-5
            assert o_own - o_new < 1e-3, "standalone+idle frame does not fit par"
            new[f"z_par_{pr}"] = q
        out = {k: v for k, v in st.items() if k not in ("dt_div", "dt")}
        out.update(new)
        np.savez(dst, dt_div=dt_div, dt=A.DT_PROD / dt_div, **out,
                 frame_convention="symmetric split z_L=z_R of standalone legs; "
                                  "concurrent pairs = standalone + omega*idle",
                 frame_source=src, omega_rest=om)
        print(f"  -> {dst}", flush=True)
    assert np.abs(wrap(w1w2_frames[0] - w1w2_frames[1])).max() < 1e-6, \
        "best and sync share the w1w2 pulse but got different frames"


if __name__ == "__main__":
    main()
