"""Exact propagation of one leg for all 16 initial states.

One process, sixteen states in sequence: the segment scans are compiled ONCE and
reused across states, which matters because the m = 12 Krylov body unrolls into
12 matvecs plus 78 re-orthogonalisation axpys.  A SLURM array over states would
pay that compile sixteen times.

Exact propagation is LINEAR, so the sixteen states are independent.  That is a
real difference from the MCTDH side, where all sixteen share one state-averaged
orbital set and are therefore coupled -- and it is part of what is under test,
not a mismatch to be corrected.

Env:
  EXACT_LEG   w0w1 | w1w2 | w2w3 | par | w0w3      PULSE_SET  best | fast | sync
  PREC        x64 | x32                            ANCHOR     primary | secondary
  DT_FINE / DT_COARSE   step policy (halve both for the convergence check)
  SNAP_EVERY  t.u. between diagnostics (0 = endpoints only)
  STATES      'all' or a comma list             SAVE_FINAL  1 to keep the states
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact_common as ec
import sil
import split_op

# strang (default, used for the paper) | sil
PROPAGATOR = os.environ.get("PROPAGATOR", "strang")
PROP = split_op if PROPAGATOR == "strang" else sil
LEG = os.environ.get("EXACT_LEG", "w0w1")
PULSE_SET = os.environ.get("PULSE_SET", "best")
PREC = os.environ.get("PREC", "x64")
ANCHOR = os.environ.get("ANCHOR", "primary")
DT_FINE = float(os.environ.get("DT_FINE", "0.5"))
DT_COARSE = float(os.environ.get("DT_COARSE", "5.0"))
SNAP_EVERY = float(os.environ.get("SNAP_EVERY", "5.0"))
SAVE_FINAL = os.environ.get("SAVE_FINAL", "0") == "1"
STATES = (list(range(16)) if os.environ.get("STATES", "all") == "all"
          else [int(x) for x in os.environ["STATES"].split(",")])

TAG = (f"leg_{LEG}_{PULSE_SET}_{PREC}_{ANCHOR}"
       f"_dt{DT_FINE:g}-{DT_COARSE:g}_{PROPAGATOR}")
OUT = os.path.join(ec.RESULTS, TAG)
os.makedirs(OUT, exist_ok=True)


def leg_setup():
    if LEG == "par":
        t01 = ec.PULSE_SETS[PULSE_SET]["w0w1"]
        t23 = ec.PULSE_SETS[PULSE_SET]["w2w3"]
        return ec.h_fns_par(t01, t23), {0: t01, 1: t01, 2: t23, 3: t23}
    taus = ec.TAU_W0W3 if LEG == "w0w3" else ec.PULSE_SETS[PULSE_SET][LEG]
    return ec.h_fns_leg(LEG, taus), {s: taus for s in range(4)}


def e_shift_for(prec):
    """0 at x64; at x32 ONE constant shared by all sixteen states.

    This is deliberately not the per-state e_shift='auto' used elsewhere in the
    codebase -- that one differs between states and therefore strips the
    entangling phase.  A single number applied identically to all sixteen is a
    pure global phase and cannot move any relative phase or fidelity.
    """
    if prec == "x64":
        return 0.0
    f = os.path.join(ec.RESULTS, "sigma_E.npz")
    if os.environ.get("E0"):
        return float(os.environ["E0"])
    assert os.path.exists(f), "x32 runs need E0 (set E0 or provide sigma_E.npz)"
    return float(np.load(f)["E"].mean())


def main():
    dev = ec.device()
    cdt, rdt = ec.dtypes(PREC)
    h_fns, taus = leg_setup()
    if PROPAGATOR == "strang":
        # Strang's error is per-step and roughly uniform in time, unlike SIL's,
        # so the ramp-localised fine/coarse schedule has nothing to buy here:
        # one dt, chosen by the order-4 convergence scan.
        T = max(2 * tr + th for tr, th in taus.values())
        n = int(round(T / DT_FINE))
        segs = [(0.0, T, T / n, n)]
    else:
        segs, T = sil.step_schedule(taus, DT_FINE, DT_COARSE)
    E0 = e_shift_for(PREC)

    print(f"=== propagate_leg {TAG} ===", flush=True)
    print(f"  dev={dev}  T={T}  taus={taus}", flush=True)
    print(f"  {sil.describe(segs)}", flush=True)
    print(f"  e_shift E0 = {E0:.6f} (uniform => global phase)", flush=True)

    with jax.default_device(dev):
        C16, bs0, _ = ec.load_primary_anchors()
        bsj = [jnp.asarray(b, dtype=jnp.complex128) for b in bs0]
        QB = {s: ec.qubit_basis16(s, C16, bs0)[0] for s in range(4)}
        q1s = [jnp.asarray(QB[s][:, 1], dtype=cdt) for s in range(4)]
        Vd = ec.build_Vd(rdt)

        sec = None
        if ANCHOR == "secondary":
            sec = [np.load(os.path.join(ec.RESULTS, f"secondary_anchor_{j:02d}.npy"))
                   for j in range(16)]

        # compile one scan per (segment length, dt); reused for all 16 states
        runners, t0c = {}, time.perf_counter()
        for (a, b, dt, n) in segs:
            step_chunk = max(1, int(round(SNAP_EVERY / dt))) if SNAP_EVERY > 0 else n
            for nn in {min(step_chunk, n), n % step_chunk or step_chunk}:
                if (dt, nn) not in runners and nn > 0:
                    runners[(dt, nn)] = PROP.make_segment(
                        h_fns, Vd, dt, nn, E0, cdt, rdt)
        print(f"  {len(runners)} distinct scans to compile", flush=True)

        M_prim = np.zeros((16, 16), complex)
        M_sec = np.zeros((16, 16), complex) if sec is not None else None
        summary = []

        # The Strang propagator is state-INDEPENDENT (exp(-i dt h_s) is four
        # 64x64 matrices shared by every state), so the ensemble travels as one
        # (B, 64, 64, 64, 64) array.  Under SIL this was impossible: the Krylov
        # basis was 3.2 GB PER STATE.
        #
        # B is capped by GPU memory, not by the method: a Strang step holds
        # roughly two ensemble-sized buffers, so B=16 at x64 peaks near 8.6 GB
        # against 7.44 GB free.  B=8 peaks near 4.3 GB and fits; at x32 the whole
        # sixteen would.
        # B=4 at x64, not 8.  Sizing this by the PROPAGATION peak (two
        # ensemble-sized buffers per Strang step) missed the CONSTRUCTION peak:
        # jnp.stack holds the individual states AND the stacked copy at once, so
        # building a group of 8 needed ~4.3 GB before a single step ran, on top
        # of the compiled scans.  Construction is now done on the host and moved
        # across in one transfer, and B is set from the larger of the two peaks.
        B = int(os.environ.get("BATCH_SIZE", "4" if PREC == "x64" else "8"))
        groups = [STATES[i:i + B] for i in range(0, len(STATES), B)]
        print(f"  ensemble batched {B} at a time -> {len(groups)} group(s)",
              flush=True)

        for grp in groups:
            # built one state at a time on the host, so the device never holds
            # more than one 268 MB state plus the final stacked ensemble
            v_host = np.empty((len(grp),) + (ec.N,) * 4, dtype=np.complex128)
            for k, j in enumerate(grp):
                if ANCHOR == "primary":
                    v_host[k] = np.asarray(ec.expand(C16[j], bsj))
                else:
                    v_host[k] = sec[j]
            v = jnp.asarray(v_host, dtype=cdt)
            del v_host
            def _diag(vv):
                o = np.stack([np.asarray(ec.occupations(q1s, vv[k]))
                              for k in range(vv.shape[0])])
                p = np.array([float(np.sum(np.abs(np.asarray(
                    ec.anchor_overlaps(C16, bsj, vv[k]))) ** 2))
                    for k in range(vv.shape[0])])
                return o, p

            n_start = np.array([float(ec.norm2(v[k])) for k in range(len(grp))])
            o0, p0 = _diag(v)
            ts, ns, p16, errs = [0.0], [o0], [p0], []
            t = 0.0
            tic = time.perf_counter()
            for (a, b, dt, n) in segs:
                step_chunk = max(1, int(round(SNAP_EVERY / dt))) if SNAP_EVERY > 0 else n
                done = 0
                while done < n:
                    nn = min(step_chunk, n - done)
                    tf, v, emax = runners[(dt, nn)](t, v)
                    t = float(tf)
                    done += nn
                    errs.append(float(emax))
                    if SNAP_EVERY > 0:
                        ts.append(t)
                        o, p = _diag(v)
                        ns.append(o)
                        p16.append(p)
            jax.block_until_ready(v)
            wall = time.perf_counter() - tic
            arr_t, arr_n, arr_p = np.array(ts), np.array(ns), np.array(p16)
            for k, j in enumerate(grp):
                n_end = float(ec.norm2(v[k]))
                ov_p = np.asarray(ec.anchor_overlaps(C16, bsj, v[k]))
                M_prim[:, j] = ov_p
                if sec is not None:
                    vh = np.asarray(v[k], dtype=np.complex128)
                    for i in range(16):
                        M_sec[i, j] = np.vdot(sec[i], vh)
                dn = abs(1.0 - n_end / n_start[k])
                print(f"  state {j:2d} {''.join(map(str, ec.CONFIGS[j]))}: "
                      f"|1-norm| {dn:.2e}  P16(T) {arr_p[-1, k]:.6f}", flush=True)
                np.savez(os.path.join(OUT, f"state_{j:02d}.npz"),
                         ov_primary=ov_p,
                         ov_secondary=(M_sec[:, j] if sec is not None
                                       else np.zeros(16, complex)),
                         t=arr_t, n_s=arr_n[:, k], p16=arr_p[:, k],
                         norm_start=n_start[k], norm_end=n_end, T=T)
                if SAVE_FINAL:
                    np.save(os.path.join(OUT, f"final_{j:02d}.npy"),
                            np.asarray(v[k], dtype=np.complex128))
                summary.append(dict(state=j, dnorm=dn, wall=wall / len(grp)))
            print(f"  group {grp[0]}-{grp[-1]} done in {wall/60:.1f} min",
                  flush=True)
            del v

        meta = dict(tag=TAG, leg=LEG, pulse_set=PULSE_SET, prec=PREC,
                    anchor=ANCHOR, T=T, e_shift=E0, propagator=PROPAGATOR,
                    dt_fine=DT_FINE, dt_coarse=DT_COARSE,
                    segments=[[a, b, dt, n] for a, b, dt, n in segs],
                    total_steps=int(sum(n for *_, n in segs)),
                    states=STATES, summary=summary,
                    compile_s=time.perf_counter() - t0c)
        json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
        np.savez(os.path.join(OUT, "M.npz"), M_primary_cfg=M_prim,
                 M_secondary_cfg=(M_sec if M_sec is not None else np.zeros((16, 16))),
                 configs=np.array(ec.CONFIGS))
    print(f"  -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
