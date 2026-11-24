"""Exact propagation of a composed ARM.

The `parallel`/`sync` circuit:

    [SWAP(0,1) || SWAP(2,3)] . sqiSWAP(1,2) . [SWAP(0,1) || SWAP(2,3)]

built from 7 leg slots and 8 coalesced one-body rotations, T = 1452.4 t.u.

The w0w1 leg alone is reproduced by MCTDH at norb=4 to dF_bar16 = -1.03e-5, so
a difference at the circuit level comes from the composition, not from
per-leg ansatz error (App. C).

THE Z-FRAMES ARE INPUTS, NOT OUTPUTS.  Each leg's local-Z frame was measured on
the MCTDH legs and stored in legs16[_<LEGS_TAG>].npz.  They are LOADED here, never re-derived
from exact legs: the circuit must be the same circuit, or the comparison stops
isolating the ansatz.  Same for the qubit orbitals Q, which come from the anchor
1-RDMs exactly as run_arm_decomp builds them.

The single-qubit layers act as U = I - QQ^H + Q u2 Q^H on that well's orbitals,
which on the exact state is a plain contraction along that axis -- no MCTDH
needed and no approximation.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact_common as ec
import split_op

PULSE_SET = os.environ.get("PULSE_SET", "sync")
PREC = os.environ.get("PREC", "x64")
DT = float(os.environ.get("DT_FINE", "0.25"))
BATCH = int(os.environ.get("BATCH_SIZE", "4" if PREC == "x64" else "8"))
ARM = os.environ.get("ARM", "parallel")
SNAP_EVERY = float(os.environ.get("SNAP_EVERY", "5.0"))
# LEGS_TAG selects the frame store, as in run_arm_decomp: "" = legs16.npz (BFGS
# frames), "symz" = legs16_symz.npz (symmetric split, the paper's).  It also
# suffixes TAG, so a re-run neither overwrites nor resumes an older one.
LEGS_TAG = os.environ.get("LEGS_TAG", "")
_SFX = f"_{LEGS_TAG}" if LEGS_TAG else ""
TAG = f"arm_{ARM}_{PULSE_SET}_{PREC}_dt{DT:g}_o{split_op.ORDER}{_SFX}"
OUT = os.path.join(ec.RESULTS, TAG)

LEGS16 = os.path.join(ec.RW, "results", f"arms_{PULSE_SET}", f"legs16{_SFX}.npz")
SDG = np.diag([1.0, -1j]).astype(complex)
BLOCK_U2 = ec.HAD @ SDG


def zframe(p, pair, side):
    """Virtual-Z frame update for one measured leg, as {well: 2x2}.

    strip_localZ returned M_stripped = z4(p0,p1) M z4(p2,p3), so the LEFT factor
    acts AFTER the leg and the RIGHT one BEFORE it.  The convention of
    run_arm_decomp.zframe -- the arm must meet the same rotations in the same
    order as the MCTDH run did.
    """
    a, b = (p[0], p[1]) if side == "after" else (p[2], p[3])
    return {pair[0]: np.diag([1.0, np.exp(1j * a)]).astype(complex),
            pair[1]: np.diag([1.0, np.exp(1j * b)]).astype(complex)}


def coalesce(sched):
    """Merge contiguous local operations into one application.

    Not cosmetic: one-body layers compose exactly, U(u)U(v) = U(uv), so merging
    changes nothing for the propagation -- but it keeps the number of physical
    rotations honest, which is what run_arm_decomp.coalesce exists for.
    """
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


def parallel_schedule(z01, z23, zent):
    blk = lambda: [("layer", {**{w: BLOCK_U2.copy() for w in (0, 1)},
                              **{w: BLOCK_U2.copy() for w in (2, 3)}}),
                   ("zframe", {**zframe(z01, (0, 1), "before"),
                               **zframe(z23, (2, 3), "before")}),
                   ("leg", "par"),
                   ("zframe", {**zframe(z01, (0, 1), "after"),
                               **zframe(z23, (2, 3), "after")})]
    sched = []
    for _ in range(3):
        sched += blk()
    sched += [("zframe", zframe(zent, (1, 2), "before")), ("leg", "w1w2"),
              ("zframe", zframe(zent, (1, 2), "after"))]
    for _ in range(3):
        sched += blk()
    return sched


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = ec.device()
    cdt, rdt = ec.dtypes(PREC)
    tp = ec.PULSE_SETS[PULSE_SET]
    store = dict(np.load(LEGS16))
    if LEGS_TAG:
        assert "frame_convention" in store, f"{LEGS16} has no frame_convention"
    z01, z23 = store["z_par_01"], store["z_par_23"]
    zent = store["z_w1w2_12"]
    sched = coalesce(parallel_schedule(z01, z23, zent))
    n_leg = sum(1 for k, _ in sched if k == "leg")
    n_loc = sum(1 for k, _ in sched if k == "local")

    h_par = ec.h_fns_par(tp["w0w1"], tp["w2w3"])
    h_ent = ec.h_fns_leg("w1w2", tp["w1w2"])
    T_par = round(max(2 * tp["w0w1"][0] + tp["w0w1"][1],
                      2 * tp["w2w3"][0] + tp["w2w3"][1]), 6)
    T_ent = ec.leg_T(tp["w1w2"])
    T_tot = 6 * T_par + T_ent

    print(f"=== propagate_arm {TAG} ===", flush=True)
    print(f"  dev={dev}  {n_leg} leg slots + {n_loc} rotations  "
          f"T = {T_tot} t.u. = {T_tot*0.0159155:.2f} ns", flush=True)
    print(f"  z frames loaded from {os.path.relpath(LEGS16, ec.ROOT)}", flush=True)
    for k in ("z_par_01", "z_par_23", "z_w1w2_12"):
        print(f"    {k} = {np.round(store[k], 6)}", flush=True)
    import compare_M
    _, mref = compare_M.load_mctdh(f"{PULSE_SET}:arm:{ARM}:{LEGS_TAG}")
    assert abs(mref["T"] - T_tot) < 1e-3, f"MCTDH arm T={mref['T']} vs {T_tot}"

    with jax.default_device(dev):
        w = ec.batch_ok()
        print(f"  batch self-test: batched vs per-state helpers agree to "
              f"{w:.2e}", flush=True)
        C16, bs0, _ = ec.load_primary_anchors()
        bsj = [jnp.asarray(b, dtype=jnp.complex128) for b in bs0]
        QB = {s: ec.qubit_basis16(s, C16, bs0)[0] for s in range(4)}
        q1s = [jnp.asarray(QB[s][:, 1], dtype=cdt) for s in range(4)]
        Vd = ec.build_Vd(rdt)

        runners = {}
        for key, hf, T in (("par", h_par, T_par), ("w1w2", h_ent, T_ent)):
            n = int(round(T / DT))
            runners[key] = (split_op.make_segment(hf, Vd, T / n, n, 0.0, cdt, rdt),
                            n, T)
            print(f"  leg '{key}': T={T}, {n} steps of dt={T/n:.6f}", flush=True)

        M = np.zeros((16, 16), complex)
        groups = [list(range(i, min(i + BATCH, 16))) for i in range(0, 16, BATCH)]
        print(f"  ensemble batched {BATCH} -> {len(groups)} group(s)", flush=True)

        for grp in groups:
            vh = np.empty((len(grp),) + (ec.N,) * 4, dtype=np.complex128)
            for k, j in enumerate(grp):
                vh[k] = np.asarray(ec.expand(C16[j], bsj))
            v = jnp.asarray(vh, dtype=cdt)
            del vh
            t_g = time.perf_counter()
            ts, ns, p16 = [0.0], [], []

            def snap(tt):
                o = np.stack([np.asarray(ec.occupations(q1s, v[k]))
                              for k in range(len(grp))])
                p = np.array([float(np.sum(np.abs(np.asarray(
                    ec.anchor_overlaps(C16, bsj, v[k]))) ** 2))
                    for k in range(len(grp))])
                ns.append(o); p16.append(p)

            snap(0.0)
            t = 0.0
            for i, (kind, val) in enumerate(sched):
                if kind == "local":
                    for s, u2 in val.items():
                        v = ec.apply_axis(ec.layer_U64(QB[s], u2), s, v)
                else:
                    run, n, T = runners[val]
                    _tf, v, _e = run(0.0, v)     # each leg's schedule starts at 0
                    jax.block_until_ready(v)
                    t += T
                    ts.append(t); snap(t)
                    print(f"    [{grp[0]}-{grp[-1]}] step {i+1}/{len(sched)} "
                          f"{val} done  t={t:.1f}  P16 min "
                          f"{p16[-1].min():.5f}  "
                          f"({(time.perf_counter()-t_g)/60:.1f} min)", flush=True)
                # ROLLING checkpoint: one file per group, overwritten.  Keeping
                # every schedule step would be 15 x 4 x 1.1 GB = 64 GB for a run
                # whose only resumable unit is "the last completed step anyway".
                if kind == "leg":
                    np.save(os.path.join(OUT, f"ckpt_g{grp[0]}.npy.tmp.npy"),
                            np.asarray(v, dtype=np.complex128))
                    os.replace(os.path.join(OUT, f"ckpt_g{grp[0]}.npy.tmp.npy"),
                               os.path.join(OUT, f"ckpt_g{grp[0]}.npy"))
                    json.dump(dict(step=i + 1, t=t),
                              open(os.path.join(OUT, f"ckpt_g{grp[0]}.json"), "w"))

            for k, j in enumerate(grp):
                M[:, j] = np.asarray(ec.anchor_overlaps(C16, bsj, v[k]))
                np.savez(os.path.join(OUT, f"state_{j:02d}.npz"),
                         ov_primary=M[:, j], t=np.array(ts),
                         n_s=np.stack(ns)[:, k], p16=np.stack(p16)[:, k],
                         norm_end=float(ec.norm2(v[k])), T=T_tot)
            print(f"  group {grp[0]}-{grp[-1]} done in "
                  f"{(time.perf_counter()-t_g)/60:.1f} min", flush=True)
            del v

        np.savez(os.path.join(OUT, "M.npz"), M_primary_cfg=M,
                 configs=np.array(ec.CONFIGS))
        json.dump(dict(tag=TAG, arm=ARM, pulse_set=PULSE_SET, prec=PREC,
                       T=T_tot, dt_fine=DT, e_shift=0.0, propagator="strang",
                       legs_tag=LEGS_TAG, legs16=os.path.relpath(LEGS16, ec.ROOT),
                       order=split_op.ORDER, n_legs=n_leg, n_local=n_loc,
                       total_steps=int(6 * runners["par"][1] + runners["w1w2"][1]),
                       summary=[dict(state=j, dnorm=0.0) for j in range(16)]),
                  open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(f"  -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
