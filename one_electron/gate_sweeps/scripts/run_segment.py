"""Segment-A / segment-B driver of the sqiSWAP sweep (mean-field spectators).

Runs lib/sweep_sa.py (exact checkpoint reuse, CMF half-step + per-chunk
regauge, gauge-corrected scoring) with its output directories redirected to
gate_sweeps/results/.

Differences from a plain sweep_sa run:
  * SWEEP_LEG_TYPE is asserted == "sqiswap" (the target gate of the sweep;
    "swap" would silently score Fbar against the wrong reference)
  * SWEEP_DEV=cuda is honoured here -- sweep_sa only applies jax.default_device
    inside its own __main__ guard, which this wrapper bypasses
  * SWEEP_DT is asserted unset (sweep_map reads it and rebuilds
    SNAP_DT = 20*SWEEP_DT, moving the 0.1 t.u. checkpoint grid)

    MODE=segA LEG=w0w1 TAU_RAMP=12.6 TH_MAX=169.0 python run_segment.py
    MODE=segB LEG=w0w1 TAU_RAMP=12.6 HOLDS=64.6,73.3 python run_segment.py
"""
from __future__ import annotations
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GS_DIR = os.path.abspath(os.path.join(HERE, ".."))          # .../gate_sweeps
LIB = os.path.abspath(os.path.join(GS_DIR, "..", "lib"))   # one_electron/lib
sys.path.insert(0, LIB)

LEG = os.environ["LEG"]
MODE = os.environ["MODE"]

# ── sweep configuration ───────────────────────────
os.environ.setdefault("SWEEP_INTEG", "cmf")
os.environ.setdefault("SWEEP_C_PROP", "diagonalize")
os.environ.setdefault("SWEEP_B_PROP", "diagonalize")
os.environ.setdefault("SWEEP_LEG_TYPE", "sqiswap")
os.environ.setdefault("MEANFIELD_SPEC", "1")
os.environ["GATE_LEG"] = LEG

assert os.environ["SWEEP_LEG_TYPE"] == "sqiswap", (
    "the sweep calibrates the sqiSWAP only; SWEEP_LEG_TYPE="
    f"{os.environ['SWEEP_LEG_TYPE']!r} would score against the wrong M_ref")
assert "SWEEP_DT" not in os.environ, (
    "SWEEP_DT is not SWEEP_DT_DIV -- sweep_map reads it and moves the SNAP_DT "
    "checkpoint grid. Unset it.")
assert os.environ.get("SWEEP_INTEG") == "cmf", "the sweep is CMF-only"

import jax
import sweep_sa as S

# ── redirect every output path into gate_sweeps/results/ ─────────────────────
# RESWEEP_TAG isolates throwaway smoke-test artifacts from the real sweep dirs,
# so a validation run can never contaminate aggregate_glob's pt_*.npz glob.
_TAG = os.environ.get("RESWEEP_TAG", "")
OUT = os.path.join(GS_DIR, "results",
                   f"{LEG}_{os.environ['SWEEP_LEG_TYPE']}"
                   f"{'_mf' if S.MEANFIELD_SPEC else ''}{_TAG}")
S.OUT_DIR = OUT
S.SEGA_DIR = os.path.join(OUT, "segA")
S.PTS_DIR = os.path.join(OUT, "sweep_pts")
for d in (S.OUT_DIR, S.SEGA_DIR, S.PTS_DIR):
    os.makedirs(d, exist_ok=True)

# Hard gate, evaluated BEFORE any propagation: every write target must live
# inside this directory and outside lib/.
_GUARD = os.path.join(GS_DIR, "results") + os.sep
for _d in (S.OUT_DIR, S.SEGA_DIR, S.PTS_DIR):
    assert _d.startswith(_GUARD), f"output outside gate_sweeps/results/: {_d}"
    assert not os.path.abspath(_d).startswith(LIB + os.sep), \
        f"output points into lib/: {_d}"
assert S.MEANFIELD_SPEC == (os.environ["MEANFIELD_SPEC"] == "1")
assert S.SWEEP_LEG_TYPE == "sqiswap"
assert abs(S.SNAP_DT - 0.1) < 1e-12, f"SNAP_DT moved to {S.SNAP_DT}"


def _main():
    tau_ramp = float(os.environ["TAU_RAMP"])
    delta = tau_ramp                       # erfc(9) tail -> exact reuse
    print(f"[resweep] leg={LEG} mode={MODE} gate={S.SWEEP_LEG_TYPE} "
          f"integ={S.SWEEP_INTEG} dt={S.SWEEP_DT:g} norb={S.NORB_LIST} "
          f"dev={os.environ.get('SWEEP_DEV','cpu')}", flush=True)
    print(f"[resweep] OUT={S.OUT_DIR}", flush=True)
    if MODE == "segA":
        S.run_segA(tau_ramp, float(os.environ["TH_MAX"]), delta)
    elif MODE == "segB":
        sega = os.path.join(S.SEGA_DIR, f"segA_tr{tau_ramp:g}.npz")
        assert os.path.exists(sega), f"missing segment A checkpoint: {sega}"
        for h in [float(x) for x in os.environ["HOLDS"].split(",")]:
            S.run_segB(tau_ramp, h, sega, delta)
    else:
        raise SystemExit(f"bad MODE={MODE}")


if __name__ == "__main__":
    # sweep_sa applies jax.default_device only inside its own __main__ guard,
    # which importing it as a module bypasses. Replicate it here.
    if os.environ.get("SWEEP_DEV", "cpu") == "cuda":
        with jax.default_device(jax.devices("cuda")[0]):
            _main()
    else:
        _main()
