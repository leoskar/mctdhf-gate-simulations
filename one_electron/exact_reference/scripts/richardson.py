"""Richardson-extrapolate a pair of runs at dt and 2*dt to the dt->0 limit.

For a method of order p in the state error, F(dt) = F(0) + C dt^p, so

    F(0) = F(dt) + [F(dt) - F(2dt)] / (2^p - 1)

with p = 4 for the Yoshida composition.  Valid only in the asymptotic regime,
which holds from dt = 0.5 for this system (factor 274 per halving against 256
predicted).  The residual after extrapolation is reported as
the remaining uncertainty -- it is NOT zero, and the point of this script is to
put an error bar on the number rather than to make one disappear.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact_common as ec

FINE, COARSE = os.environ["TAG_FINE"], os.environ["TAG_COARSE"]
P = int(os.environ.get("ORDER", "4"))


def load(tag):
    d = np.load(os.path.join(ec.RESULTS, tag, "M.npz"))
    m = json.load(open(os.path.join(ec.RESULTS, tag, "meta.json")))
    return ec.to_bit_order(d["M_primary_cfg"]), m


Mf, mf = load(FINE)
Mc, mc = load(COARSE)
assert abs(mf["T"] - mc["T"]) < 1e-3, "different circuits"
r = mc["dt_fine"] / mf["dt_fine"]
print(f"  fine dt={mf['dt_fine']}  coarse dt={mc['dt_fine']}  ratio {r:g}  order {P}")

Ff, xf, _ = ec.fbar16(Mf)
Fc, xc, _ = ec.fbar16(Mc)
den = r ** P - 1
Fx = Ff + (Ff - Fc) / den
xx = xf + (xf - xc) / den
print(f"  F_bar16   coarse {Fc:.8f}   fine {Ff:.8f}   extrapolated {Fx:.8f}")
print(f"  fine-run step error (|fine - extrapolated|) = {abs(Ff-Fx):.3e}")
print(f"  F_xc      coarse {xc:.8f}   fine {xf:.8f}   extrapolated {xx:.8f}")

# THREE-POINT CHECK.  With a third, coarsest run the same limit can be reached
# from two independent pairs: (coarsest, coarse) and (coarse, fine).  If they
# agree, the scheme is asymptotic across this whole dt range and a single-pair
# extrapolation elsewhere is trustworthy.  If they disagree, the extrapolation
# itself is outside its regime -- which no amount of extra precision in one pair
# would reveal.
if os.environ.get("TAG_COARSEST"):
    Mx, mx = load(os.environ["TAG_COARSEST"])
    Fx2 = ec.fbar16(Mx)[0]
    r2 = mx["dt_fine"] / mc["dt_fine"]
    Fx_from_coarse_pair = Fc + (Fc - Fx2) / (r2 ** P - 1)
    print(f"\n  three-point check (dt = {mx['dt_fine']}, {mc['dt_fine']}, "
          f"{mf['dt_fine']}):")
    print(f"    extrapolated from the COARSE pair = {Fx_from_coarse_pair:.8f}")
    print(f"    extrapolated from the FINE   pair = {Fx:.8f}")
    d = abs(Fx_from_coarse_pair - Fx)
    print(f"    the two limits differ by {d:.3e}   "
          f"{'PASS -- asymptotic across the range' if d < 5e-5 else 'FAIL -- not asymptotic'}")
    print(f"    (for scale, the fine-pair correction was {abs(Ff-Fx):.3e})")

# a single leg also scored against its OWN sqiSWAP target (confirm_k16
# convention); everything above stays on the (0,3) target
import compare_M
PAIR = compare_M.PAIRS.get(mf.get("leg"))
OWN = PAIR is not None and tuple(PAIR) != (0, 3)
if OWN:
    Pf, Pc = ec.fbar16(Mf, PAIR)[0], ec.fbar16(Mc, PAIR)[0]
    Px = Pf + (Pf - Pc) / den
    print(f"\n  vs sqiSWAP{PAIR}: F_bar16 coarse {Pc:.8f}   fine {Pf:.8f}   "
          f"extrapolated {Px:.8f}   fine-run step error {abs(Pf-Px):.3e}")
    if os.environ.get("TAG_COARSEST"):
        Px2 = ec.fbar16(Mx, PAIR)[0]
        Px_cp = Pc + (Pc - Px2) / (r2 ** P - 1)
        print(f"    three-point: COARSE pair {Px_cp:.8f}   FINE pair {Px:.8f}   "
              f"differ by {abs(Px_cp - Px):.3e}")

if os.environ.get("MCTDH"):
    B, mb = compare_M.load_mctdh(os.environ["MCTDH"])
    FB = ec.fbar16(B)[0]
    print(f"\n  MCTDH F_bar16 = {FB:.8f}")
    print(f"  dF_bar16 at fine dt      = {Ff - FB:+.3e}")
    print(f"  dF_bar16 EXTRAPOLATED    = {Fx - FB:+.3e}   <-- quote this")
    print(f"  residual reference error = {abs(Ff-Fx):.3e}")
    if OWN:
        PB = ec.fbar16(B, PAIR)[0]
        print(f"  vs sqiSWAP{PAIR}: MCTDH F_bar16 {PB:.8f}   dF_bar16 fine "
              f"{Pf - PB:+.3e}   EXTRAPOLATED {Px - PB:+.3e}   residual "
              f"{abs(Pf-Px):.3e}")
