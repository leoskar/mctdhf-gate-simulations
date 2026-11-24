"""The theta = pi/2 schedules on 80 points (Table S6): the 64-point ramps
(routed_gates/results/pulses.json), holds rescaled to the 80-point exchange.

    python -m gates_80pt.pulses80 [--correct LEG=Delta_eff ...]

T_eff,80 = T_eff,64 * Delta_64 / Delta_80 (T_eff = tau_ramp + tau_hold),
Delta_80 from results/oppoints80.json, tau_hold = T_eff,80 - tau_ramp
snapped to the 0.1 t.u. block (tau_ramp is already on it). ``--correct``
rescales one leg's hold by a measured Delta_eff instead,
T_eff = (pi/2) / Delta_eff (not used for the published schedules). The
output is results/pulses80.json.
"""
import argparse
import os

import numpy as np

from . import common as T

BLOCK = 0.1


def snap(v):
    return round(round(v / BLOCK) * BLOCK, 6)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--correct", nargs="*", default=[])
    a = p.parse_args(argv)
    op = T.load_json(os.path.join(T.RESULTS, "oppoints80.json"))
    path = os.path.join(T.RESULTS, "pulses80.json")
    out = T.load_json(path) if os.path.exists(path) else {}
    corr = dict(c.split("=") for c in a.correct)
    for leg in T.LEGS3:
        tr = T.REF64["ramp"][leg]
        if leg in corr:
            De = float(corr[leg])
            Teff = (np.pi / 2) / De
            prev = out.get(leg)
            out[leg] = dict(tau_ramp=tr, tau_hold=snap(Teff - tr),
                            T_eff_exact=Teff, rule="corrected",
                            Delta_eff=De, previous=prev)
        elif leg not in out or out[leg].get("rule") != "corrected":
            r = op.get(leg, {})
            if r.get("status") != "ok":
                print(f"  {leg}: no oppoints80 result, skipped")
                continue
            D80, D64 = r["final"]["Delta"], T.REF64["Delta"][leg]
            Teff = T.REF64["T_eff"][leg] * D64 / D80
            out[leg] = dict(tau_ramp=tr, tau_hold=snap(Teff - tr),
                            T_eff_exact=Teff, rule="Delta_64/Delta_80",
                            Delta_80=D80, Delta_64=D64,
                            T_eff_64=T.REF64["T_eff"][leg])
        o = out[leg]
        o["T"] = round(2 * o["tau_ramp"] + o["tau_hold"], 6)
        o["T_eff"] = round(o["tau_ramp"] + o["tau_hold"], 6)
        print(f"  {leg}: tau_ramp {o['tau_ramp']}  tau_hold {o['tau_hold']}  "
              f"T_eff {o['T_eff']} (exact {o['T_eff_exact']:.3f})  T {o['T']}"
              f"  [{o['rule']}]")
    T.save_json(path, out)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
