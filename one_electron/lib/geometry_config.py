"""Geometry of the four-well system, with the operating-point (anchor) files
and the result-folder tag that belong to it.

``GEOMETRY`` selects the well spacing; the paper uses "new_2um" (wells 2 µm
apart).  The anchor files are tied to the geometry here, so the grid and the
anchors cannot mismatch.  The module needs only ``numpy`` and ``pint``, so
importing it does not build a solver.
"""
from __future__ import annotations

import os

import numpy as np
import pint

# ── Toggle ────────────────────────────────────────────────────────────────────
GEOMETRY = "new_2um"   # "new_2um" | "old_1um"

# ── Shared (geometry-independent) constants ───────────────────────────────────
NUM_WELLS = 4
NUM_DVR = 64
TYPICAL_FREQ_GHZ = 10.0

# pint unit system: the same omega, length unit and Coulomb constant as
# optimize_qubit_params.py.
_ureg = pint.UnitRegistry()
_omega = 2 * np.pi * TYPICAL_FREQ_GHZ * _ureg.GHz
_energy_unit = _ureg.hbar * _omega
_length_unit = np.sqrt(_ureg.hbar / (_ureg.m_e * _omega))
_time_unit = _ureg.hbar / _energy_unit
LENGTH_UNIT_UM = _length_unit.to(_ureg.micrometer).magnitude
TIME_UNIT_PS = _time_unit.to(_ureg.picosecond).magnitude
KAPPA = (
    _length_unit * _ureg.m_e * _ureg.e ** 2
    / (4 * _ureg.pi * _ureg.epsilon_0 * _ureg.hbar ** 2)
).to(_ureg.dimensionless).magnitude

WELL_WIDTHS_UM = np.array([1.0, 1.0, 1.0, 1.0])

_HERE = os.path.dirname(os.path.abspath(__file__))
# The optimized operating points (imaginary-time optimizer output).
_ITE = os.path.join(_HERE, "..", "operating_points", "ITE_trajectories")


def _a(name):
    return os.path.join(_ITE, name)


# ── Per-geometry definitions ──────────────────────────────────────────────────
if GEOMETRY == "new_2um":
    WELL_CENTERS_UM = np.array([-3.0, -1.0, 1.0, 3.0])
    GEOM_TAG = "geom_2um"
    # inner adjacent pair (wells 1, 2) for the 2-well FCI reduction (grid check)
    TWO_WELL_CENTERS_UM = np.array([-1.0, 1.0])
    _F = "chain_frozen_20260621_140848"
    ANCHOR_PATHS = {
        1:    _a("new_geometry_step1_stage1_singles_ground.npz"),
        2:    _a(f"{_F}_step2_stage2_superpos_w0w1.npz"),
        # The frozen chain has a single w1w2 / w2w3 anchor each (no _from1
        # duplicates), so the 'a'/'b' keys resolve to the same file.
        "3a": _a(f"{_F}_step3_stage3_superpos_w1w2.npz"),
        "3b": _a(f"{_F}_step3_stage3_superpos_w1w2.npz"),
        "4a": _a(f"{_F}_step4_stage4_superpos_w2w3.npz"),
        "4b": _a(f"{_F}_step4_stage4_superpos_w2w3.npz"),
        5:    _a(f"{_F}_step5_stage5_superpos_w0w3.npz"),
    }
elif GEOMETRY == "old_1um":
    WELL_CENTERS_UM = np.array([-1.5, -0.5, 0.5, 1.5])
    GEOM_TAG = "geom_1um"
    TWO_WELL_CENTERS_UM = np.array([-0.5, 0.5])
    _G = "chain_global_20260512_181719"
    ANCHOR_PATHS = {
        1:    _a("chain_step1.npz"),
        2:    _a(f"{_G}_step2_stage2_superpos_w0w1.npz"),
        "3a": _a(f"{_G}_step3_stage3_superpos_w1w2.npz"),
        "3b": _a(f"{_G}_step3_from1_stage3_superpos_w1w2.npz"),
        "4a": _a(f"{_G}_step4_stage4_superpos_w2w3.npz"),
        "4b": _a(f"{_G}_step4_from1_stage4_superpos_w2w3.npz"),
        5:    _a(f"{_G}_step5_from1_stage5_superpos_w0w3.npz"),
    }
else:
    raise ValueError(
        f"unknown GEOMETRY={GEOMETRY!r}; use 'new_2um' or 'old_1um'"
    )

# Derived dimensionless geometry (convenience for consumers; consumers needing a
# specific dtype / jnp array convert these themselves).
WELL_CENTERS = WELL_CENTERS_UM / LENGTH_UNIT_UM
WELL_WIDTHS = WELL_WIDTHS_UM / LENGTH_UNIT_UM
