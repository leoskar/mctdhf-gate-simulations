# Three electrons per well

Multilayer (ML-)MCTDHF on the four-well register with three electrons per well
(Sec. IV B of the paper, Apps. B 2 and B 3, and Secs. S3 B, S4 A-B, S5 F and
S6 B of the supplementary material). The base code is `mctdhf` 0.8.0 from the
multi-species-mctdhf repository.

Each folder is a Python package. Run everything from this folder:

    python -m <package>.<module> [arguments]

The operating-point code reuses the one-electron cost functions in
`../one_electron/operating_points/losses/` and starting points in
`../one_electron/operating_points/ITE_trajectories/`.

**Grid.** `system/fourwell.py` reads the number of sine-DVR points per well
from `FOURWELL_NUM_DVR` (default 64). Use `FOURWELL_NUM_DVR=80` for
`gates_80pt`, `orbital_convergence` and `sbs_check`, and leave it unset for the
other packages. The 80-point modules assert the grid, so a missing variable
stops them at once.

**Slurm.** `slurm/gpu.sbatch` and `slurm/cpu.sbatch` run one module, e.g.

    mkdir -p logs
    FOURWELL_NUM_DVR=80 sbatch slurm/gpu.sbatch -m gates_80pt.mf80 gate --leg w1w2 --norb 22 ...
    sbatch slurm/cpu.sbatch -m gate_timing.timing

They use `$PY` if set and otherwise the first `python` on `PATH`; partition,
memory and time limits are left to you. Each module's docstring lists its
commands.

Time unit: 1 t.u. = 1/(2π · 10 GHz) = 15.9155 ps.

## Pipeline

| Step | Package | What it gives |
|---|---|---|
| 1 | `system/` | the four-well system: Hamiltonian, operating points, register states |
| 2 | `operating_points/` | idle and target operating points on 64 points (Tables S3, S4) and the initial states |
| 3 | `gate_timing/` | timing windows and the 64-point mean-field sweeps (Sec. S5 F), mixed-precision test (Sec. S6 B) |
| 4 | `routed_gates/` | the three-electron schedules from the sweeps (Sec. S5 F); single-qubit gates (Secs. S4 A-B) |
| 5 | `gates_80pt/` | everything on the 80-point grid: spectra, register, operating points, schedules (Table S6), gates, integration checks (Sec. S6 B), spectator check |
| 6 | `orbital_convergence/` | hold scans (Fig. 6), orbital ladders and levels (Fig. 7) |
| 7 | `grid_convergence/` | the 64 → 128-point grid check (App. B 3) |
| 8 | `sbs_check/` | four vs six SBS per active well (Sec. IV B) |

## 1. `system/`

`fourwell.py` builds the four-well system and its control points;
`register.py` identifies the 16 register configurations as eigenstates of the
ML-projected Hamiltonian; `prep.py` relaxes ML states; `mixed.py` and
`_precision.py` hold the mixed-precision layer solves. `check_replica.py`
checks the system against reference energies of an earlier standard-MCTDHF
three-electron calculation (not included).

## 2. `operating_points/` (Sec. S3 B)

The published runs (14 orbitals and 4 SBS per well, 64 grid points):

    python -m operating_points.optimize_idle ml_20260923_idle
    IDLE=operating_points/results/ml_20260923_idle_stage1_singles_ground.npz
    python -m operating_points.optimize_target ml_sa16_20260924_target stage2_superpos_w0w1 $IDLE --relax 16 --lever beta1
    python -m operating_points.optimize_target ml_sa16_20260924_target stage3_superpos_w1w2 $IDLE --relax 16 --lever beta2
    python -m operating_points.optimize_target ml_sa16_20260924_target stage4_superpos_w2w3 $IDLE --relax 16 --lever beta3
    python -m operating_points.optimize_target ml_sa16_20260924_target stage5_superpos_w0w3 $IDLE --relax 16 --lever beta0
    python -m operating_points.prepare_states      # the 16 idle states used as initial states
    python -m operating_points.validate <test>     # equiv, traced, loss, fixedpoint, smoke

The target runs are independent; each freezes the spectators at the idle
optimum and starts the active wells from the one-electron target point, after
the resonance search along `--lever`. Table S4 lists stages 2-4; the w0w3
target (stage 5) is used only by `gate_timing/timing.py`. Results in
`results/`: the idle run `ml_20260923_idle_stage1_singles_ground*`, the targets
`ml_sa16_20260924_target_stage{2,3,4,5}_*` and the initial states
`idle_states_sa16_ns4_norb14.{npz,json}`. The `*.evals.jsonl` files log every
cost evaluation. `validate equiv` compares against checkpoints that are not
included.

## 3. `gate_timing/` (Secs. S5 F, S6 B)

    python -m gate_timing.timing                                       # hold-time and ramp windows, sweep grid
    python -m gate_timing.sweep segA --leg w1w2 --tau-ramp 28.3 --grid  # mean-field sweep, see the docstring

`timing.py` applies the rules of Secs. S5 A-C to the three-electron operating
points (with the 1→2 spacing as ω_min, Sec. S5 F) and writes
`results/timing/{physics.json,tables.md}` and the sweep grid `grids.json`; the
paper does not tabulate these windows. `sweep.py` runs the 64-point sweeps with
14 orbitals and 4 SBS per active well and mean-field spectators; the maps are
`results/<gate>/map_<gate>_<settings>.npz` and the points
`results/<gate>/<settings>/pt_*.npz`. `integ_study.py` holds the
mixed-precision test of Sec. S6 B (`results/study/rep_w1w2_tr28.3/`,
2.5 × 10⁻⁹ against double precision). `exact_check.py` and `level_check.py` are
the diagnostics of the |11⟩ leakage; `exact_check.py` also gives the 64-point
gates with 18 and 20 orbitals compared in App. B 3.

## 4. `routed_gates/`

    python -m routed_gates.routed pulses    # schedules from the 64-point sweep maps
    python -m routed_gates.routed relax     # register in double precision
    python -m routed_gates.routed check     # single-qubit gates, no propagation

`pulses` takes the second-shortest ramp time of each gate and interpolates the
hold time to θ = π/2 (Sec. S5 F) → `results/pulses.json`. `check` evaluates the
single-qubit gates of Secs. S4 A-B on the idle register (`results/check.npz`):
Hadamard F̄ = 0.998–0.9998 (0.42–0.47 with the one-body operator), HS† on a
pair 0.9986–0.9993. The other modes of
`routed.py` (the routed circuits with three electrons) are not part of the paper
and no results are included.

## 5. `gates_80pt/` (FOURWELL_NUM_DVR=80)

| Module | Paper |
|---|---|
| `spectra80.py` | 80-point detunings (App. B 3: 0.003 Δ against 128 points; Δ changes by ≤ 0.06 %) |
| `register80.py`, `oppoints80.py` | idle register and target points re-evaluated on 80 points (Tables S3, S4) |
| `pulses80.py` | the schedules moved to 80 points by the ratio of the splittings (Table S6) |
| `mf80.py` | gates with mean-field spectators, 14-22 orbitals: θ = π/2 within 5 × 10⁻⁴ at 22 orbitals; retention +0.15–0.18 against 64 points (App. B 3); integration checks of Sec. S6 B (`compare_g4_*.json`: 1.3 × 10⁻⁶, 5.9 × 10⁻⁹, 1.7 × 10⁻⁶) |
| `fullsys80.py` | spectators with 16 orbitals (Sec. IV B: changes < 10⁻²) |

The files `g4_*` and `compare_g4_*` are the integration checks of `mf80.py`.

## 6. `orbital_convergence/` (FOURWELL_NUM_DVR=80)

| Module | Paper |
|---|---|
| `holdscan80.py` | hold scans at 22 orbitals, Fig. 6 (`results/holdscan/`, summaries `results/holdscan_summary*.json`) |
| `ladder80.py` | the gates at 6-26 orbitals, Fig. 7(a) (`results/ladderp3_*`) |
| `ramp80.py` | levels in the orbitals at the end of the ramp, Fig. 7(b) (`results/ramp80/`) |
| `levels_hartree80.py` | the frozen mean field and the 80-orbital reference lines of Fig. 7(b) |
| `levels_own80.py`, `dressed80.py` | earlier level studies that led to `ramp80.py` |
| `overlap_patch.py` | memory-light determinant overlaps for 26 orbitals |
| `analyse.py` | tables from the result files; `analyse holdscan <gate>` rewrites the hold-scan summaries |

The ladder files of Fig. 7(a) are named `ladderp3_*` after the `--pulse p3`
option of `ladder80.py`, which selects each gate's θ = π/2 schedule.

## 7. `grid_convergence/` (App. B 3)

    python -m grid_convergence.grid_check    # detunings on 64, 80, 96, 128 points -> results/grid_check.json
    python -m grid_convergence.spectra       # single-well spectra and centre-of-mass check

With 64 points the detunings differ from 128 points by up to 0.48 Δ.
`results/w1w2/gate_ladder_act22s4_*.npz` is the 64-point gate with 22 orbitals
used in the retention comparison; the script that made it is not included.

## 8. `sbs_check/` (FOURWELL_NUM_DVR=80)

    python -m sbs_check.sbs relax|pilot|gate --leg <gate> --nspf 6
    python sbs_check/summary.py

Six SBS per active well against four: all changes < 10⁻² (Sec. IV B).

## Data

Result files keep their original names; the large state arrays (`y`, `y0`,
`y_t`, `ys`) were removed and are listed per folder in `REMOVED_KEYS.md`.
Restarting a calculation from a stored state therefore needs that state to be
recomputed first.

Some checks (`system/check_replica.py`, `gate_timing/timing.py`) compare with
reference numbers from an earlier standard-MCTDHF three-electron calculation,
which is not included.
