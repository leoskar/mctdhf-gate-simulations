# mctdhf-gate-simulations

Simulation scripts and data for

> O. Leinonen, *Multi-species MCTDHF for gate simulations in multi-qubit hardware* (2026).

The paper simulates √iSWAP gates in a register of four anharmonic wells, each
holding one qubit, with one electron per well (MCTDH) and with three electrons
per well (multilayer MCTDHF). The solver is the `mctdhf` package from the
companion repository [multi-species-mctdhf](https://github.com/leoskar/multi-species-mctdhf); this
repository holds everything specific to the paper.

The scripts are the ones that produced the published results. For this
release, paths, imports, names and comments were adapted, and code not used for
the paper was removed.

The solver grew from a spin-unrestricted MCTDHF code written by the author. Its
extensions to several species and to the multilayer form were written with an
AI coding assistant (Claude, Anthropic) and tested against that code (see the
multi-species-mctdhf README). The simulation scripts in this repository were also
written with Claude under the author's direction, continuing from the author's
initial simulations. The author is responsible for the results, which are
checked by:

- the untruncated 64⁴ reference propagation (App. C, Table I);
- convergence in grid points, orbitals, SBS and time step (Sec. IV B, App. B 3,
  Sec. S6);
- version 0.8.0 reproducing the one-electron results of version 0.7.3;
- all figures and tables regenerating from the stored data.

## Layout

    one_electron/       one electron per well (Sec. IV A)          -> one_electron/README.md
      lib/                shared code: geometry, propagation, scoring
      operating_points/   idle and target operating points
      grid_convergence/   64-point grid check
      gate_sweeps/        mean-field sweeps, four-orbital checks, time-step check
      routed_gates/       direct and routed √iSWAP(0,3)
      exact_reference/    untruncated 64^4 reference propagation
    three_electrons/    three electrons per well (Sec. IV B)       -> three_electrons/README.md
      system/  operating_points/  gate_timing/  routed_gates/  gates_80pt/
      orbital_convergence/  grid_convergence/  sbs_check/  slurm/
    figures/            scripts for Figs. 4-7 and S1, named after the paper's figure files
    paper_figures/      TikZ sources of Figs. 1-3

Each folder keeps its results next to its scripts, in `results/`. The two
READMEs above give the commands of each step.

## Installation

    python -m venv .venv
    source .venv/bin/activate
    pip install "multi-species-mctdhf @ git+https://github.com/leoskar/multi-species-mctdhf"   # or: pip install -e ../multi-species-mctdhf
    pip install -r requirements.txt

For a GPU, install `jax[cuda12]`. The production runs used one GPU each; the
analysis, the tables and the figures run on a CPU from the stored results. The
figure scripts typeset with LaTeX (`text.usetex`, `mathptmx`), so they need a
TeX installation.

## Where each result comes from

Paths are relative to this folder. 1e = `one_electron/`, 3e = `three_electrons/`.

**Main text**

| Item | Script | Data |
|---|---|---|
| Fig. 1 | `paper_figures/fig_ml.tex` | |
| Fig. 2 | `paper_figures/swap_decomposition.tex` | |
| Fig. 3 | `paper_figures/circuit_routed.tex` | |
| Fig. 4 | `figures/mf_fig_wide.py` | 1e `gate_sweeps/results/`: `*_sqiswap_mf/sweep_pts/` (mean field), `*_sqiswap/sweep_pts/` and `norb4_confirm/` (four orbitals) |
| Sec. IV A 1 (F̄₄, F̄₁₆ of the optima) | 1e `gate_sweeps/scripts/confirm_norb4.py`, `confirm_k16.py`, `rescore_k16.py` | 1e `gate_sweeps/results/{norb4,k16}_confirm/` |
| Sec. IV A 2 (F̄₁₆ and durations of the routings) | 1e `routed_gates/scripts/aggregate.py` | 1e `routed_gates/results/arms_*/`, `gate_sweeps/results/k16_confirm/` |
| Fig. 5 | `figures/methods_fid.py` | 1e `routed_gates/results/prodmats_*.npz` (from `impl_locals.py`) |
| Table I | 1e `exact_reference/analysis.sh` (`exact_channels.py`) | 1e `exact_reference/results/` |
| Fig. 6 | `figures/3e_gate.py` | 3e `orbital_convergence/results/holdscan*`, `gates_80pt/results/mf80_leg_*_n22.npz` |
| Fig. 7 | `figures/3e_convergence.py` | 3e `orbital_convergence/results/{ladderp3_*,ramp80/,levels_hartree80/}`, `gates_80pt/results/` |
| Sec. IV B, spectators with 16 orbitals | 3e `gates_80pt/fullsys80.py` | 3e `gates_80pt/results/fs80_leg_*` |
| Sec. IV B, six SBS | 3e `sbs_check/` | 3e `sbs_check/results/` |
| App. B 3, one electron | 1e `grid_convergence/run_dvr_convergence.py` | 1e `grid_convergence/results/` |
| App. B 3, three electrons | 3e `grid_convergence/grid_check.py`, `gates_80pt/spectra80.py`, `gates_80pt/mf80.py` | 3e `grid_convergence/results/`, `gates_80pt/results/` |
| App. C | 1e `exact_reference/` | 1e `exact_reference/results/` |

**Supplementary material**

| Item | Script | Data |
|---|---|---|
| Tables S1, S2 | 1e `operating_points/print_tables.py` | 1e `operating_points/ITE_trajectories/` |
| Tables S3, S4 | 3e `operating_points/` (64 points); costs and populations on 80 points from 3e `gates_80pt/register80.py`, `oppoints80.py` | 3e `operating_points/results/`, `gates_80pt/results/` |
| Sec. S4 B, one electron | 1e `routed_gates/scripts/impl_locals.py` | 1e `routed_gates/results/impl_locals_*.json` |
| Secs. S4 A-B, three electrons | 3e `routed_gates/routed.py check` | 3e `routed_gates/results/check.npz` |
| Sec. S4 C | 1e `routed_gates/scripts/canonical_frames.py` | 1e `routed_gates/results/arms_*/legs16_symz.npz` |
| Table S5 | 1e `gate_sweeps/scripts/grid.py` | |
| Sec. S5 D | 1e `gate_sweeps/scripts/run_segment.py` | |
| Fig. S1 | `figures/1e_mf_sweep.py` | 1e `gate_sweeps/results/*_sqiswap_mf/*.csv` |
| Table S6, one electron | 1e `gate_sweeps/` (optima), `gate_sweeps/scripts/opt_shared_pulse.py` (synchronized) | 1e `routed_gates/slurm/_common.sh` lists all schedules |
| Sec. S5 F and Table S6, three electrons | 3e `gate_timing/`, `routed_gates/routed.py pulses`, `gates_80pt/pulses80.py` | 3e `routed_gates/results/pulses.json`, `gates_80pt/results/pulses80.json` |
| Sec. S6 A | 1e `gate_sweeps/timestep_check/`, `routed_gates/slurm/job_dt_test.sh` | 1e `gate_sweeps/timestep_check/results/`, `routed_gates/results/arms_fast/` |
| Sec. S6 B | 3e `gates_80pt/mf80.py compare`, `gate_timing/integ_study.py` | 3e `gates_80pt/results/compare_g4_*.json`, `gate_timing/results/study/` |

The figure scripts are run from any folder, e.g. `python figures/3e_gate.py`, and
write their PDF and PNG next to themselves. Fig. 4, Fig. S1 and Fig. 5 were made
with matplotlib 3.10.9 and Figs. 6 and 7 with 3.11.2; with these versions the
output is pixel-identical to the paper's figures. Other matplotlib versions can
shift the layout by a pixel. `swap_decomposition.tex` needs a recent `quantikz`
(it does not compile with TeX Live 2022).

## Data

The results are the stored outputs of the published runs, 36 MB in total. Large
arrays were removed to keep the repository small: wavefunction checkpoints
(including the 17 GB of the 64⁴ reference runs) and state trajectories. Every file
keeps its original name, and the removed keys are listed in a
`REMOVED_KEYS.md` in each affected folder. Everything removed can be
regenerated by rerunning the step that made it; a run that restarts from a
stored state needs that state to be recomputed first.

Provenance fields in the result files (JSON metadata and string fields of
`.npz` files) give paths relative to this folder or to the package that wrote
them. The scripts find their inputs relative to the repository and do not read
these fields.

## Software versions

The three-electron results were computed with `mctdhf` 0.8.0, Python 3.12.1 and
jax 0.9.1. The one-electron results were computed with the single-layer code at
version 0.7.3, Python 3.14.1 and jax 0.10.0; version 0.8.0 reproduces them (see
the multi-species-mctdhf README).

## License

The code is licensed under the MIT License (`LICENSE`) and the data under
CC BY 4.0 (`LICENSE-DATA`, which also lists which files count as data).

## Citation

If you use this code or data, please cite the paper above.
