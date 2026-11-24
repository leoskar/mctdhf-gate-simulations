# One electron per well

Single-layer MCTDH on the four-well register with one electron per well
(Sec. IV A of the paper, Apps. B 1, B 3 and C, and Secs. S3 A, S4 B-C, S5 and
S6 A of the supplementary material). The base code is `mctdhf` from the multi-species-mctdhf
repository. These results were computed with its single-layer code at version
0.7.3; version 0.8.0 reproduces them (see that repository's README).

Every script is run from its own folder (the folder named in each section
below). The scripts put `lib/` on `sys.path` themselves. The Slurm scripts
assume the Python environment is already active, write logs to `logs/` (run
`mkdir -p logs` first), and leave partition, memory and time limits to you.

Time unit: 1 t.u. = 1/(2π · 10 GHz) = 15.9155 ps.

## Pipeline

| Step | Folder | What it gives |
|---|---|---|
| 1 | `operating_points/` | idle and target operating points, Tables S1, S2 |
| 2 | `grid_convergence/` | 64-point grid check (App. B 3) |
| 3 | `gate_sweeps/` | mean-field ramp/hold sweeps (Fig. S1), four-orbital hold rows (Fig. 4), confirmed schedules (Table S6), time-step check (Sec. S6 A) |
| 4 | `routed_gates/` | direct and routed √iSWAP(0,3) circuits (Sec. IV A 2), Fig. 5 data, single-qubit-gate error (Sec. S4 B) |
| 5 | `exact_reference/` | untruncated 64⁴ reference propagation (App. C), Table I |

`lib/` holds the shared code: the geometry and operating points
(`geometry_config.py`), the Hamiltonian and propagation step (`rte_step.py`),
gate legs and sweeps (`qoct_gate_leg.py`, `sweep_map.py`, `sweep_sa.py`), the
16-state register runs and scoring (`run_route_sa.py`, `run_route_par.py`,
`run_direct_sa.py`, `run_direct_sa_gauge.py`, `common.py`), the gauge-corrected
fidelity (`gate_fidelity.py`), the error-channel decomposition of Sec. III B 3
(`error_channels.py`) and the QR re-orthonormalization (`regauge.py`). Some lib
modules create an empty `lib/results/` at import; it is git-ignored.

## 1. Operating points — `operating_points/`

    sbatch slurm/job_idle.sh               # idle point (stage 1)
    ./slurm/submit_targets.sh              # the four target points (stages 2-5)
    python print_tables.py                 # Tables S1 and S2, from the stored runs

`losses/` holds the cost functions of the five stages. The optimizer output of
the published runs is in `ITE_trajectories/` (base name
`chain_frozen_20260621_140848` for the targets). `print_tables.py` re-evaluates
each stored optimum and asserts that it reproduces the stored cost.

## 2. Grid convergence — `grid_convergence/`

    python run_dvr_convergence.py

Lowest eigenvalues of a pair of neighbouring wells for 32-96 grid points
(App. B 3). Result: `results/geom_2um/dvr_convergence.npz`.
`two_well_config_geom_2um.npz` is the two-well configuration the check uses;
it comes from an earlier two-well optimization that is not included.

## 3. Gate sweeps — `gate_sweeps/`

Mean-field sweeps: one orbital in each spectator well, four in each active well.

1. `python scripts/grid.py` writes the sweep grid as the manifests
   `scripts/manifest_seg{A,B}_{cpu,gpu}.txt`. The windows in `grid.py` (Table S5)
   follow from the idle and target eigenstates by the rules of Secs. S5 A-C;
   that derivation is not included as a script, but
   `three_electrons/gate_timing/timing.py` applies the same rules.
2. Segment A, one propagation per (gate, ramp time), storing the state every
   0.1 t.u.: `slurm/job_segA_{cpu,gpu}.sh`. Segment B restarts each hold time
   from the stored state (Sec. S5 D): `slurm/job_segB_{cpu,gpu}.sh`.
3. `python scripts/fine_pass.py` adds the finer hold times around each maximum
   (`scripts/manifest_segB_fine.txt`, also run with `job_segB_*`).
4. `python scripts/aggregate.py` and `python scripts/export_tables.py` give the
   maxima and the tables `results/<gate>_sqiswap_mf/{coarse,fine}_sweep_fidelities.csv`
   that `figures/1e_mf_sweep.py` plots.

Four-orbital checks:

* `slurm/job_norb4.sh` (`confirm_norb4.py`) propagates each optimum with four
  orbitals in every well: F̄₄ = 0.99600, 0.99927, 0.99799 (Sec. IV A 1).
  Results: `results/norb4_confirm/`.
* `slurm/job_k16_{cpu,gpu}.sh` (`confirm_k16.py`) then `python scripts/rescore_k16.py`
  score the same gates on the 16-state register: F̄₁₆ = 0.9413, 0.9983, 0.9977.
  Results: `results/k16_confirm/` (`*_rescored.npz` are the paper's numbers).
* The four-orbital hold rows of Fig. 4: `slurm/job_holdrow_*.sh` and
  `slurm/job_norb4_rows_{cpu,gpu}.sh` with their manifests. Results:
  `results/{w0w1,w2w3,w0w3}_sqiswap/sweep_pts/`; the w1w2 row is in
  `results/norb4_confirm/`.

Synchronized schedule (Table S6): `python scripts/opt_shared_pulse.py` and
`python scripts/plot_shared_pulse.py`. They need the shared-pulse maps, 43 MB
each, which are not included; rerun `opt_shared_pulse.py` to make them.

Time-step check (Sec. S6 A), in `timestep_check/`:

    LEG=sqi12 INTEG=cmf N_CHUNKS=1637 SCREEN_DEV=cuda DT_DIV=1   sbatch job_screen.sh
    LEG=sqi12 INTEG=cmf N_CHUNKS=1637 SCREEN_DEV=cuda DT_DIV=0.1 SCREEN_TAG=_rg sbatch job_screen.sh
    python cmp_screen.py results/screen_sqi12_cmf_1_cuda.npz results/screen_sqi12_cmf_0.1_cuda_n1637_rg.npz

Stored sweep points (`sweep_pts/pt_*.npz`) keep the gate matrices and scores
but not the state trajectory along the ramp-down (see `REMOVED_KEYS.md` in each
folder). The segment-A checkpoints are not included, so rerunning a segment-B
point needs its segment A first.

## 4. Routed and direct long-range gates — `routed_gates/`

`slurm/_common.sh` holds the three pulse sets: `best` (the individually
optimized schedules), `sync` (the synchronized routing) and `fast` (shorter
ramps, used only in the time-step test).

1. `sbatch slurm/job_legs16.sh`: per-gate 16×16 matrices at 0.80 ps (`best`,
   `fast`).
2. `sbatch slurm/job_legs16_4dt.sh`: the same at 0.32 ps (`best`, `sync`).
3. `python scripts/canonical_frames.py`: the local-Z frames of Sec. S4 C
   (`legs16_symz.npz`).
4. `sbatch slurm/job_symz.sh`: the serial, parallel and synchronized routings
   at 0.32 ps.
5. `PULSE_SET=all python scripts/aggregate.py` prints F̄₁₆ and the gate
   durations: direct 0.966 / 37.0 ns, serial 0.873 / 41.1 ns, parallel
   0.823 / 26.0 ns, synchronized 0.856 / 23.1 ns.
6. `PULSE_SET=best python scripts/impl_locals.py` and `PULSE_SET=sync python scripts/impl_locals.py`:
   the 4×4 and 16×16 product models of Fig. 5 (`results/prodmats_*.npz`,
   `results/impl_locals_*.json`), and the single-qubit-gate error of Sec. S4 B
   (0.931 → 0.875, 0.891 → 0.828, 0.915 → 0.836).

`sbatch slurm/job_dt_test.sh` is the 0.32 vs 0.16 ps test of Sec. S6 A (needs
`results/arms_fast/legs16.npz`). The direct gate is `gate_sweeps`' K=16
confirmation of w0w3. Arm files keep the gate matrices and diagnostics; the
per-step checkpoints are not included.

## 5. Untruncated reference — `exact_reference/`

    EXACT_LEG=w0w1 DT_FINE=0.125 sbatch slurm/job_leg.sh   # w0w1, w1w2, w2w3 at 0.125, 0.25, 0.5; w0w3 at 0.25, 0.5
    DT_FINE=0.25 sbatch slurm/job_arm.sh                   # synchronized routing, also at 0.5
    ./analysis.sh                                           # Table I, from the stored M.npz (CPU, minutes)

Each run folder in `results/` keeps `M.npz` (the 16×16 matrix), `meta.json` and
the final states; the full 64⁴ checkpoints (17 GB) are not included.

## Notes

* `optimize_qubit_params.py` and `lib/geometry_config.py` were edited after the
  published optimization runs. The stored runs in `ITE_trajectories/` are the
  reference; `print_tables.py` re-evaluates them with the current code.
* The idle-point optimization ran in single precision (complex64).
* `scripts/manifest_segB_fine.txt` was rebuilt from the fine-sweep tables,
  because the manifests that `fine_pass.py` wrote for the published runs were
  not kept.
