# Multispecies transport: 1D reductionist and 3D full PNP models

Python code accompanying the manuscript
**"Multispecies transport shapes electrochemical environments in perforated cells."**

This repository contains two independent Poisson-Nernst-Planck (PNP)
simulators that implement the framework developed in the paper.

| Directory  | Model | Backend | Purpose |
|-----------|-------|---------|---------|
| `1D_model/` | 1D reductionist PNP | CPU (NumPy/SciPy) | planar-limit, single coordinate normal to the membrane |
| `3D_model/` | 3D full PNP with explicit pores | GPU (CuPy + custom CUDA kernels) | cytosol / lipid bilayer with cylindrical pores / extracellular bulk |

Both models solve the same governing equations, coupling drift-diffusion transport of four ions (Na⁺, K⁺, Ca²⁺, Cl⁻) and one charged protein species to a variable-permittivity Poisson equation, yet they resolve this shared physics at markedly different spatial scales and computational cost.

---

## Repository layout

```
.
├── 1D_model/
│   ├── configuration.py       # all parameters (edit here)
│   ├── initialization.py      # grid, initial fields, resting-potential imposition, Donnan reference
│   ├── solver.py              # Scharfetter–Gummel fluxes + RK45 time integration
│   ├── main.py                # entry point (runs a single simulation)
│   ├── visualization.py       # spatial/temporal figures + timescale panel
│   ├── timescale_fit.py       # first-order relaxation fits (shared with main)
│   └── timescale_free.py      # model-free fractional-crossing times on a saved run
│
├── 3D_model/
│   ├── configuration.py       # YAML schema, defaults, ion/protein defaults, validation
│   ├── initialization.py      # mesh, pores, masks, Poisson solver assembly, initial state
│   ├── solver.py              # CUDA kernels: Scharfetter–Gummel, batched Thomas, PCG Poisson
│   ├── time_loop.py           # adaptive time loop, dynamic pores, diagnostics, NPZ/VTR I/O
│   ├── output.py              # logger, timers, VTR writer, params/CSV writers
│   ├── main.py                # entry point (YAML-driven)
│   └── visualization.py       # NPZ → MP4 videos + trajectory plots (post-processing)
│
└── README.md
```

---

## Requirements

### 1D model (CPU)

- Python ≥ 3.9
- `numpy`, `scipy`, `matplotlib`, `pandas`

```bash
pip install numpy scipy matplotlib pandas
```

### 3D model (GPU)

- Python ≥ 3.9
- NVIDIA GPU with a CUDA-compatible driver
  (tested on an NVIDIA RTX PRO 4500 Blackwell, CUDA 12.x)
- `cupy` matching your CUDA version
- `numpy`, `pyyaml`, `matplotlib`, `pandas`
- `ffmpeg` available on `PATH` (only needed by `visualization.py` for MP4 encoding)

```bash
pip install numpy pyyaml matplotlib pandas
pip install cupy-cuda12x            # replace with cupy-cuda11x, etc., to match local CUDA
```

---

## Running the 1D reductionist model

All physical, geometrical, and numerical settings reside in
`1D_model/configuration.py`, and a typical run therefore follows this workflow:

```bash
cd 1D_model
# (optional) edit configuration.py: SIM_T, ion concentrations, V_REST_MV, ...
python main.py
```

Each run creates a timestamped output folder (e.g. `1D_1423/`) containing:

- `Spatial_final.svg`, `Temporal_all.svg`, `Donnan_convergence.svg`,
  `Timescales_difference.svg`
- `Spatial_datapoints.csv`, `Spatial_snapshots.csv`,
  `Temporal_datapoints.csv`
- `Parameters_and_Data.csv` — full parameter provenance
- `run_log.txt`
- `sol_checkpoint.pkl` — solution checkpoint (set
  `RELOAD_FROM_CHECKPOINT = True` in `main.py` to replot from it without
  re-running the integrator)

To regenerate model-free timescale metrics (*t*₁₀ / *t*₅₀ / *t*₉₀ and the integral
relaxation time *τ*_int) on an existing run, call:

```bash
python timescale_free.py <path_to_run_folder>
```

---

## Running the 3D full model

The 3D model runs entirely from a single YAML configuration file. A minimal
example looks like:

```yaml
run:
  case: A               # A: single pore, B: grid, C: shifted cluster, D: dynamic insertion
  gpu: 0
  out: my_run
  dtype: float32

domain:  {Lx_nm: 1000, Ly_nm: 1000, Lcyto_nm: 1000, Lbulk_nm: 1000, Lmem_nm: 4}
physics: {Rpore_nm: 10.75, sigma_wall: -0.02, v_rest: -0.075,
          Cprot_mM: 1.0, zprot: -10, Rprot_nm: 5.0}
pores:   {n_pores: 1}
time:    {dt: 2.5e-12, t_final: 1.0e-8, t_resolve_ns: 50.0, dt_growth: 1.03}
output:  {save_number: 200}
```

Then:

```bash
cd 3D_model
python main.py --config my_config.yaml
```

Each run creates a directory (default: `<case>_<timestamp>/`) containing:

- `params.json` — full provenance (config, mesh, pore list, dynamic
  events, ions, protein species, etc.)
- `trajectory.csv` — per-print row of *V*_m, currents, per-species
  cumulative efflux, protein leakage, and total charge
- `npz_frames/state_*******.npz` — 3D field snapshots on the ROI
- `vtk/fields_*******.vtr` — optional full-domain VTR fields
- `sanity.txt` — everything printed to stdout during the run

### Four pore-opening scenarios

- **Case A** considers a single central pore.
- **Case B** places a centered *N × N* grid of pores; setting
  `pores.tangential: true` positions them mouth to mouth, with spacing
  equal to 2·*R*_pore.
- **Case C** uses the same grid but shifts it entirely to *x > 0*.
- **Case D** inserts pores dynamically, triggered either by a fixed
  time schedule (`dyn_trigger: schedule`) or by a positive-feedback
  rule that responds to cumulative K⁺ efflux (`dyn_trigger: k_efflux`).

Whenever a new pore opens under Case D, the simulation rebuilds all masks and
the Poisson operator, reseeds concentrations in the newly opened voxels from
the smooth initial profile, and re-enforces local Cl⁻ electroneutrality
there.

### Post-processing (videos and trajectory plots)

`visualization.py` turns the NPZ frame stream into publication-ready
plots and MP4s:

```bash
python visualization.py <run_folder>
```

It produces:

- `videos/ions.mp4`, `videos/electrostatics.mp4`,
  `videos/osm_and_protein.mp4`
- `trajectory.png` and `trajectory.svg`
- `trajectory_field_ranges.csv`

CLI flags such as `--cmap rwb`, `--crop saved`, `--osm-x-range-nm ...`,
`--svg`, and `--workers 8` tune rendering behaviour, including crop,
colormap, out-of-range coloring, osmolarity averaging window, per-frame
SVG output, and parallel NPZ reads. Placing a `plot_limits.json` file
inside the run folder grants finer control over color limits, y-limits
for trajectory panels, display domain, and overlay toggles; running
`visualization.py` without one prints a template to stdout on first use.

Run

```bash
python visualization.py --help
```

for the full list of options.
