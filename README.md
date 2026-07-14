# Multispecies transport: 1D reductionist and 3D full PNP models

Python code accompanying the manuscript
**"Multispecies transport shapes electrochemical environments in perforated cells."**

This repository contains two independent Poisson–Nernst–Planck (PNP)
simulators used in the paper:

| Directory  | Model | Backend | Purpose |
|-----------|-------|---------|---------|
| `1D_model/` | 1D reductionist PNP | CPU (NumPy/SciPy) | planar-limit, single coordinate normal to the membrane |
| `3D_model/` | 3D full PNP with explicit pores | GPU (CuPy + custom CUDA kernels) | cytosol / lipid bilayer with cylindrical pores / extracellular bulk |

Both models solve the same governing equations — drift–diffusion for
four ions (Na⁺, K⁺, Ca²⁺, Cl⁻) and one charged protein species, coupled
to a variable-permittivity Poisson equation — but at very different
spatial resolutions and computational cost.

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

All physical, geometrical, and numerical settings live in
`1D_model/configuration.py`. The typical workflow is:

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

Model-free timescale metrics (*t*₁₀ / *t*₅₀ / *t*₉₀ and the integral
relaxation time *τ*_int) on an existing run can be regenerated with:

```bash
python timescale_free.py <path_to_run_folder>
```

---

## Running the 3D full model

The 3D model is entirely YAML-driven. A minimal configuration file
looks like:

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

- **Case A** — single central pore.
- **Case B** — centered *N × N* grid of pores. Setting
  `pores.tangential: true` places them mouth-to-mouth
  (spacing = 2·*R*_pore).
- **Case C** — same grid but shifted entirely to *x > 0*.
- **Case D** — dynamic pore insertion, triggered either by a fixed
  time schedule (`dyn_trigger: schedule`) or by a positive-feedback
  rule on cumulative K⁺ efflux (`dyn_trigger: k_efflux`).

Whenever a new pore opens (Case D), all masks and the Poisson operator
are rebuilt, concentrations in newly opened voxels are re-seeded from
the smooth initial profile, and local Cl⁻ electroneutrality is
re-enforced.

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

Rendering behaviour (crop, colormap, out-of-range coloring, osmolarity
averaging window, per-frame SVG output, parallel NPZ reads, etc.) can be
tuned via CLI flags such as `--cmap rwb`, `--crop saved`,
`--osm-x-range-nm ...`, `--svg`, `--workers 8`. Finer control of color
limits, y-limits for trajectory panels, display domain, and overlay
toggles is available by placing a `plot_limits.json` inside the run
folder; a template is printed to stdout the first time
`visualization.py` runs without one. Run

```bash
python visualization.py --help
```

for the full list of options.
