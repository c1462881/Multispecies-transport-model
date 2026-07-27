# Multispecies transport PNP models

Python code accompanying the manuscript
**"Multispecies transport shapes electrochemical environments in perforated cells."**

This repository contains the Python implementations used for the reductionist
one-dimensional model and the full three-dimensional pore-resolved model
described in the accompanying paper.

## Repository structure

```text
.
├── 1D_model/
│   ├── configuration.py
│   ├── initialization.py
│   ├── solver.py
│   ├── main.py
│   ├── visualization.py
│   ├── timescale_fit.py
│   └── timescale_free.py
├── 3D_model/
│   ├── configuration.py
│   ├── initialization.py
│   ├── solver.py
│   ├── time_loop.py
│   ├── output.py
│   ├── main.py
│   └── visualization.py
└── README.md
```

## Reductionist one-dimensional model

The reductionist model solves a one-dimensional Poisson–Nernst–Planck system
normal to a planar membrane. It includes four physiological ions, a charged
protein, and auxiliary charged and neutral species used to initialize
electroneutral and osmotically balanced reservoirs. The membrane potential is
obtained from Poisson's equation at every transport right-hand-side evaluation.

### Scripts

- `configuration.py`: physical constants, species properties, geometry,
  simulation duration, output settings, and plotting styles.
- `initialization.py`: grid and membrane masks, diffusivity and mobility fields,
  auxiliary-species concentrations, resting-potential initialization, and
  Donnan reference calculation.
- `solver.py`: Poisson solve, Scharfetter–Gummel ion fluxes, protein flux,
  positivity limiting, and time integration.
- `main.py`: simulation entry point, diagnostics, checkpointing, plotting, and
  CSV export.
- `visualization.py`: spatial and temporal figures and monitor-point extraction.
- `timescale_fit.py`: exponential-relaxation fitting and theoretical timescale
  diagnostics used by `main.py`.
- `timescale_free.py`: post-processing of exported temporal data using
  model-free crossing and finite-window integral metrics.

### Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- pandas
- Matplotlib

Install the CPU-model dependencies with:

```bash
python -m pip install numpy scipy pandas matplotlib
```

### Run

Run commands from the model directory because the scripts use local imports:

```bash
cd reductionist_model
python main.py
```

Parameters are set directly in `configuration.py`. Each run creates a
time-stamped output directory containing logs, a checkpoint, figures, parameter
tables, and spatial and temporal CSV files. To analyze an existing output
directory with the model-free timescale script:

```bash
python timescale_free.py PATH_TO_OUTPUT_DIRECTORY
```

The supplied implementation calls SciPy's explicit `RK45` integrator directly.
The configured names `SOLVER_METHOD`, `JAC_BW`, `N_DEBYE_LAYERS`, and
`V_REST_MAX_ITER` do not change that implementation as supplied.

## Full three-dimensional pore-resolved model

The full model solves multispecies electrodiffusion on a nonuniform
three-dimensional Cartesian grid with an explicit membrane and cylindrical
pores. It supports protein crowding, steric transport gates, static or
sequentially inserted pores, a variable-permittivity Poisson solve, and
GPU-accelerated Scharfetter–Gummel transport.

### Scripts

- `configuration.py`: default configuration, YAML overrides, validation,
  physical constants, ion and protein definitions, mesh generation, and pore
  placement utilities.
- `initialization.py`: GPU selection, grid and mask construction, field
  initialization, resting-potential preparation, Poisson setup, and state
  assembly.
- `solver.py`: convolution utilities, crowding fields, fused
  Scharfetter–Gummel kernels, electric-field calculation, and the
  preconditioned-conjugate-gradient Poisson solver.
- `time_loop.py`: explicit time stepping, adaptive step schedule, sequential
  pore insertion, convergence checks, diagnostics, and state output.
- `output.py`: run directories, logging and timing, parameter export, trajectory
  CSV output, and binary VTK rectilinear-grid output.
- `main.py`: YAML-driven command-line entry point.
- `visualization.py`: trajectory plots, field ranges, cross-sectional frames,
  and optional MP4 video generation from saved states.

### Requirements

- Python 3.10 or newer
- An NVIDIA GPU with a compatible CUDA driver
- CuPy built for the installed CUDA version
- NumPy
- SciPy
- PyYAML
- Matplotlib
- FFmpeg, only when generating MP4 videos

Install the Python dependencies after selecting the appropriate CuPy package
for the local CUDA installation. For example, for a CUDA 12 installation:

```bash
python -m pip install cupy-cuda12x numpy scipy pyyaml matplotlib
```

### Configure and run

The full model requires a YAML file. Unspecified values are filled from
`DEFAULT_CONFIG` in `configuration.py`; therefore, a file containing only
`{}` runs the default case.

```bash
cd full_model
printf '{}\n' > config.yaml
python main.py --config config.yaml
```

A minimal override may instead be written as:

```yaml
run:
  case: A
  gpu: 0
  out: out_gsdmd

time:
  t_final: 1.0e-8

output:
  save_number: 200
  vtk_number: 0
```

Cases `A`, `B`, and `C` define static pore arrangements. Case `D` enables
sequential pore insertion according to a schedule or a cumulative potassium
efflux threshold. See `DEFAULT_CONFIG` and its inline comments for all
available parameters.

To generate plots or videos after a run:

```bash
python visualization.py PATH_TO_OUTPUT_DIRECTORY
```

Use `python visualization.py --help` for cropping, limit, parallelism, and
video options. Video generation additionally requires `ffmpeg` on the system
path.

## Outputs

The reductionist model writes figures and CSV summaries directly to its run
directory. The full model writes logs and parameters together with compressed
NPZ state files, optional VTK rectilinear-grid files, and a trajectory CSV.
Output volume can be controlled with the corresponding configuration settings.

## Citation

If you use this code, please cite the accompanying paper. Replace this section
with the final article citation and DOI when available.

## License

No license was included with the supplied scripts. Add an explicit license
before public release so reuse conditions are clear.

