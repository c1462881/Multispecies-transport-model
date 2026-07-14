#!/usr/bin/env python3
"""
configuration.py

Physical constants, default ion definitions, pore helpers, and YAML config loading.
"""

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


# =====================================================================
# Physical constants
# =====================================================================
F = 96485.33212
R = 8.31446262
NA = 6.02214076e23
eps0 = 8.8541878128e-12
kB = 1.380649e-23

Tem = 310.0
VT = R * Tem / F

eps_w = 80.0 * eps0
eps_l = 2.0 * eps0


# =====================================================================
# Default ions
# Concentrations are mol/m^3 == mM numerically.
# =====================================================================
DEFAULT_IONS = [
    dict(name="Na", z=+1, C_cyto=10.0,   C_bulk=145.0, D=1.33e-9),
    dict(name="K",  z=+1, C_cyto=140.0,  C_bulk=5.0,   D=1.96e-9),
    dict(name="Ca", z=+2, C_cyto=1.0e-4, C_bulk=1.8,   D=0.79e-9),
    dict(name="Cl", z=-1, C_cyto=10.0,   C_bulk=110.0, D=2.03e-9),  
]


# =====================================================================
# Main default YAML schema
# =====================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    "run": {
        "case": "A",
        "gpu": 0,
        "out": "out_gsdmd",
        "dtype": "float32",
        "profile": False,
    },
    "convergence": {
        "enable": False,                       # off by default
        "efflux_rate_tol_mol_per_s": 1.0e-18,  # |dΣ/dt| below this counts as quiet
        "consecutive_checks": 5,               # need N quiet print-intervals in a row
        "min_time_ns": 50.0,                   # never trigger before this sim time
        "min_efflux_mol":   1.0e-22,           # require some efflux to exist (avoid t≈0 triggering)
        "species": "cations",                  # "cations" | "all" | "Na" | "K" | "Ca" | "Cl"
    },
    "domain": {
        "Lx_nm": 1000.0,
        "Ly_nm": 1000.0,
        "Lcyto_nm": 1000.0,
        "Lbulk_nm": 1000.0,
        "Lmem_nm": 4.0,
    },

    "mesh": {
        "dx_min_nm": 0.5,
        "dy_min_nm": 0.5,
        "dz_min_nm": 0.2,
        "ratio_xy": 1.12,
        "ratio_z": 1.10,
        "dx_max_nm": 10.0,
        "dy_max_nm": 10.0,
        "dz_max_nm": 40.0,
    },

    "physics": {
        "Rpore_nm": 10.75,
        "sigma_wall": -0.02,
        "wall_screen_init": True,
        "wall_screen_radius_lambda": 2.0,
        "wall_screen_ion": "auto",

        "v_rest": -0.075,

        "Cprot_mM": 1.0,
        "zprot": -10,
        "Rprot_nm": 5.0,
        "protein_pack_phi_max": 0.64,
        "protein_block_phi": 0.55,
        "pore_block_extra_nm": None,
        "proteins": None, 
        "fixed_charge": None,

        # Diffusivity overrides.
        # null means keep DEFAULT_IONS value.
        "D_override": {
            "Na": None,
            "K": None,
            "Ca": None,
            "Cl": None,
        },
        "z_immobile":        -1,    
        "balance_osm":       True,  
    },

    "pores": {
        "n_pores": 1,
        "pore_spacing_nm": 80.0,
        # explicit pore list in nm, e.g. [[0,0], [21.5,0]]
        "pores_xy_nm": None,
        # if true and case B/C and no explicit pores, spacing is set to 2*Rpore_nm
        "tangential": False,
    },

    "dynamic": {
        "dyn_t_start": 0.0,
        "dyn_interval": 0.0,
        "dyn_zone_x_nm": None,
        "dyn_zone_y_nm": None,
        "dyn_min_dist_nm": None,
        "dyn_seed": 1,

        # Case-D positive feedback.
        # "schedule" or "k_efflux"
        "dyn_trigger": "schedule",
        "dyn_k_threshold_mol": 1.0e-22,
    },
    "time": {
        "dt": 2.5e-12,            # initial dt, used through t_resolve_ns
        "t_final": 1.0e-8,
        "dt_max": None,           # null = auto = 0.9 * dt_diff (explicit CFL)
        "t_resolve_ns": 50.0,     # keep dt = initial dt until this time
        "dt_growth": 1.03,        # per-step geometric growth in ramp phase
    },

    "solver": {
        "cg_rtol": 1.0e-5,
        "cg_maxit": 3000,
        "conv": "auto",
        "direct_kernel_max": 4096,
    },

    "output": {
        # Target total number of frames, including final.
        # 0 disables.
        "save_number": 200,
        "vtk_number": 0,
        "roi_xy_nm": 200.0,
        "roi_z_nm": 200.0,
    },
}


# =====================================================================
# Config object with attribute-style access
# =====================================================================
class AttrDict(dict):
    def __getattr__(self, key):
        try:
            v = self[key]
            if isinstance(v, dict) and not isinstance(v, AttrDict):
                v = AttrDict(v)
                self[key] = v
            return v
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml_config(path: str) -> AttrDict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = deep_update(DEFAULT_CONFIG, user_cfg)
    validate_config(cfg)
    return AttrDict(cfg)


def validate_config(cfg: Dict[str, Any]):
    case = cfg["run"]["case"]
    if case not in ["A", "B", "C", "D"]:
        raise ValueError("run.case must be A, B, C, or D")

    dtype = cfg["run"]["dtype"]
    if dtype not in ["float32", "float64"]:
        raise ValueError("run.dtype must be float32 or float64")

    conv = cfg["solver"]["conv"]
    if conv not in ["auto", "direct", "fft"]:
        raise ValueError("solver.conv must be auto, direct, or fft")

    trigger = cfg["dynamic"]["dyn_trigger"]
    if trigger not in ["schedule", "k_efflux"]:
        raise ValueError("dynamic.dyn_trigger must be schedule or k_efflux")

    if cfg["time"]["dt"] <= 0:
        raise ValueError("time.dt must be positive")
    if cfg["time"]["t_final"] <= 0:
        raise ValueError("time.t_final must be positive")

    if cfg["output"]["save_number"] < 0:
        raise ValueError("output.save_number must be >= 0")
    if cfg["output"]["vtk_number"] < 0:
        raise ValueError("output.vtk_number must be >= 0")


# =====================================================================
# Ion utilities
# =====================================================================
def setup_immobile_background(
    ions,
    z_protein,
    Cprot_cyto_mM,
    Cprot_bulk_mM=0.0,
    z_immobile=-1,
    balance_osm=True,
    fixed_signed_cyto=0.0,
    fixed_signed_bulk=0.0,
    fixed_osm_cyto=0.0,
    fixed_osm_bulk=0.0,
):
    ions = [dict(s) for s in ions]

    def signed_sum(side, include_prot):
        s = sum(sp["z"] * sp["C_" + side] for sp in ions)
        if include_prot:
            s += z_protein * (Cprot_cyto_mM if side == "cyto" else Cprot_bulk_mM)
        s += (fixed_signed_cyto if side == "cyto" else fixed_signed_bulk)
        return s

    # Electroneutrality: signed_sum + z_X * [X] = 0  ->  [X] = -signed_sum / z_X
    s_c = signed_sum("cyto", True)
    s_b = signed_sum("bulk", True)

    if z_immobile == 0:
        raise ValueError("z_immobile must be nonzero.")

    X_cyto = -s_c / z_immobile
    X_bulk = -s_b / z_immobile

    if X_cyto < 0 or X_bulk < 0:
        raise ValueError(
            f"Immobile background concentration negative "
            f"(X_cyto={X_cyto}, X_bulk={X_bulk}). "
            f"Choose z_immobile of opposite sign or adjust ion concentrations."
        )

    # Osmolarity = sum of all particle concentrations.
    def osm_total(side, Xside, Yside):
        return sum(sp["C_" + side] for sp in ions) + \
               (Cprot_cyto_mM if side == "cyto" else Cprot_bulk_mM) + Xside + Yside + \
               (fixed_osm_cyto if side == "cyto" else fixed_osm_bulk)

    Y_cyto = 0.0
    Y_bulk = 0.0

    if balance_osm:
        osm_c = osm_total("cyto", X_cyto, 0.0)
        osm_b = osm_total("bulk", X_bulk, 0.0)
        if osm_c < osm_b:
            Y_cyto = osm_b - osm_c
        elif osm_b < osm_c:
            Y_bulk = osm_c - osm_b

    return ions, X_cyto, X_bulk, Y_cyto, Y_bulk, z_immobile


def apply_diffusivity_overrides(ions, D_override, log=None):
    ions = [dict(s) for s in ions]
    D_override = D_override or {}

    for s in ions:
        if s["name"] in D_override and D_override[s["name"]] is not None:
            old = s["D"]
            s["D_default"] = old
            s["D"] = float(D_override[s["name"]])
            if log is not None:
                log(
                    f"  diffusivity OVERRIDE: {s['name']} "
                    f"D={s['D']:.3e} m^2/s, default={old:.3e}"
                )

    return ions

def get_protein_species(cfg_physics):
    """Return list of protein species dicts: name, Cprot_mM, zprot, Rprot_nm.
    Backward compatible: if 'proteins' is None/empty, build one species from
    the legacy scalars Cprot_mM / zprot / Rprot_nm."""
    plist = cfg_physics.get("proteins", None)
    if not plist:
        return [dict(
            name="protein",
            Cprot_mM=float(cfg_physics.get("Cprot_mM", 1.0)),
            zprot=float(cfg_physics.get("zprot", -10)),
            Rprot_nm=float(cfg_physics.get("Rprot_nm", 5.0)),
        )]
    out = []
    for i, p in enumerate(plist):
        out.append(dict(
            name=str(p.get("name", f"protein{i+1}")),
            Cprot_mM=float(p.get("Cprot_mM", 0.0)),
            zprot=float(p.get("zprot", 0.0)),
            Rprot_nm=float(p.get("Rprot_nm", 5.0)),
        ))
    return out
    
def debye_length(ions, side="cyto", Cprot_mM=0.0, z_protein=-5):
    I = 0.0
    key = "C_cyto" if side == "cyto" else "C_bulk"
    for s in ions:
        I += s[key] * (s["z"] ** 2)
    I += Cprot_mM * (z_protein ** 2)
    return np.sqrt(eps_w * R * Tem / (F**2 * I))


# =====================================================================
# Non-uniform grid
# =====================================================================
def stretched_axis(L_neg, L_pos, h_min, ratio=1.12, h_max=None):
    if h_max is None:
        h_max = max(L_neg, L_pos) / 8.0

    def one_side(L):
        xs = [0.0]
        h = h_min
        while xs[-1] < L:
            xs.append(xs[-1] + h)
            h = min(h * ratio, h_max)
        xs[-1] = L
        return np.array(xs, dtype=np.float64)

    xp = one_side(L_pos)
    xn = one_side(L_neg)
    return np.concatenate([-xn[:0:-1], xp]).astype(np.float64)


def cell_metrics(x):
    """
    Return h_face and h_dual.
    h_face[i] = x[i+1] - x[i]
    h_dual[i] = local FV dual-cell width
    """
    h_face = np.diff(x)
    h_dual = np.empty_like(x)
    h_dual[1:-1] = 0.5 * (h_face[:-1] + h_face[1:])
    h_dual[0] = 0.5 * h_face[0]
    h_dual[-1] = 0.5 * h_face[-1]
    return h_face, h_dual


# =====================================================================
# Pore placement helpers
# =====================================================================
def pores_single():
    return [(0.0, 0.0)]


def pores_grid(n, spacing_nm):
    spacing = spacing_nm * 1e-9
    m = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(m) - (m - 1) / 2) * spacing
    ys = (np.arange(m) - (m - 1) / 2) * spacing
    out = []
    for x in xs:
        for y in ys:
            out.append((float(x), float(y)))
            if len(out) == n:
                return out
    return out


def pores_half_cluster(n, spacing_nm):
    spacing = spacing_nm * 1e-9
    m = int(np.ceil(np.sqrt(n)))
    xs = np.arange(m) * spacing + spacing
    ys = (np.arange(m) - (m - 1) / 2) * spacing
    out = []
    for x in xs:
        for y in ys:
            out.append((float(x), float(y)))
            if len(out) == n:
                return out
    return out


def parse_explicit_pores_nm(pores_xy_nm):
    if pores_xy_nm is None:
        return None

    if isinstance(pores_xy_nm, str):
        obj = json.loads(pores_xy_nm)
    else:
        obj = pores_xy_nm

    return [(float(p[0]) * 1e-9, float(p[1]) * 1e-9) for p in obj]


def dynamic_random_schedule(
    n_new,
    t_start,
    interval,
    zone_x_nm,
    zone_y_nm,
    min_dist_nm=30.0,
    seed=1,
    existing=None,
):
    rng = np.random.default_rng(seed)
    # Pre-existing pores (e.g. case-D seed pore) are honored for the
    # min-distance constraint but do not generate events.
    pores = list(existing) if existing else []
    events = []

    min_dist = min_dist_nm * 1e-9
    xmin, xmax = zone_x_nm[0] * 1e-9, zone_x_nm[1] * 1e-9
    ymin, ymax = zone_y_nm[0] * 1e-9, zone_y_nm[1] * 1e-9

    for k in range(n_new):
        for _ in range(10000):
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            if all(np.hypot(x - a, y - b) > min_dist for a, b in pores):
                pores.append((x, y))
                events.append(
                    dict(
                        t=t_start + k * interval,
                        x=float(x),
                        y=float(y),
                    )
                )
                break

    return events