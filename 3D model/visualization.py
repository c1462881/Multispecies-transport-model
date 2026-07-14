#!/usr/bin/env python3
"""
post.py

Grouped MP4 videos + trajectory plots from core.py NPZ frames.

Speed notes:
- NPZ reads use mmap_mode='r' and a parallel ThreadPool (--workers).
- Per-frame SVG is OFF by default; use --svg to enable.
- MP4 encoding pipes raw RGBA into ffmpeg (no PNG round-trip).
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from copy import copy

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle

# ---- Global font/text style ----
matplotlib.rcParams.update({
    "font.family":       "Liberation Sans",
    "font.sans-serif":   ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.titlesize":  8,
    "mathtext.fontset":  "dejavusans",
    "svg.fonttype":      "none",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# ============================================================
# Fixed visualization settings
# ============================================================
VIDEO_DIR = "videos"
FRAME_DIR = "video_frames"

FPS = 10.0
CRF = 18
DPI = 140

MAX_LIMIT_FRAMES = 40
SPATIAL_LOW_PCT = 1.0
SPATIAL_HIGH_PCT = 99.0
TEMPORAL_LOW_PCT = 25.0
TEMPORAL_HIGH_PCT = 75.0

SOFT_BLUE = "#0E55A1"
SOFT_WHITE = "#FAFAFA"
SOFT_RED = "#9C0F0F"

UNDER_COLOR = "#10F0F0"
OVER_COLOR = "#F60FF6"
BAD_COLOR = "#000000"
# When True: out-of-range values get cyan (under) / magenta (over).
# When False: out-of-range values are clamped to the colormap endpoints
#             (blue for < vmin, red for > vmax).
SHOW_EXTEND_COLORS = True
# ============================================================
# Field definitions
# ============================================================
ION_SPECS = [
    dict(id="Na", source="Na", label="Na$^+$ concentration [mM]", kind="positive"),
    dict(id="K",  source="K",  label="K$^+$ concentration [mM]", kind="positive"),
    dict(id="Ca", source="Ca", label="Ca$^{2+}$ concentration [mM]", kind="positive"),
    dict(id="Cl", source="Cl", label="Cl$^-$ concentration [mM]", kind="positive"),
]

ELECTROSTATIC_SPECS = [
    dict(id="phi_mV",    source="phi_mV",    label="electric potential $\\phi$ [mV]", kind="linear"),
    dict(id="Emag_Vpm",  source="Emag_Vpm",  label="$|E|$ [V/m]", kind="positive_zero"),
    dict(id="rho_Cpm3",  source="rho_Cpm3",  label="charge density $\\rho$ [C/m$^3$]", kind="signed_zero"),
]

OSM_PROTEIN_SPECS = [
    dict(id="osmolarity_mM", source="osmolarity_mM", label="osmolarity [mM]", kind="positive"),
    dict(id="protein_mM",   source="protein_mM",    label="protein concentration [mM]", kind="positive_zero"),
]

VIDEO_GROUPS = [
    dict(name="ions",            title="Ion concentrations",                            specs=ION_SPECS),
    dict(name="electrostatics",  title="Potential, electric field, and charge density", specs=ELECTROSTATIC_SPECS),
    dict(name="osm_and_protein", title="Osmolarity and protein fields",                 specs=OSM_PROTEIN_SPECS),
]

RANGE_FIELDS = [
    ("Na", "Na_mM"),
    ("K", "K_mM"),
    ("Ca", "Ca_mM"),
    ("Cl", "Cl_mM"),
    ("osmolarity_mM", "osmolarity_mM"),
    ("protein_mM", "protein_mM"),
    ("phi_mV", "phi_mV"),
    ("Emag_Vpm", "Emag_Vpm"),
    ("rho_Cpm3", "rho_Cpm3"),
]

def compute_osm_plane_profiles(files, distances_nm, workers=4,
                               x_range_nm=None, y_range_nm=None):
    """Return (t_ns[], profiles) where profiles[d] = dict(neg=arr, pos=arr).
    Plane average = mean over x,y of osmolarity_mM at plane nearest z=+/-d.

    x_range_nm / y_range_nm: optional (lo, hi) tuples in nm. When given, the
    plane average is restricted to cells with x (or y) inside that range.
    """
    def read_one(f):
        with np.load(f, mmap_mode="r", allow_pickle=True) as d:
            if "osmolarity_mM" not in d.files or "z_nm" not in d.files:
                return None
            t = get_time_ns(d)
            z = np.asarray(d["z_nm"], dtype=float)
            osm = np.asarray(d["osmolarity_mM"], dtype=np.float64)  # (nx,ny,nz)

            # Build x,y masks if a range is requested and axes are present.
            ix = slice(None)
            iy = slice(None)
            if x_range_nm is not None and "x_nm" in d.files:
                x = np.asarray(d["x_nm"], dtype=float)
                lo, hi = min(x_range_nm), max(x_range_nm)
                mx = np.where((x >= lo) & (x <= hi))[0]
                if mx.size:
                    ix = mx
            if y_range_nm is not None and "y_nm" in d.files:
                y = np.asarray(d["y_nm"], dtype=float)
                lo, hi = min(y_range_nm), max(y_range_nm)
                my = np.where((y >= lo) & (y <= hi))[0]
                if my.size:
                    iy = my

            sub = osm[ix, :, :][:, iy, :]      # (nx',ny',nz)
            prof = sub.mean(axis=(0, 1))        # (nz,)
            return t, z, prof

    if workers > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(read_one, files))
    else:
        res = [read_one(f) for f in files]
    res = [r for r in res if r is not None]
    if not res:
        return None, None

    t_ns = np.array([r[0] for r in res])
    order = np.argsort(t_ns)
    t_ns = t_ns[order]
    z = res[0][1]
    profs = np.array([res[i][2] for i in order])  # (nt, nz)

    def plane_idx(zt):
        return int(np.argmin(np.abs(z - zt)))

    out = {}
    for d in distances_nm:
        if d == 0.0:
            k = plane_idx(0.0)
            out[d] = dict(neg=profs[:, k], pos=profs[:, k])
        else:
            out[d] = dict(neg=profs[:, plane_idx(-d)],
                          pos=profs[:, plane_idx(+d)])
    return t_ns, out

# ============================================================
# Utility
# ============================================================
def safe_name(s):
    s = s.replace("+", "p").replace("-", "m")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def parse_limit_pair(v):
    if v is None:
        return None
    try:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            lo = float(v[0]); hi = float(v[1])
            if np.isfinite(lo) and np.isfinite(hi):
                if hi < lo: lo, hi = hi, lo
                return (lo, hi)
        if isinstance(v, dict):
            for k0, k1 in [("min","max"), ("vmin","vmax"), ("ymin","ymax")]:
                if k0 in v and k1 in v:
                    lo = float(v[k0]); hi = float(v[k1])
                    if np.isfinite(lo) and np.isfinite(hi):
                        if hi < lo: lo, hi = hi, lo
                        return (lo, hi)
    except Exception:
        return None
    return None


def load_plot_limit_config(out_dir):
    path = os.path.join(out_dir, "plot_limits.json")
    if not os.path.exists(path):
        print()
        print("No editable manual limit file found.")
        print(f"  expected: {path}")
        print("Using auto-detected video color limits and auto trajectory y-limits.")
        print()
        print("To manually control limits, create/edit this file:")
        print(f"  {path}")
        print()
        print("Example plot_limits.json:")
        print(json.dumps(example_plot_limits_json(), indent=2))
        print()
        return {}, path, False
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            print(f"WARNING: {path} is not a JSON object; ignoring.")
            return {}, path, False
        print()
        print("Loaded editable manual limit file:")
        print(f"  {path}")
        print()
        return cfg, path, True
    except Exception as e:
        print()
        print(f"WARNING: failed to read {path}: {e}")
        print("Using auto-detected limits.")
        print()
        return {}, path, False


def example_plot_limits_json():
    return {
        "display_domain": {"x_nm": None, "y_nm": None, "z_nm": None,
                           "xy_z_nm": 0.0, "xz_y_nm": None},
        "overlays": {"pores": True, "membrane": True},
        "osm_monitor": {"local_distance_nm": None},
        "video_clims": {k: None for k in ["Na","K","Ca","Cl","osmolarity_mM",
                                          "protein_mM","phi_mV","Emag_Vpm","rho_Cpm3"]},
        "trajectory_ylims": {k: None for k in ["vm","osm","protein","protein_flux",
                                               "q","current","pores"]},
    }


def limits_from_config_section(cfg, section_name):
    out = {}
    section = cfg.get(section_name, {})
    if not isinstance(section, dict):
        return out
    for key, value in section.items():
        lim = parse_limit_pair(value)
        if lim is not None:
            out[str(key)] = lim
    return out


def display_domain_from_config(cfg):
    section = cfg.get("display_domain", {})
    if not isinstance(section, dict):
        section = {}
    xlim = parse_limit_pair(section.get("x_nm"))
    ylim = parse_limit_pair(section.get("y_nm"))
    zlim = parse_limit_pair(section.get("z_nm"))

    def parse_plane_value(key, default):
        v = section.get(key, default)
        if v is None:
            return default
        try:
            v = float(v)
            if np.isfinite(v): return v
        except Exception:
            pass
        return default

    return dict(
        xlim=xlim, ylim=ylim, zlim=zlim,
        xy_z_nm=parse_plane_value("xy_z_nm", 0.0),
        xz_y_nm=parse_plane_value("xz_y_nm", None),
    )


def overlay_config_from_json(cfg):
    section = cfg.get("overlays", {})
    if not isinstance(section, dict):
        section = {}
    return dict(
        pores=bool(section.get("pores", True)),
        membrane=bool(section.get("membrane", True)),
    )


def print_used_video_limits(all_used):
    if not all_used:
        return
    print()
    print("=" * 70)
    print("USED VIDEO COLOR LIMITS")
    print("Copy values into plot_limits.json -> video_clims if requires manual control.")
    print("=" * 70)
    for field, info in all_used.items():
        lim = info["lim"]; actual = info["actual"]; source = info["source"]
        print(f"  {field:14s}: [{lim[0]:+.10g}, {lim[1]:+.10g}]  "
              f"actual=[{actual[0]:+.10g}, {actual[1]:+.10g}]  source={source}")


def print_used_trajectory_ylims(used_ylims):
    if not used_ylims:
        return
    print()
    print("=" * 70)
    print("USED TRAJECTORY Y-LIMITS")
    print("Copy values into plot_limits.json -> trajectory_ylims if requires manual control.")
    print("=" * 70)
    for key, lim in used_ylims.items():
        if lim is None: continue
        print(f"  {key:14s}: [{lim[0]:+.10g}, {lim[1]:+.10g}]")

SINGLE_COLOR = "#0E55A1"   # blue endpoint of blue–white map
DIVERGE_POS  = "#9C0F0F"   # red   (high end of diverging map)
DIVERGE_NEG  = "#0E55A1"   # blue  (low  end of diverging map)

# Set from CLI in main(): "bw" (blue–white) or "rwb" (red–white–blue)
COLORMAP_MODE = "bw"


def make_cmap(mode=None):
    """Build the field colormap.
    mode='bw'  : white -> blue      
    mode='rwb' : red   -> white -> blue 
    """
    if mode is None:
        mode = COLORMAP_MODE

    if mode == "rwb":
        cmap = LinearSegmentedColormap.from_list(
            "blue_white_red",
            [(0.0, DIVERGE_NEG), (0.5, SOFT_WHITE), (1.0, DIVERGE_POS)],
            N=256,
        )
        low_end, high_end = DIVERGE_NEG, DIVERGE_POS
    else:  # "bw"
        cmap = LinearSegmentedColormap.from_list(
            "white_to_color",
            [(0.0, SOFT_WHITE), (1.0, SINGLE_COLOR)],
            N=256,
        )
        low_end, high_end = SOFT_WHITE, SINGLE_COLOR

    cmap = copy(cmap)
    if SHOW_EXTEND_COLORS:
        cmap.set_under(UNDER_COLOR)
        cmap.set_over(OVER_COLOR)
    else:
        cmap.set_under(low_end)
        cmap.set_over(high_end)
    cmap.set_bad(BAD_COLOR)
    return cmap

def load_params(out_dir):
    path = os.path.join(out_dir, "params.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception:
        return {}
    flat = dict(raw)
    cfg = raw.get("config", {})
    if isinstance(cfg, dict):
        for section_name, section in cfg.items():
            if isinstance(section, dict):
                for k, v in section.items():
                    if k not in flat:
                        flat[k] = v
    return flat


def get_param_float(params, key, default=None):
    try:
        return float(params.get(key, default))
    except Exception:
        return default


def resolve_dirs(folder):
    if os.path.isdir(os.path.join(folder, "npz_frames")):
        out_dir = folder
        npz_dir = os.path.join(folder, "npz_frames")
    else:
        npz_dir = folder
        if os.path.basename(os.path.abspath(npz_dir)) == "npz_frames":
            out_dir = os.path.dirname(os.path.abspath(npz_dir))
        else:
            out_dir = folder
    return os.path.abspath(out_dir), os.path.abspath(npz_dir)


def field_available(npz, spec):
    return spec["source"] in npz.files


def field_array(npz, spec):
    return np.asarray(npz[spec["source"]], dtype=np.float32)


def get_time_ns(npz):
    if "t_s" in npz.files:
        return float(np.asarray(npz["t_s"])) * 1e9
    return np.nan


def get_pores_nm(npz):
    if "pores" not in npz.files:
        return np.zeros((0, 2), dtype=float)
    p = np.asarray(npz["pores"])
    if p.size == 0:
        return np.zeros((0, 2), dtype=float)
    p = p.reshape(-1, 2).astype(float)
    return p * 1e9


def collect_all_pores_nm(files):
    all_pores = []
    for f in files:
        try:
            with np.load(f, mmap_mode="r", allow_pickle=True) as d:
                p = get_pores_nm(d)
                if p.size:
                    all_pores.append(p)
        except Exception:
            pass
    if not all_pores:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(all_pores)


def nearest_index(a, value):
    return int(np.argmin(np.abs(a - value)))


def interval_indices(axis, lim):
    if lim is None:
        return np.arange(axis.size)
    lo, hi = float(lim[0]), float(lim[1])
    if hi < lo: lo, hi = hi, lo
    lo = max(lo, float(axis.min()))
    hi = min(hi, float(axis.max()))
    idx = np.where((axis >= lo) & (axis <= hi))[0]
    if idx.size < 2:
        c = 0.5 * (lo + hi)
        k = nearest_index(axis, c)
        a = max(0, k - 1)
        b = min(axis.size, k + 2)
        idx = np.arange(a, b)
    return idx


def compute_crop(files, crop_mode, margin_nm, display_domain=None):
    if display_domain is None:
        display_domain = {}

    with np.load(files[0], mmap_mode="r", allow_pickle=True) as d0:
        x = np.asarray(d0["x_nm"], dtype=float)
        y = np.asarray(d0["y_nm"], dtype=float)
        z = np.asarray(d0["z_nm"], dtype=float)

    all_pores = collect_all_pores_nm(files)

    xlim = display_domain.get("xlim", None)
    ylim = display_domain.get("ylim", None)
    zlim = display_domain.get("zlim", None)

    if crop_mode == "pores" and all_pores.size:
        if xlim is None:
            xlim = (float(np.min(all_pores[:, 0]) - margin_nm),
                    float(np.max(all_pores[:, 0]) + margin_nm))
        if ylim is None:
            ylim = (float(np.min(all_pores[:, 1]) - margin_nm),
                    float(np.max(all_pores[:, 1]) + margin_nm))

    ix = interval_indices(x, xlim)
    iy = interval_indices(y, ylim)
    iz = interval_indices(z, zlim)

    if display_domain.get("xz_y_nm", None) is not None:
        plane_y = float(display_domain["xz_y_nm"])
    elif all_pores.size:
        plane_y = float(np.median(all_pores[:, 1]))
    else:
        plane_y = 0.0

    plane_z = float(display_domain.get("xy_z_nm", 0.0))

    j0 = nearest_index(y, plane_y)
    k0 = nearest_index(z, plane_z)

    crop = dict(
        x=x, y=y, z=z, ix=ix, iy=iy, iz=iz,
        j0=j0, k0=k0,
        plane_y_nm=float(y[j0]),
        plane_z_nm=float(z[k0]),
        all_pores_nm=all_pores,
        xlim_requested=xlim,
        ylim_requested=ylim,
        zlim_requested=zlim,
    )

    print("Display window:")
    print(f"  x = [{x[ix[0]]:.3g}, {x[ix[-1]]:.3g}] nm, {ix.size} points")
    print(f"  y = [{y[iy[0]]:.3g}, {y[iy[-1]]:.3g}] nm, {iy.size} points")
    print(f"  z = [{z[iz[0]]:.3g}, {z[iz[-1]]:.3g}] nm, {iz.size} points")
    print(f"  XY slice z = {crop['plane_z_nm']:.3g} nm")
    print(f"  XZ slice y = {crop['plane_y_nm']:.3g} nm")

    return crop


# ============================================================
# Robust color limits (with optional parallel sample reads)
# ============================================================
def sampled_files(files, max_frames=MAX_LIMIT_FRAMES):
    if len(files) <= max_frames:
        return files
    idx = np.linspace(0, len(files) - 1, max_frames).round().astype(int)
    idx = np.unique(idx)
    return [files[i] for i in idx]


def slice_values_for_limits(a, crop):
    ix = crop["ix"]; iy = crop["iy"]; iz = crop["iz"]
    j0 = crop["j0"]; k0 = crop["k0"]
    xy = a[np.ix_(ix, iy, [k0])][:, :, 0]
    xz = a[np.ix_(ix, [j0], iz)][:, 0, :]
    v = np.concatenate([xy.ravel(), xz.ravel()])
    return v[np.isfinite(v)]


def robust_limit(files, spec, crop, workers=1):
    samples = sampled_files(files)

    def read_one(f):
        with np.load(f, mmap_mode="r", allow_pickle=True) as d:
            if not field_available(d, spec):
                return None
            return slice_values_for_limits(field_array(d, spec), crop)

    if workers > 1 and len(samples) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            value_lists = list(ex.map(read_one, samples))
    else:
        value_lists = [read_one(f) for f in samples]

    lows, highs, abs_highs = [], [], []
    actual_min = np.inf
    actual_max = -np.inf

    for v in value_lists:
        if v is None or v.size == 0:
            continue
        actual_min = min(actual_min, float(np.nanmin(v)))
        actual_max = max(actual_max, float(np.nanmax(v)))
        if spec["kind"] == "signed_zero":
            abs_highs.append(float(np.nanpercentile(np.abs(v), SPATIAL_HIGH_PCT)))
        else:
            lows.append(float(np.nanpercentile(v, SPATIAL_LOW_PCT)))
            highs.append(float(np.nanpercentile(v, SPATIAL_HIGH_PCT)))

    if not np.isfinite(actual_min) or not np.isfinite(actual_max):
        return (0.0, 1.0), (np.nan, np.nan)

    if spec["kind"] == "signed_zero":
        if not abs_highs:
            m = max(abs(actual_min), abs(actual_max), 1.0)
        else:
            m = float(np.nanpercentile(abs_highs, TEMPORAL_HIGH_PCT))
        if not np.isfinite(m) or m <= 0:
            m = max(abs(actual_min), abs(actual_max), 1.0)
        vmin, vmax = -m, +m
    else:
        if not lows or not highs:
            vmin, vmax = actual_min, actual_max
        else:
            vmin = float(np.nanpercentile(lows,  TEMPORAL_LOW_PCT))
            vmax = float(np.nanpercentile(highs, TEMPORAL_HIGH_PCT))
        if spec["kind"] in ("positive", "positive_zero"):
            if actual_min >= 0:
                if spec["kind"] == "positive_zero":
                    vmin = 0.0
                else:
                    vmin = max(0.0, vmin)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            c = 0.0 if not np.isfinite(vmin) else vmin
            eps = 1.0 if c == 0.0 else abs(c) * 0.05
            vmin, vmax = c - eps, c + eps

    if vmax <= vmin:
        c = 0.5 * (vmin + vmax)
        eps = 1.0 if c == 0.0 else abs(c) * 0.05
        vmin, vmax = c - eps, c + eps

    return (float(vmin), float(vmax)), (float(actual_min), float(actual_max))


# ============================================================
# Pore / membrane overlays
# ============================================================
def overlay_pores(ax_xy, ax_xz, pores_nm, params, crop):
    rp = get_param_float(params, "Rpore_nm", None)
    lm = get_param_float(params, "Lmem_nm", 4.0)
    if rp is None or rp <= 0:
        return

    plane_z = float(crop["plane_z_nm"])
    if lm is not None and np.isfinite(lm) and lm > 0:
        z0 = -0.5 * lm; z1 = +0.5 * lm
        inside_membrane = (z0 <= plane_z <= z1)
    else:
        inside_membrane = False

    if inside_membrane:
        for px, py in pores_nm:
            ax_xy.add_patch(Circle(
                (px, py), rp, fill=False, edgecolor="black",
                linewidth=0.8, alpha=0.85, zorder=10,
            ))

    yplane = float(crop["plane_y_nm"])
    for px, py in pores_nm:
        dy = abs(py - yplane)
        if dy <= rp:
            half_width = np.sqrt(max(rp * rp - dy * dy, 0.0))
            if half_width <= 0:
                continue
            ax_xz.add_patch(Rectangle(
                (px - half_width, -0.5 * lm), 2.0 * half_width, lm,
                fill=False, edgecolor="black", linewidth=0.8,
                alpha=0.85, zorder=10,
            ))


def overlay_membrane(ax_xy, ax_xz, params, crop):
    lm = get_param_float(params, "Lmem_nm", 4.0)
    if lm is None or not np.isfinite(lm) or lm <= 0:
        return

    z0 = -0.5 * lm; z1 = +0.5 * lm

    x = crop["x"]; y = crop["y"]
    ix = crop["ix"]; iy = crop["iy"]
    xmin = float(x[ix[0]]); xmax = float(x[ix[-1]])
    ymin = float(y[iy[0]]); ymax = float(y[iy[-1]])

    ax_xz.add_patch(Rectangle(
        (xmin, z0), xmax - xmin, z1 - z0,
        facecolor="gray", edgecolor="black",
        linewidth=0.8, alpha=0.14, zorder=8,
    ))
    ax_xz.axhline(z0, color="black", lw=0.8, alpha=0.45, zorder=9)
    ax_xz.axhline(z1, color="black", lw=0.8, alpha=0.45, zorder=9)

    plane_z = float(crop["plane_z_nm"])
    if z0 <= plane_z <= z1:
        ax_xy.add_patch(Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            facecolor="gray", edgecolor="black",
            linewidth=0.8, alpha=0.06, zorder=8,
        ))
        ax_xy.text(
            0.02, 0.98, "membrane slice",
            transform=ax_xy.transAxes, ha="left", va="top",
            fontsize=8, color="black", alpha=0.75, zorder=10,
        )


# ============================================================
# Frame figure builder (NO save) and standalone frame writer
# ============================================================
def _plot_group_figure(
    npz_file, group, active_specs, static_limits, crop, params,
    show_pores, show_membrane, run_name,
):
    """Build the matplotlib figure for one frame and return it.
    Caller is responsible for saving and closing."""
    with np.load(npz_file, mmap_mode="r", allow_pickle=True) as d:
        pores_nm = get_pores_nm(d)
        t_ns = get_time_ns(d)
        frame_data = []
        for spec in active_specs:
            a = field_array(d, spec)
            ix = crop["ix"]; iy = crop["iy"]; iz = crop["iz"]
            j0 = crop["j0"]; k0 = crop["k0"]
            xy = a[np.ix_(ix, iy, [k0])][:, :, 0]
            xz = a[np.ix_(ix, [j0], iz)][:, 0, :]
            frame_data.append((spec, xy, xz))

    x = crop["x"]; y = crop["y"]; z = crop["z"]
    ix = crop["ix"]; iy = crop["iy"]; iz = crop["iz"]

    nrows = len(frame_data)
    fig_h = max(4.2, 2.8 * nrows)
    fig_w = 12.5

    fig, ax = plt.subplots(
        nrows, 2, figsize=(fig_w, fig_h), squeeze=False,
        constrained_layout=True, dpi=DPI,
    )

    cmap = make_cmap()

    for r, (spec, xy, xz) in enumerate(frame_data):
        lim_entry = static_limits[spec["id"]]
        if isinstance(lim_entry, tuple) and len(lim_entry) == 2 and isinstance(lim_entry[0], tuple):
            (vmin, vmax), lim_source = lim_entry
        else:
            # backward-compatible fallback if someone stores a bare (vmin, vmax)
            vmin, vmax = lim_entry
            lim_source = "auto"

        if COLORMAP_MODE == "rwb" and lim_source == "auto":
            m = max(abs(vmin), abs(vmax))
            if m > 0:
                vmin, vmax = -m, +m
        # manual limits from plot_limits.json are used exactly as written

        norm = Normalize(vmin=vmin, vmax=vmax, clip=False)

        qm0 = ax[r, 0].pcolormesh(x[ix], y[iy], xy.T, shading="auto", cmap=cmap, norm=norm)
        qm0.set_rasterized(True)
        ax[r, 0].set_aspect("equal", adjustable="box")
        ax[r, 0].set_title(f"{spec['label']} | XY, z={crop['plane_z_nm']:.3g} nm")
        ax[r, 0].set_xlabel("x [nm]")
        ax[r, 0].set_ylabel("y [nm]")
        ax[r, 0].set_xlim(x[ix[0]], x[ix[-1]])
        ax[r, 0].set_ylim(y[iy[0]], y[iy[-1]])

        im1 = ax[r, 1].pcolormesh(x[ix], z[iz], xz.T, shading="auto", cmap=cmap, norm=norm)
        im1.set_rasterized(True)
        ax[r, 1].set_title(f"{spec['label']} | XZ, y={crop['plane_y_nm']:.3g} nm")
        ax[r, 1].set_xlabel("x [nm]")
        ax[r, 1].set_ylabel("z [nm]")
        ax[r, 1].set_xlim(x[ix[0]], x[ix[-1]])
        ax[r, 1].set_ylim(z[iz[0]], z[iz[-1]])

        if show_membrane:
            overlay_membrane(ax[r, 0], ax[r, 1], params, crop)
        if show_pores:
            overlay_pores(ax[r, 0], ax[r, 1], pores_nm, params, crop)

        cbar = fig.colorbar(
            im1, ax=[ax[r, 0], ax[r, 1]],
            shrink=0.92, pad=0.015,
            extend="both" if SHOW_EXTEND_COLORS else "neither",
        )
        cbar.set_label(spec["label"])

    if np.isfinite(t_ns):
        # Fixed-width: "t = SSSS.SS ns" — sign + 4 integer digits + 2 decimals.
        # Adjust INT_W / DEC_W below if runs exceed 10^4 ns.
        INT_W, DEC_W = 4, 2
        total_w = INT_W + 1 + DEC_W           # digits + dot + decimals
        num_str = f"{t_ns:>{total_w}.{DEC_W}f}"
        time_label = f"t = {num_str} ns"
    else:
        time_label = os.path.basename(npz_file)

    fig.suptitle(f"{run_name} | {group['title']}")
    fig.text(
        0.012, 0.012, time_label,
        ha="left", va="bottom", color="black",
        family="monospace",                    # equal-width digits + space
        bbox=dict(facecolor="white", edgecolor="none",
                  alpha=0.75, boxstyle="round,pad=0.25"),
    )
    return fig


def plot_group_frame(
    npz_file, group, active_specs, static_limits, crop, params,
    show_pores, show_membrane, out_png, run_name, write_svg=False,
):
    """Standalone single-frame writer (PNG + optional SVG)."""
    fig = _plot_group_figure(
        npz_file, group, active_specs, static_limits, crop, params,
        show_pores, show_membrane, run_name,
    )
    fig.savefig(out_png, dpi=DPI)
    if write_svg:
        fig.savefig(os.path.splitext(out_png)[0] + ".svg", dpi=600)
    plt.close(fig)


# ============================================================
# Video rendering via ffmpeg rawvideo pipe
# ============================================================
def _open_ffmpeg_pipe(out_mp4, W, H, fps):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{W}x{H}", "-r", f"{fps:.6g}",
        "-i", "-",
        "-vf", "format=yuv420p,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-crf", str(CRF), "-movflags", "+faststart",
        out_mp4,
    ]
    print("ffmpeg pipe:", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10 ** 8)


def render_group_video(
    files, group, active_specs, static_limits, crop, params,
    out_dir, run_name, duration, show_pores, show_membrane,
    keep_frames, write_svg,
):
    video_dir = os.path.join(out_dir, VIDEO_DIR)
    frame_root = os.path.join(out_dir, FRAME_DIR)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frame_root, exist_ok=True)

    group_name = safe_name(group["name"])
    frame_dir = os.path.join(frame_root, group_name)
    os.makedirs(frame_dir, exist_ok=True)

    out_mp4 = os.path.join(video_dir, f"{group_name}.mp4")

    print()
    print("=" * 70)
    print(f"Rendering {group_name}")
    print(f"  output mp4 : {out_mp4}")
    print(f"  frame dir  : {frame_dir}")
    print(f"  fields     : {', '.join([s['id'] for s in active_specs])}")

    if duration is not None and duration > 0:
        fps = max(len(files) / duration, 0.001)
    else:
        fps = FPS

    # Per-frame on-disk artifact only when user explicitly asks.
    save_per_frame = keep_frames or write_svg
    pipe = None

    for i, f in enumerate(files):
        fig = _plot_group_figure(
            f, group, active_specs, static_limits, crop, params,
            show_pores, show_membrane, run_name,
        )
        fig.canvas.draw()
        W, H = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba())  # (H, W, 4) uint8

        if pipe is None:
            pipe = _open_ffmpeg_pipe(out_mp4, W, H, fps)
        try:
            pipe.stdin.write(buf.tobytes())
        except BrokenPipeError:
            plt.close(fig)
            raise RuntimeError(
                "ffmpeg died mid-encode. "
                "Re-run with --keep-frames to inspect produced frames."
            )

        if save_per_frame:
            out_png = os.path.join(frame_dir, f"frame_{i:07d}.png")
            fig.savefig(out_png, dpi=DPI)
            if write_svg:
                fig.savefig(os.path.splitext(out_png)[0] + ".svg")

        plt.close(fig)

        if (i + 1) % max(1, len(files) // 10) == 0 or i == len(files) - 1:
            print(f"  frames: {i + 1}/{len(files)}")

    if pipe is not None:
        pipe.stdin.close()
        rc = pipe.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exit code {rc}")

    # Delete temporary PNGs unless user wants them. SVGs are always preserved
    # if write_svg=True (since they live alongside the PNGs in frame_dir).
    if not keep_frames:
        for png in glob.glob(os.path.join(frame_dir, "frame_*.png")):
            try:
                os.remove(png)
            except OSError:
                pass

    print(f"  wrote: {out_mp4}")
    if write_svg:
        print(f"  svg frames retained in: {frame_dir}")


# ============================================================
# Field-range CSV (parallel)
# ============================================================
def write_field_ranges_csv(files, out_dir, workers=4):
    out_csv = os.path.join(out_dir, "trajectory_field_ranges.csv")

    header = ["frame", "t_ns"]
    for source, name in RANGE_FIELDS:
        header.append(f"{name}_min")
        header.append(f"{name}_max")

    def work(args):
        i, f = args
        row = [i]
        with np.load(f, mmap_mode="r", allow_pickle=True) as d:
            row.append(get_time_ns(d))
            for source, _ in RANGE_FIELDS:
                if source in d.files:
                    a = np.asarray(d[source], dtype=np.float64)
                    finite = a[np.isfinite(a)]
                    if finite.size:
                        row.append(float(np.nanmin(finite)))
                        row.append(float(np.nanmax(finite)))
                    else:
                        row.append(np.nan); row.append(np.nan)
                else:
                    row.append(np.nan); row.append(np.nan)
        return row

    rows = [None] * len(files)
    if workers > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for row in ex.map(work, list(enumerate(files))):
                rows[row[0]] = row
    else:
        for i, f in enumerate(files):
            rows[i] = work((i, f))

    with open(out_csv, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(
                [f"{v:.10g}" if isinstance(v, float) else str(v) for v in row]
            ) + "\n")

    print(f"field range csv: {out_csv}")
    return out_csv


# ============================================================
# Trajectory plotting
# ============================================================
def apply_ylim(ax, lim):
    if lim is None: return
    if len(lim) != 2: return
    lo, hi = float(lim[0]), float(lim[1])
    if hi < lo: lo, hi = hi, lo
    ax.set_ylim(lo, hi)


def normalize_name(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def detect_protein_species(header, ion_names):
    ionset = {normalize_name(n) for n in ion_names}
    names, seen = [], set()
    pat = re.compile(r"(.+?)_(?:leakage_z0_(?:molecules|mol)|flux_z0_mol_s)$")
    for h in header:
        m = pat.match(h.strip())
        if not m:
            continue
        base = m.group(1)
        if normalize_name(base) in ionset:      # skip Na/K/Ca/Cl flux cols
            continue
        if base not in seen:
            seen.add(base)
            names.append(base)
    return names or ["protein"]

def find_col(header, candidates):
    norm_header = [normalize_name(h) for h in header]
    norm_candidates = [normalize_name(c) for c in candidates]
    for c in norm_candidates:
        for i, h in enumerate(norm_header):
            if h == c: return i
    for c in norm_candidates:
        for i, h in enumerate(norm_header):
            if c in h: return i
    return None


def load_csv_numeric(path):
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    return header, arr


def load_field_ranges(path):
    if not os.path.exists(path):
        return None, None
    try:
        return load_csv_numeric(path)
    except Exception:
        return None, None


def col_or_default(header, arr, candidates, default_idx=None):
    idx = find_col(header, candidates)
    if idx is not None and idx < arr.shape[1]:
        return arr[:, idx], idx
    if default_idx is not None and default_idx < arr.shape[1]:
        return arr[:, default_idx], default_idx
    return None, None


def plot_trajectory(out_dir, field_range_csv=None, traj_ylims=None,
                    files=None, params=None, workers=4,
                    osm_x_range_nm=None, osm_y_range_nm=None):
    if traj_ylims is None:
        traj_ylims = {}
    used_ylims = {}

    csv = os.path.join(out_dir, "trajectory.csv")
    if not os.path.exists(csv):
        return used_ylims

    try:
        header, arr = load_csv_numeric(csv)
        if arr.size == 0:
            return used_ylims

        t_ns, _  = col_or_default(header, arr, ["t_ns", "time_ns", "t"], default_idx=3)
        Vm, _    = col_or_default(header, arr, ["Vm_mV", "Vm"], default_idx=4)
        npores,_ = col_or_default(header, arr, ["n_pores", "npore", "pores"], default_idx=5)
        Q, _     = col_or_default(header, arr, ["Q_total_C", "Q_C", "charge_C"], default_idx=None)
        I, _     = col_or_default(header, arr,
                                  ["I_z0_A", "I_A", "current_A"], default_idx=None)

        ion_names = ["Na", "K", "Ca", "Cl"]
        # ---- detect protein species directly from the CSV header ----
        prot_names = detect_protein_species(header, ion_names + ["X", "Y"])
        print(f"[trajectory] protein species detected: {prot_names}")
        fig, ax = plt.subplots(4, 3, figsize=(15, 20)) 

        if Vm is not None:
            ax[0, 0].plot(t_ns, Vm)
        ax[0, 0].set_title("Vm [mV]"); ax[0, 0].set_xlabel("t [ns]")
        apply_ylim(ax[0, 0], traj_ylims.get("vm"))

        OSM_DISTANCES_NM = [3, 5, 10, 20, 40]  # planes at |z|=d
        OSM_LW           = 1.3
        # Cytosol (z<0): small dist -> dark red ... -> yellow (large dist)
        OSM_CYTO_COLORS  = [ "#882615",'#EB4224','#F39180',"#F9C3B9","#FDE8E5",]
        # Bulk (z>0): small dist -> purple ... -> light green (large dist)
        OSM_BULK_COLORS  = [ "#153988","#2563EB","#81A5F3","#B9CDF9", "#E5ECFD",]     

        ax_osm = ax[0, 1]
        ot, profs = (compute_osm_plane_profiles(
                    files, OSM_DISTANCES_NM, workers=workers,
                    x_range_nm=osm_x_range_nm, y_range_nm=osm_y_range_nm)
                if files else (None, None))
        if ot is not None:
            from matplotlib.colors import LinearSegmentedColormap as _LSC
            pos_d = [d for d in OSM_DISTANCES_NM if d > 0]
            dmax  = max(pos_d) if pos_d else 1.0
            cyto_cmap = _LSC.from_list("osm_cyto", OSM_CYTO_COLORS)
            bulk_cmap = _LSC.from_list("osm_bulk", OSM_BULK_COLORS)

            def _frac(d):
                return (d / dmax) if dmax > 0 else 0.0

            for d in OSM_DISTANCES_NM:
                if d == 0:
                    ax_osm.plot(ot, profs[d]["pos"], color="k",
                                lw=OSM_LW + 0.4, label="|z| = 0 nm")
                else:
                    fr = _frac(d)
                    ax_osm.plot(ot, profs[d]["neg"], color=cyto_cmap(fr),
                                lw=OSM_LW, label=f"cyto |z|={d} nm")
                    ax_osm.plot(ot, profs[d]["pos"], color=bulk_cmap(fr),
                                lw=OSM_LW, label=f"bulk |z|={d} nm")
            ax_osm.legend(fontsize=6, ncol=2, framealpha=0.9)

            if osm_x_range_nm is not None or osm_y_range_nm is not None:
                xr = ("all" if osm_x_range_nm is None
                      else f"[{osm_x_range_nm[0]:g},{osm_x_range_nm[1]:g}]")
                yr = ("all" if osm_y_range_nm is None
                      else f"[{osm_y_range_nm[0]:g},{osm_y_range_nm[1]:g}]")
                win_txt = f"\nxy window: x={xr}, y={yr} nm"
            else:
                win_txt = ""
            ax_osm.set_title("osmolarity plane averages [mM]\n"
                             "cyto z<0: red->yellow | bulk z>0: blue->green "
                             "(small->large |z|)" + win_txt)
        else:
            ax_osm.text(0.5, 0.5, "osmolarity planes unavailable",
                        ha="center", va="center")
            ax_osm.set_title("osmolarity planes")
        ax_osm.set_xlabel("t [ns]")

        ax_p = ax[0, 2]; ax_pf = ax_p.twinx()
        ax_pf.set_box_aspect(1)
        colors = ["tab:green", "tab:purple", "tab:orange", "tab:brown",
                  "tab:cyan", "tab:olive"]
        plotted_cum = plotted_flx = False
        for i, nm in enumerate(prot_names):
            cum, _ = col_or_default(header, arr,
                [f"{nm}_leakage_z0_molecules", f"{nm}_leakage_z0_mol"])
            flx, _ = col_or_default(header, arr, [f"{nm}_flux_z0_mol_s"])
            c = colors[i % len(colors)]
            if cum is not None:
                ax_p.plot(t_ns, cum, color=c, label=f"{nm} cum"); plotted_cum = True
            if flx is not None:
                ax_pf.plot(t_ns, flx, color=c, ls="--", alpha=0.5,
                           label=f"{nm} flux"); plotted_flx = True
        ax_p.set_title("protein leakage (z<0 -> z>0)")
        ax_p.set_xlabel("t [ns]"); ax_p.set_ylabel("cumulative")
        ax_pf.set_ylabel("flux [mol/s]")
        if plotted_cum:
            ax_p.legend(fontsize=6, loc="upper left")
        if plotted_flx:
            ax_pf.legend(fontsize=6, loc="lower right")
        if not (plotted_cum or plotted_flx):
            ax_p.text(0.5, 0.5, "protein leakage unavailable",
                      ha="center", va="center", transform=ax_p.transAxes)
        apply_ylim(ax_p, traj_ylims.get("protein"))

        if Q is not None:
            ax[1, 0].plot(t_ns, Q)
        ax[1, 0].set_title("Q total [C]"); ax[1, 0].set_xlabel("t [ns]")
        apply_ylim(ax[1, 0], traj_ylims.get("q"))

        if I is not None:
            ax[1, 1].plot(t_ns, I); ax[1, 1].set_ylabel("I [A]")
            ax[1, 1].set_title("I through z=0\npositive: z<0 -> z>0")
        else:
            ax[1, 1].text(0.5, 0.5, "I not found", ha="center", va="center")
            ax[1, 1].set_title("I through z=0")
        ax[1, 1].set_xlabel("t [ns]")
        apply_ylim(ax[1, 1], traj_ylims.get("current"))

        if npores is not None:
            ax[1, 2].plot(t_ns, npores)
        ax[1, 2].set_title("number of pores"); ax[1, 2].set_xlabel("t [ns]")
        apply_ylim(ax[1, 2], traj_ylims.get("pores"))

        ion_axes = [ax[2, 0], ax[2, 1], ax[2, 2], ax[3, 0]]
        for nm, a in zip(ion_names, ion_axes):
            cum, _ = col_or_default(header, arr, [f"{nm}_efflux_z0_mol"])
            flx, _ = col_or_default(header, arr, [f"{nm}_flux_z0_mol_s"])
            atw = a.twinx()
            atw.set_box_aspect(1)
            if cum is not None:
                a.plot(t_ns, cum, color="tab:blue", label="cumulative")
            if flx is not None:
                atw.plot(t_ns, flx, color="tab:red", alpha=0.55, label="flux")
            a.set_title(f"{nm} flux z=0 (+: z<0->z>0)")
            a.set_xlabel("t [ns]")
            a.set_ylabel("cum [mol]", color="tab:blue")
            atw.set_ylabel("flux [mol/s]", color="tab:red")

        ax[3, 1].axis("off")
        ax[3, 2].axis("off")

        for a in ax.flat:
            a.grid(alpha=0.25)
            a.set_box_aspect(1)   

        used_ylims["vm"]      = tuple(float(v) for v in ax[0, 0].get_ylim())
        used_ylims["protein"] = tuple(float(v) for v in ax[0, 2].get_ylim())
        used_ylims["q"]       = tuple(float(v) for v in ax[1, 0].get_ylim())
        used_ylims["current"] = tuple(float(v) for v in ax[1, 1].get_ylim())
        used_ylims["pores"]   = tuple(float(v) for v in ax[1, 2].get_ylim())

        fig.tight_layout()
        out = os.path.join(out_dir, "trajectory.png")
        fig.savefig(out, dpi=130)
        fig.savefig(os.path.splitext(out)[0] + ".svg")
        plt.close(fig)

        print(f"trajectory plot: {out}")
        return used_ylims

    except Exception as e:
        print("trajectory plot failed:", e)
        return used_ylims
# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Make grouped MP4 visualizations from core.py NPZ frames. "
            "Outputs videos/ions.mp4, videos/electrostatics.mp4, "
            "videos/osm_and_protein.mp4, trajectory.png, and "
            "trajectory_field_ranges.csv."
        )
    )
    ap.add_argument("folder",
                    help="Run folder containing npz_frames/, or the npz_frames folder itself.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Use every Nth saved NPZ frame. Default: 1.")
    ap.add_argument("--duration", type=float, default=None,
                    help="Playback duration of each MP4 in seconds. Default: use fixed fps.")
    ap.add_argument("--crop", choices=["pores", "saved"], default="pores",
                    help="pores: crop around pore region. saved: show full saved NPZ region. Default: pores.")
    ap.add_argument("--margin-nm", type=float, default=180.0,
                    help="Margin around all pores when --crop pores. Default: 180 nm.")
    ap.add_argument("--keep-frames", action="store_true",
                    help="Keep temporary PNG frames. Default: delete after MP4.")
    ap.add_argument("--svg", action="store_true",
                    help="Also write per-frame SVGs (slow; off by default).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel NPZ read workers for stats/range CSV. Default 4.")
    ap.add_argument("--no-extend-colors", action="store_true",
                    help="Disable cyan/magenta out-of-range coloring; "
                         "clamp out-of-range data to blue (min) / red (max).")
    ap.add_argument("--cmap", choices=["bw", "rwb"], default="bw",                                                                  
                    help="Video/frame colormap: 'bw' = white->blue (sequential, "
                        "default); 'rwb' = red->white->blue (diverging, "
                        "symmetric around the midpoint of the color scale).")  
    ap.add_argument("--osm-x-range-nm", type=float, nargs=2, default=None,
                    metavar=("XLO", "XHI"),
                    help="Restrict osmolarity plane average to this x range "
                         "[nm], e.g. --osm-x-range-nm -200 200. Default: full x.")
    ap.add_argument("--osm-y-range-nm", type=float, nargs=2, default=None,
                    metavar=("YLO", "YHI"),
                    help="Restrict osmolarity plane average to this y range "
                         "[nm], e.g. --osm-y-range-nm -200 200. Default: full y.")
    args = ap.parse_args()
    global SHOW_EXTEND_COLORS, COLORMAP_MODE
    SHOW_EXTEND_COLORS = not args.no_extend_colors
    COLORMAP_MODE = args.cmap
    out_dir, npz_dir = resolve_dirs(args.folder)
    files_all = sorted(glob.glob(os.path.join(npz_dir, "state_*.npz")))
    if not files_all:
        print(f"No state_*.npz files found in {npz_dir}")
        return

    stride = max(1, args.stride)
    files = files_all[::stride]
    if not files:
        print("No frames selected.")
        return

    print(f"{len(files_all)} total frames in {npz_dir}")
    print(f"{len(files)} selected frames after stride={stride}")

    os.makedirs(os.path.join(out_dir, VIDEO_DIR), exist_ok=True)

    params = load_params(out_dir)
    print(f"[overlay] Rpore_nm = {params.get('Rpore_nm')}, "
          f"Lmem_nm = {params.get('Lmem_nm')}, "
          f"n pore positions = {len(params.get('pores', []))}")
    run_name = os.path.basename(os.path.abspath(out_dir))

    plot_limit_cfg, plot_limit_file, _ = load_plot_limit_config(out_dir)
    display_domain = display_domain_from_config(plot_limit_cfg)
    overlay_cfg = overlay_config_from_json(plot_limit_cfg)

    crop = compute_crop(files, args.crop, args.margin_nm, display_domain=display_domain)

    file_video_clims = limits_from_config_section(plot_limit_cfg, "video_clims")
    file_traj_ylims = limits_from_config_section(plot_limit_cfg, "trajectory_ylims")
    manual_clims = file_video_clims

    field_range_csv = write_field_ranges_csv(files, out_dir, workers=args.workers)
    traj_ylims = file_traj_ylims
    used_traj_ylims = plot_trajectory(
        out_dir, field_range_csv, traj_ylims=traj_ylims,
        files=files, params=params, workers=args.workers,
        osm_x_range_nm=tuple(args.osm_x_range_nm) if args.osm_x_range_nm else None,
        osm_y_range_nm=tuple(args.osm_y_range_nm) if args.osm_y_range_nm else None,
    )
    print_used_trajectory_ylims(used_traj_ylims)

    all_used_video_limits = {}
    with np.load(files[0], mmap_mode="r", allow_pickle=True) as d0:
        for group in VIDEO_GROUPS:
            active_specs = []
            for spec in group["specs"]:
                if field_available(d0, spec):
                    active_specs.append(spec)
                else:
                    print(f"WARNING: field {spec['source']} not found; skipping {spec['id']}")

            if not active_specs:
                print(f"WARNING: group {group['name']} has no available fields; skipping.")
                continue

            static_limits = {}
            print()
            print(f"Automatic robust color limits for {group['name']}:")
            for spec in active_specs:
                lim, actual = robust_limit(files, spec, crop, workers=args.workers)

                if spec["id"] in manual_clims:
                    lim = manual_clims[spec["id"]]; source_txt = "manual"
                elif spec["source"] in manual_clims:
                    lim = manual_clims[spec["source"]]; source_txt = "manual"
                else:
                    source_txt = "auto"

                static_limits[spec["id"]] = (lim, source_txt)  
                all_used_video_limits[spec["id"]] = dict(
                    lim=lim, actual=actual, source=source_txt,
                )
                print(f"  {spec['id']:14s} "
                      f"scale=[{lim[0]:+.6g}, {lim[1]:+.6g}]  "
                      f"actual=[{actual[0]:+.6g}, {actual[1]:+.6g}]  "
                      f"source={source_txt}")

            render_group_video(
                files=files,
                group=group,
                active_specs=active_specs,
                static_limits=static_limits,
                crop=crop,
                params=params,
                out_dir=out_dir,
                run_name=run_name,
                duration=args.duration,
                show_pores=overlay_cfg["pores"],
                show_membrane=overlay_cfg["membrane"],
                keep_frames=args.keep_frames,
                write_svg=args.svg,
            )

    print_used_video_limits(all_used_video_limits)

    print()
    print("Done.")
    print(f"MP4 files are in: {os.path.join(out_dir, VIDEO_DIR)}")
    print(f"Field ranges CSV: {field_range_csv}")
    print(f"Editable manual limit file path: {plot_limit_file}")


if __name__ == "__main__":
    main()