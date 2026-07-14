#!/usr/bin/env python3
"""
output.py

Output directories, logger, VTK writer, NPZ writer helpers.
"""

import base64
import json
import os
import struct
from collections import defaultdict
from datetime import datetime

import cupy as cp
import numpy as np


# =====================================================================
# Timer
# =====================================================================
class Timer:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.acc = defaultdict(float)
        self.cnt = defaultdict(int)
        self._ev = {}

    def tic(self, k):
        if not self.enabled:
            return
        ev = cp.cuda.Event()
        ev.record()
        self._ev[k] = ev

    def toc(self, k):
        if not self.enabled:
            return
        s = self._ev.pop(k, None)
        if s is None:
            return
        e = cp.cuda.Event()
        e.record()
        e.synchronize()
        self.acc[k] += cp.cuda.get_elapsed_time(s, e) * 1e-3
        self.cnt[k] += 1

    def report(self, L):
        L("")
        L("=== Wall-time report ===")
        if not self.enabled:
            L("  profiling disabled.")
            return

        tot = sum(self.acc.values()) + 1e-30
        for k, v in sorted(self.acc.items(), key=lambda kv: -kv[1]):
            L(
                f"  {k:24s}  {v:8.2f} s "
                f"({100*v/tot:5.1f}%)  calls={self.cnt[k]:6d}  "
                f"avg={v/max(1,self.cnt[k])*1e3:8.2f} ms"
            )
        L(f"  {'TOTAL':24s}  {tot:8.2f} s")


# =====================================================================
# Logger
# =====================================================================
class RunLogger:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.lines = []
        self.sanity_path = os.path.join(out_dir, "sanity.txt")

    def __call__(self, s=""):
        print(s, flush=True)
        self.lines.append(str(s))

    def save(self):
        with open(self.sanity_path, "w") as f:
            f.write("\n".join(self.lines))


# =====================================================================
# Output directory setup
# =====================================================================
def setup_output_dirs(cfg):
    case = cfg.run.case
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{case}_{stamp}"

    out = cfg.run.out
    if out == "out_gsdmd":
        out_dir = f"untitled_{run_tag}"
    else:
        out_dir = out

    vtk_dir = os.path.join(out_dir, "vtk")
    npz_dir = os.path.join(out_dir, "npz_frames")

    os.makedirs(vtk_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)

    return dict(
        run_tag=run_tag,
        out_dir=out_dir,
        vtk_dir=vtk_dir,
        npz_dir=npz_dir,
        params_path=os.path.join(out_dir, "params.json"),
        csv_path=os.path.join(out_dir, "trajectory.csv"),
    )


# =====================================================================
# VTK rectilinear binary writer
# =====================================================================
def _b64(arr):
    arr = np.ascontiguousarray(arr.astype(np.float32).ravel(order="F"))
    raw = struct.pack("<I", arr.nbytes) + arr.tobytes(order="C")
    return base64.b64encode(raw).decode("ascii")


def write_vtr_binary(fname, fields, x_nodes, y_nodes, z_nodes):
    Nx, Ny, Nz = x_nodes.size, y_nodes.size, z_nodes.size
    with open(fname, "w") as f:
        f.write(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="RectilinearGrid" version="0.1" '
            'byte_order="LittleEndian" header_type="UInt32">\n'
        )
        f.write(f'<RectilinearGrid WholeExtent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}">\n')
        f.write(f'<Piece Extent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}">\n')
        f.write("<Coordinates>\n")
        for nm, arr in [("x", x_nodes), ("y", y_nodes), ("z", z_nodes)]:
            f.write(
                f'<DataArray type="Float32" Name="{nm}" format="binary">'
                f"{_b64(arr)}</DataArray>\n"
            )
        f.write("</Coordinates>\n<PointData>\n")
        for name, arr in fields.items():
            f.write(
                f'<DataArray type="Float32" Name="{name}" format="binary">'
                f"{_b64(arr)}</DataArray>\n"
            )
        f.write("</PointData>\n</Piece>\n</RectilinearGrid>\n</VTKFile>\n")


# =====================================================================
# Plot limit config
# =====================================================================
def load_plot_limits(out_dir):
    path = os.path.join(out_dir, "plot_limits.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {}


# =====================================================================
# Save config / params
# =====================================================================
def save_params_json(state, cfg, paths, extra=None):
    extra = extra or {}

    # convert AttrDict to regular JSON-ish dict
    def to_plain(x):
        if isinstance(x, dict):
            return {k: to_plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_plain(v) for v in x]
        return x

    pj = {
        "config": to_plain(cfg),
        "run_tag": paths["run_tag"],
        "out_dir": paths["out_dir"],
        "ions": state.ions,
        "proteins": state.proteins,
        "pores": state.pores,
        "dynamic_events": state.dynamic_events,
        "x_nm": (state.x_np * 1e9).tolist(),
        "y_nm": (state.y_np * 1e9).tolist(),
        "z_nm": (state.z_np * 1e9).tolist(),
        "lambda_D_used_m": state.lD,
        "current_plane_z_nm": state.z0_face_nm,
        "protein_exclusion_z_nm": {
            p["name"]: [
                (state.z_inner_face - p["Rprot_m"]) * 1e9,
                (state.z_outer_face + p["Rprot_m"]) * 1e9,
            ]
            for p in state.proteins
        },
        "protein_xy_shrunk_pore_radius_nm": {
            p["name"]: max(state.Rp - p["Rprot_m"], 0.0) * 1e9
            for p in state.proteins
        },
        "osm_local_distance_nm": state.osm_local_dist_m * 1e9,
        "solver": "PCG_lineJacobiZ_nonuniform",
        "np_scheme": "Scharfetter-Gummel nonuniform_FV",
    }

    pj.update(extra)

    with open(paths["params_path"], "w") as f:
        json.dump(pj, f, indent=2, default=str)


# =====================================================================
# CSV writer
# =====================================================================
def write_trajectory_csv(csv_path, header, log_rows):
    if log_rows:
        np.savetxt(csv_path, np.array(log_rows), delimiter=",", header=header, comments="")
    else:
        with open(csv_path, "w") as f:
            f.write(header + "\n")
