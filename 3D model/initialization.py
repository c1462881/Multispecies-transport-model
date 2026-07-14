#!/usr/bin/env python3
"""
initialization.py

Build mesh, pores, masks, concentrations, wall charges, Poisson solver,
diagnostic closures, and return a SimulationState.
"""

import os
from types import SimpleNamespace

import cupy as cp
import numpy as np

from configuration import (
    F, R, NA, eps0, kB, Tem, VT, eps_w, eps_l,
    DEFAULT_IONS,
    setup_immobile_background, 
    apply_diffusivity_overrides,
    debye_length,
    stretched_axis,
    cell_metrics,
    get_protein_species,
    pores_grid,
    pores_half_cluster,
    parse_explicit_pores_nm,
    dynamic_random_schedule,
)
from output import load_plot_limits
from solver import (
    harm,
    make_ball_kernel,
    make_shell_kernel,
    Convolver3D,
    PhysicsCache,
    PoissonPCG_LJZ,
    bern_cp,
)


def build_initial_state(cfg, paths, L):
    cp.cuda.Device(cfg.run.gpu).use()

    real_dtype = cp.dtype(cfg.run.dtype)
    real_t = real_dtype.type

    dev = cp.cuda.Device(cfg.run.gpu)
    props = cp.cuda.runtime.getDeviceProperties(cfg.run.gpu)
    fm, tm = dev.mem_info

    L(f"[{paths['run_tag']}]")
    L(f"  out: {paths['out_dir']}/")
    L(f"  GPU{cfg.run.gpu} {props['name'].decode()} mem {fm/1e9:.1f}/{tm/1e9:.1f} GB")
    L(f"  dtype = {cfg.run.dtype}")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    Lx = cfg.domain.Lx_nm * 1e-9
    Ly = cfg.domain.Ly_nm * 1e-9
    Lc = cfg.domain.Lcyto_nm * 1e-9
    Lb = cfg.domain.Lbulk_nm * 1e-9
    Lm = cfg.domain.Lmem_nm * 1e-9

    Rp = cfg.physics.Rpore_nm * 1e-9

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    x_np = stretched_axis(
        Lx / 2,
        Lx / 2,
        cfg.mesh.dx_min_nm * 1e-9,
        cfg.mesh.ratio_xy,
        cfg.mesh.dx_max_nm * 1e-9,
    )
    y_np = stretched_axis(
        Ly / 2,
        Ly / 2,
        cfg.mesh.dy_min_nm * 1e-9,
        cfg.mesh.ratio_xy,
        cfg.mesh.dy_max_nm * 1e-9,
    )
    z_np = stretched_axis(
        Lc,
        Lb,
        cfg.mesh.dz_min_nm * 1e-9,
        cfg.mesh.ratio_z,
        cfg.mesh.dz_max_nm * 1e-9,
    )

    Nx, Ny, Nz = x_np.size, y_np.size, z_np.size
    shape = (Nx, Ny, Nz)

    x = cp.asarray(x_np, dtype=real_dtype)
    y = cp.asarray(y_np, dtype=real_dtype)
    z = cp.asarray(z_np, dtype=real_dtype)

    hxf_np, dxc_np = cell_metrics(x_np)
    hyf_np, dyc_np = cell_metrics(y_np)
    hzf_np, dzc_np = cell_metrics(z_np)

    hxf = cp.asarray(hxf_np, dtype=real_dtype)
    hyf = cp.asarray(hyf_np, dtype=real_dtype)
    hzf = cp.asarray(hzf_np, dtype=real_dtype)

    dxc = cp.asarray(dxc_np, dtype=real_dtype)
    dyc = cp.asarray(dyc_np, dtype=real_dtype)
    dzc = cp.asarray(dzc_np, dtype=real_dtype)

    vol_field = (dxc[:, None, None] * dyc[None, :, None] * dzc[None, None, :]).astype(real_dtype)

    L(f"  Mesh: {Nx}*{Ny}*{Nz} = {Nx*Ny*Nz}")
    L(f"  dx,dy,dz min nm: {hxf_np.min()*1e9:.3f}, {hyf_np.min()*1e9:.3f}, {hzf_np.min()*1e9:.3f}")
    L(f"  dx,dy,dz max nm: {hxf_np.max()*1e9:.3f}, {hyf_np.max()*1e9:.3f}, {hzf_np.max()*1e9:.3f}")

    dz_min = float(hzf_np.min())
    dx_min = float(hxf_np.min())
    dy_min = float(hyf_np.min())

    mem_z = cp.abs(z) <= (Lm / 2 + 0.5 * dz_min)
    n_mem = int(cp.sum(mem_z).get())
    if n_mem <= 0:
        raise RuntimeError("No membrane z planes; reduce dz_min_nm or Lmem_nm.")

    z_mem_vals = z[mem_z]
    z_inner_face = float(cp.min(z_mem_vals).get()) - 0.5 * dz_min
    z_outer_face = float(cp.max(z_mem_vals).get()) + 0.5 * dz_min

    cyto_z = z < z_inner_face
    bulk_z = z > z_outer_face

    mem_view = mem_z[None, None, :]
    cyto_view = cyto_z[None, None, :]
    bulk_view = bulk_z[None, None, :]

    cyto = cp.broadcast_to(cyto_view, shape).copy()
    bulk = cp.broadcast_to(bulk_view, shape).copy()

    # ------------------------------------------------------------------
    # Protein species 
    # ------------------------------------------------------------------
    proteins = get_protein_species(cfg.physics)
    PHI_MAX = cfg.physics.protein_pack_phi_max
    eta = 0.6913e-3
    for p in proteins:
        Rp_i = p["Rprot_nm"] * 1e-9
        p["Rprot_m"]   = Rp_i
        p["V_prot"]    = (4.0 / 3.0) * np.pi * Rp_i**3
        p["nprot_max"] = real_t(PHI_MAX / p["V_prot"])
        Dpw_i = kB * Tem / (6 * np.pi * eta * Rp_i)
        p["Dpc"] = Dpw_i / 3.0
        L(f"  protein '{p['name']}': C={p['Cprot_mM']:.3f} mM  z={p['zprot']:+g}  "
          f"R={p['Rprot_nm']:.3f} nm  Dpc={p['Dpc']:.3e}")

    # ------------------------------------------------------------------
    # Ions and immobile background
    # ------------------------------------------------------------------
    zC_prot = sum(p["zprot"] * p["Cprot_mM"] for p in proteins)

    fixed_cfg = cfg.physics.get("fixed_charge", None)
    if fixed_cfg:
        z_fix    = float(fixed_cfg.get("zfix", -10))
        C_fix    = float(fixed_cfg.get("Cfix_mM", 1.0))
        fix_side = str(fixed_cfg.get("side", "cyto"))
    else:
        z_fix = 0.0; C_fix = 0.0; fix_side = "cyto"

    fixed_signed_cyto = z_fix * C_fix if fix_side in ("cyto", "both") else 0.0
    fixed_signed_bulk = z_fix * C_fix if fix_side in ("bulk", "both") else 0.0
    fixed_osm_cyto    = C_fix if fix_side in ("cyto", "both") else 0.0
    fixed_osm_bulk    = C_fix if fix_side in ("bulk", "both") else 0.0

    ions, X_cyto_mM, X_bulk_mM, Y_cyto_mM, Y_bulk_mM, z_X = setup_immobile_background(
        DEFAULT_IONS, 1.0, zC_prot, 0.0,
        z_immobile=cfg.physics.get("z_immobile", -1),
        balance_osm=cfg.physics.get("balance_osm", True),
        fixed_signed_cyto=fixed_signed_cyto,
        fixed_signed_bulk=fixed_signed_bulk,
        fixed_osm_cyto=fixed_osm_cyto,
        fixed_osm_bulk=fixed_osm_bulk,
    )
    ions = apply_diffusivity_overrides(ions, cfg.physics.D_override, log=L)

    L(f"  immobile background X (z={z_X}): cyto={X_cyto_mM:.4f} mM, bulk={X_bulk_mM:.4f} mM")
    L(f"  immobile osmolyte  Y (z=0):     cyto={Y_cyto_mM:.4f} mM, bulk={Y_bulk_mM:.4f} mM")

    Cprot_tot = sum(p["Cprot_mM"] for p in proteins)
    zprot_avg = (zC_prot / Cprot_tot) if Cprot_tot > 0 else 0.0
    lD_c = debye_length(ions, "cyto", Cprot_tot, zprot_avg)
    lD_b = debye_length(ions, "bulk", 0.0, zprot_avg)
    lD = min(lD_c, lD_b)

    L(f"  lambda_D cyto={lD_c*1e9:.3f} nm bulk={lD_b*1e9:.3f} nm used={lD*1e9:.3f} nm")
    L(f"  lambda_D / dz_min = {lD/dz_min:.2f}")

    # dt stability
    Dmax = max([s["D"] for s in ions] + [p["Dpc"] for p in proteins])
    dt_diff = 0.45 / (2.0 * Dmax * (1.0 / dx_min**2 + 1.0 / dy_min**2 + 1.0 / dz_min**2))

    dt_init = float(cfg.time.dt)
    dt_max_cfg = cfg.time.get("dt_max", None)
    dt_max = 0.9 * dt_diff if dt_max_cfg is None else float(dt_max_cfg)

    L(f"  estimated explicit-diffusion CFL dt <= {dt_diff:.3e} s")
    L(f"  initial dt = {dt_init:.3e} s   dt_max = {dt_max:.3e} s")
    L(f"  t_resolve = {cfg.time.get('t_resolve_ns', 50.0):.1f} ns   "
      f"growth = {cfg.time.get('dt_growth', 1.03):.4f}/step")

    if dt_init > 0.9 * dt_diff:
        raise RuntimeError(
            f"time.dt={dt_init:.3e} exceeds 0.9*dt_diff={0.9*dt_diff:.3e}"
        )
    if dt_max > 0.9 * dt_diff:
        L(f"  WARNING: dt_max exceeds explicit CFL; will be capped at 0.9*dt_diff")
        dt_max = 0.9 * dt_diff

    # ------------------------------------------------------------------
    # Pores
    # ------------------------------------------------------------------
    explicit = parse_explicit_pores_nm(cfg.pores.pores_xy_nm)

    pore_spacing_nm = cfg.pores.pore_spacing_nm
    if cfg.pores.tangential:
        pore_spacing_nm = 2.0 * cfg.physics.Rpore_nm
        L(f"  pores.tangential=True -> pore_spacing_nm={pore_spacing_nm:.6g}")

    dynamic_events = []

    if cfg.run.case in ["A", "B", "C"] and explicit is not None:
        pores = explicit
    elif cfg.run.case == "A":
        pores = [(0.0, 0.0)]
    elif cfg.run.case == "B":
        pores = pores_grid(cfg.pores.n_pores, pore_spacing_nm)
    elif cfg.run.case == "C":
        pores = pores_half_cluster(cfg.pores.n_pores, pore_spacing_nm)
    else:
        # Case D: seed one initial pore at (0,0) so the k_efflux trigger
        # has something to leak through. For schedule trigger this simply
        # means pore #1 exists at t=0 instead of waiting one interval.
        if explicit is not None and len(explicit) > 0:
            pores = [explicit[0]]
            extra_explicit = list(explicit[1:])
        else:
            pores = [(0.0, 0.0)]
            extra_explicit = []

    if cfg.run.case == "D":
        n_dynamic = max(cfg.pores.n_pores - len(pores), 0)

        interval = cfg.dynamic.dyn_interval
        if interval <= 0.0:
            interval = cfg.time.t_final / max(cfg.pores.n_pores, 1)

        zx = tuple(cfg.dynamic.dyn_zone_x_nm) if cfg.dynamic.dyn_zone_x_nm is not None else (
            -cfg.domain.Lx_nm / 4,
            cfg.domain.Lx_nm / 4,
        )
        zy = tuple(cfg.dynamic.dyn_zone_y_nm) if cfg.dynamic.dyn_zone_y_nm is not None else (
            -cfg.domain.Ly_nm / 4,
            cfg.domain.Ly_nm / 4,
        )
        md = cfg.dynamic.dyn_min_dist_nm
        if md is None:
            md = 3.0 * cfg.physics.Rpore_nm

        dynamic_events = dynamic_random_schedule(
            n_dynamic,
            cfg.dynamic.dyn_t_start,
            interval,
            zx,
            zy,
            md,
            cfg.dynamic.dyn_seed,
            existing=pores,
        )

    L(f"  initial pores = {len(pores)}")
    if pores:
        for i, (px, py) in enumerate(pores):
            L(f"    pore {i}: x={px*1e9:+.3f} nm y={py*1e9:+.3f} nm")

    # ------------------------------------------------------------------
    # Convolution kernels / cache
    # ------------------------------------------------------------------
    
    conv = Convolver3D(cfg.solver.conv, cfg.solver.direct_kernel_max)
    for p in proteins:
        p["K_ball"]  = make_ball_kernel(p["Rprot_m"], dx_min, dy_min, dz_min,
                                        real_dtype, normalize=True)
        p["K_shell"] = make_shell_kernel(p["Rprot_m"], dx_min, dy_min, dz_min, real_dtype)
    L(f"  conv backend={cfg.solver.conv}  n_protein_species={len(proteins)}")

    x2 = x[:, None]
    y2 = y[None, :]
    z3 = z[None, None, :]

    # ------------------------------------------------------------------
    # build_masks closure
    # ------------------------------------------------------------------
    def build_masks(p_list):
        porexy = cp.zeros((Nx, Ny), dtype=cp.bool_)
        wallxy = cp.zeros((Nx, Ny), dtype=cp.bool_)

        shell = max(dx_min, dy_min)

        for cx, cy in p_list:
            r = cp.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
            porexy |= r <= Rp
            wallxy |= cp.abs(r - Rp) <= 0.5 * shell

        pore = (mem_view & porexy[:, :, None]).copy()
        lipid = (mem_view & (~porexy[:, :, None])).copy()
        mobile = (~lipid).astype(real_dtype)
        wall = (mem_view & wallxy[:, :, None]).copy()

        prot_mob_list = []
        for p in proteins:
            Rprot_i = p["Rprot_m"]
            z_excluded_1d = (z >= (z_inner_face - Rprot_i)) & (z <= (z_outer_face + Rprot_i))
            z_excluded_3d = z_excluded_1d[None, None, :]

            R_shrink = max(Rp - Rprot_i, 0.0)
            if p_list and R_shrink > 0.0:
                inside_shrunk_xy = cp.zeros((Nx, Ny), dtype=cp.bool_)
                for cx, cy in p_list:
                    r = cp.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
                    inside_shrunk_xy |= r <= R_shrink
                inside_shrunk = inside_shrunk_xy[:, :, None]
            else:
                inside_shrunk = cp.zeros((Nx, Ny, 1), dtype=cp.bool_)

            prot_accessible = (~z_excluded_3d) | inside_shrunk
            pm = (mobile.astype(cp.bool_) & prot_accessible).astype(real_dtype)
            prot_mob_list.append(pm)

        block_extra_nm = cfg.physics.pore_block_extra_nm
        if block_extra_nm is None:
            block_extra_nm = max(p["Rprot_nm"] for p in proteins)
        bx = block_extra_nm * 1e-9
        pbz = (
            porexy[:, :, None]
            & (z3 >= z_inner_face - bx)
            & (z3 <= z_outer_face + bx)
        ).astype(real_dtype)

        return pore, lipid, mobile, wall, prot_mob_list, pbz

    pore, lipid, mobile, wall, prot_mob_list, pore_block_zone = build_masks(pores)
    eps = cp.where(lipid, eps_l, eps_w).astype(real_dtype)

    L(f"  protein-accessible cells = {int(cp.sum(prot_mob_list[0] > 0.5).get())} of {int(cp.sum(mobile > 0.5).get())} mobile cells (species 0)")
    # ------------------------------------------------------------------
    # Initial C / D / protein
    # ------------------------------------------------------------------
    alpha = (0.5 * (1.0 - cp.tanh(z / (2 * Lm))))[None, None, :]

    C = []
    D_base = []
    for s in ions:
        Ci = (alpha * s["C_cyto"] + (1 - alpha) * s["C_bulk"]) * mobile
        Ci += real_t(1e-30) * (1.0 - mobile)
        Di = cp.where(lipid, 1e-24, s["D"]).astype(real_dtype)
        C.append(Ci.astype(real_dtype))
        D_base.append(Di)
    nprot_list = []
    Dprot_list = []
    for i, p in enumerate(proteins):
        npi = ((p["Cprot_mM"] * NA) * alpha * prot_mob_list[i]).astype(real_dtype)
        cp.minimum(npi, p["nprot_max"], out=npi)
        nprot_list.append(npi)
        Dprot_list.append(cp.where(lipid, 1e-24, p["Dpc"]).astype(real_dtype))

    phi = cp.zeros(shape, dtype=real_dtype)
    rw = cp.zeros(shape, dtype=real_dtype)

    prot_species = [dict(name=p["name"], zprot=p["zprot"],
                         K_ball=p["K_ball"], V_prot=p["V_prot"]) for p in proteins]
    cache = PhysicsCache(shape, real_dtype, ions, prot_species, conv, PHI_MAX)

    # --- Immobile fixed charge field (persistent Donnan source) ----------
    if fix_side == "cyto":
        fix_mask = cyto
    elif fix_side == "bulk":
        fix_mask = bulk
    else:
        fix_mask = cyto | bulk
    n_fix = (real_t(C_fix) * fix_mask.astype(real_dtype)) * mobile
    n_fix = n_fix.astype(real_dtype)

    ion_signed = cp.zeros_like(C[0])
    for Ci, sp in zip(C, ions):
        ion_signed += sp["z"] * Ci
    prot_signed = cp.zeros_like(ion_signed)
    for p, npi in zip(proteins, nprot_list):
        prot_signed += real_t(p["zprot"]) * (npi / NA)
    fixed_signed = real_t(z_fix) * n_fix
    n_X = (-(ion_signed + prot_signed + fixed_signed) / real_t(z_X))
    n_X = cp.maximum(n_X, 0.0) * mobile
    n_X = n_X.astype(real_dtype)

    ion_total = cp.zeros_like(n_X)
    for Ci in C:
        ion_total += Ci
    prot_total = cp.zeros_like(n_X)
    for npi in nprot_list:
        prot_total += npi / NA
    osm_no_Y = (ion_total + prot_total + n_X + n_fix) * mobile

    # Use the maximum among mobile cells as the flat target
    osm_target = float(cp.max(osm_no_Y * mobile).get())
    n_Y = cp.maximum(real_t(osm_target) - osm_no_Y, 0.0) * mobile
    n_Y = n_Y.astype(real_dtype)

    # --- Make X and Y MOBILE species -------------------------------------
    # X (z=z_X): charged background that neutralizes net charge locally.
    # Y (z=0)  : neutral osmolyte that flattens the initial osmolarity gap.
    # They are initialized to the balancing fields n_X, n_Y, then left free
    # to diffuse. They are NOT an immobile background: the only persistent
    # driving source for flux/swelling remains the charged protein.
    D_XY = 1.0e-9  # m^2/s; below the CFL-limiting ion D so dt stays valid

    ions.append(dict(name="X", z=z_X, D=D_XY,
                     C_cyto=X_cyto_mM, C_bulk=X_bulk_mM))
    ions.append(dict(name="Y", z=0.0, D=D_XY,
                     C_cyto=Y_cyto_mM, C_bulk=Y_bulk_mM))

    C.append((n_X * mobile).astype(real_dtype))
    C.append((n_Y * mobile).astype(real_dtype))
    D_base.append(cp.where(lipid, 1e-24, D_XY).astype(real_dtype))
    D_base.append(cp.where(lipid, 1e-24, D_XY).astype(real_dtype))

    # X and Y live in the mobile ion arrays. The ONLY persistent immobile
    # background is the fixed charge n_fix (contributes to both charge and osm).
    if float(cp.max(cp.abs(n_fix)).get()) > 0.0:
        rho_imm = (real_t(F * z_fix) * n_fix).astype(real_dtype)
        osm_imm = n_fix.astype(real_dtype)
        cache.set_immobile(rho_imm, osm_imm)
        L(f"  immobile fixed charge: z={z_fix:+g}, C={C_fix:.3f} mM, side={fix_side}")
    else:
        cache.set_immobile(None, None)

    L(f"  mobile X (z={z_X}): min={float(cp.min(n_X[mobile>0.5]).get()):.3f} "
    f"max={float(cp.max(n_X).get()):.3f} mM (initial local electroneutrality)")
    L(f"  mobile Y (z=0):   min={float(cp.min(n_Y[mobile>0.5]).get()):.3f} "
    f"max={float(cp.max(n_Y).get()):.3f} mM (initial osm target = {osm_target:.3f} mM)")
    # ------------------------------------------------------------------
    # Utility closures
    # ------------------------------------------------------------------
    def avg_region(a, m):
        w = m.astype(real_dtype) * vol_field
        return float((cp.sum(a * w) / (cp.sum(w) + 1e-300)).get())

    def rho_wall_field():
        if not pores:
            return cp.zeros_like(phi)

        shell_vol = float(cp.sum(wall.astype(real_dtype) * vol_field).get())
        if shell_vol <= 0:
            return cp.zeros_like(phi)

        area = len(pores) * 2.0 * np.pi * Rp * Lm
        dens = cfg.physics.sigma_wall * area / shell_vol
        return (dens * wall.astype(real_dtype)).astype(real_dtype)

    # ------------------------------------------------------------------
    # Local Cl electroneutrality
    # ------------------------------------------------------------------
    def enforce_local_cl_electroneutrality(label="", where_mask=None):
        cl_i = [i for i, sp in enumerate(ions) if sp["name"] == "Cl"][0]
        zcl = ions[cl_i]["z"]

        if where_mask is None:
            where = mobile > 0.5
        else:
            where = (where_mask > 0.5) & (mobile > 0.5)

        rm_no_cl = cp.zeros_like(phi)
        for q, sp in enumerate(ions):
            if q == cl_i:
                continue
            rm_no_cl += sp["z"] * C[q]
        for p, npi in zip(proteins, nprot_list):
            rm_no_cl += p["zprot"] * (npi / NA)
        Ccl_new = -rm_no_cl / zcl
        Ccl_new = cp.maximum(Ccl_new, real_t(1e-30))

        C[cl_i][...] = cp.where(where, Ccl_new, C[cl_i])
        C[cl_i][...] = cp.where(mobile > 0.5, C[cl_i], real_t(1e-30))

        cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
        L(f"  local Cl electroneutrality correction {label}, cells={int(cp.sum(where).get())}")

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    gate_one = cp.ones_like(phi)
    gate_prot = cp.empty_like(phi)
    protein_solid = cp.zeros_like(phi)
    ion_mobile_eff = cp.empty_like(phi)

    def update_gates():
        # Smooth crowding mobility factor, 1 in dilute regions, 0 near packing.
        cp.divide(cache.steric, PHI_MAX, out=gate_prot)
        cp.subtract(1.0, gate_prot, out=gate_prot)
        cp.clip(gate_prot, 0.0, 1.0, out=gate_prot)
        cp.multiply(gate_prot, gate_prot, out=gate_prot)
        cp.copyto(ion_mobile_eff, mobile)
        ps = (cache.steric >= cfg.physics.protein_block_phi) & (pore_block_zone > 0.5)
        protein_solid[...] = ps.astype(real_dtype)
    def enforce_ion_access():
        blocked = 1.0 - ion_mobile_eff
        for q in range(len(C)):
            C[q] *= ion_mobile_eff
            C[q] += real_t(1e-30) * blocked

    cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
    update_gates()
    enforce_ion_access()


    # ------------------------------------------------------------------
    # Wall screening closure
    # ------------------------------------------------------------------
    def initialize_wall_counterion_screening(label=""):
        nonlocal C

        if not cfg.physics.wall_screen_init:
            L(f"  wall counterion screening {label}: disabled")
            return

        if abs(cfg.physics.sigma_wall) <= 0.0 or not pores:
            L(f"  wall counterion screening {label}: skipped")
            return

        Q_wall = float(cp.sum(rw.astype(cp.float64) * vol_field.astype(cp.float64)).get())
        if abs(Q_wall) < 1e-30:
            L(f"  wall counterion screening {label}: skipped Q_wall~0")
            return

        Q_need = -Q_wall
        name_to_idx = {sp["name"]: i for i, sp in enumerate(ions)}

        if cfg.physics.wall_screen_ion != "auto":
            ion_name = cfg.physics.wall_screen_ion
        else:
            if Q_need > 0.0:
                ion_name = "K" if "K" in name_to_idx else "Na"
            else:
                ion_name = "Cl"

        qi = name_to_idx[ion_name]
        zion = ions[qi]["z"]

        mol_need = Q_need / (F * zion)
        if mol_need <= 0:
            raise RuntimeError("Wall screening selected wrong counterion sign.")

        R_screen = max(cfg.physics.wall_screen_radius_lambda * lD, 1.5 * max(dx_min, dy_min, dz_min))
        K_screen = make_ball_kernel(R_screen, dx_min, dy_min, dz_min, real_dtype, normalize=False)

        w = conv(wall.astype(real_dtype), K_screen)
        w = cp.maximum(w, 0.0)
        w *= ion_mobile_eff
        w *= mobile

        denom = cp.sum(w.astype(cp.float64) * vol_field.astype(cp.float64))
        if float(denom.get()) <= 0:
            raise RuntimeError("Wall screening: no accessible cells.")

        dc = (mol_need * w) / denom
        C[qi] += dc.astype(real_dtype)

        blocked = 1.0 - ion_mobile_eff
        C[qi] *= ion_mobile_eff
        C[qi] += real_t(1e-30) * blocked

        cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
        L(f"  wall counterion screening {label}: ion={ion_name}, Q_wall={Q_wall:+.3e} C")

    # ------------------------------------------------------------------
    # Poisson solver
    # ------------------------------------------------------------------
    L("  building Poisson PCG...")
    poisson = PoissonPCG_LJZ(eps, hxf, hyf, hzf, dxc, dyc, dzc)

    # ------------------------------------------------------------------
    # Diagnostic closures
    # ------------------------------------------------------------------
    def Vm():
        slab = max(lD, dz_min)
        in_m = (z3 >= z_inner_face - slab) & (z3 <= z_inner_face) & mobile.astype(cp.bool_)
        out_m = (z3 >= z_outer_face) & (z3 <= z_outer_face + slab) & mobile.astype(cp.bool_)
        return avg_region(phi, in_m) - avg_region(phi, out_m)

    def total_charge_C():
        rho = cache.charge_density(mobile, rw, use_wall=True)
        return float(cp.sum(rho * vol_field).get())

    z_face_np = 0.5 * (z_np[:-1] + z_np[1:])
    k_z0_face = int(np.argmin(np.abs(z_face_np)))
    z0_face_nm = float(z_face_np[k_z0_face] * 1e9)

    L(f"  current diagnostic plane z={z0_face_nm:+.4f} nm")
    L("  current convention: positive means z<0 -> z>0")

    # Persistent scratch for batched z=0 flux diagnostic
    n_prot = len(proteins)
    _flux_out_gpu = cp.zeros(1 + n_prot + len(ions), dtype=cp.float64)
    _flux_area_z  = (dxc[:, None] * dyc[None, :]).astype(cp.float64)
    _flux_I_tot   = cp.zeros((), dtype=cp.float64)

    def all_z0_fluxes_gpu():
        """Return [I_total_A, prot_flux_0..P-1, ion_flux_0..n-1] as one GPU array.
        Positive convention: z<0 -> z>0."""
        k = k_z0_face
        out_gpu = _flux_out_gpu; out_gpu.fill(0.0)
        area_z = _flux_area_z
        I_tot = _flux_I_tot; I_tot.fill(0.0)

        for qi in range(len(ions)):
            Ci = C[qi]; Di = D_base[qi]; zval = ions[qi]["z"]
            c0 = cp.maximum(Ci[:, :, k],     real_t(1e-30))
            c1 = cp.maximum(Ci[:, :, k + 1], real_t(1e-30))
            D0 = Di[:, :, k]     * ion_mobile_eff[:, :, k]
            D1 = Di[:, :, k + 1] * ion_mobile_eff[:, :, k + 1]
            Df = harm(D0, D1)
            psi = zval * (phi[:, :, k + 1] - phi[:, :, k]) / VT
            J = -(Df / hzf[k]) * (c1 * bern_cp(-psi) - c0 * bern_cp(psi))
            J = cp.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
            J64 = J.astype(cp.float64) * area_z
            out_gpu[1 + n_prot + qi] = cp.sum(J64)
            I_tot = I_tot + (zval * F) * cp.sum(J64)

        for pi, p in enumerate(proteins):
            npi = nprot_list[pi]; Dp = Dprot_list[pi]; pm = prot_mob_list[pi]
            zp = p["zprot"]
            c0 = cp.maximum(npi[:, :, k]     / NA, real_t(1e-30))
            c1 = cp.maximum(npi[:, :, k + 1] / NA, real_t(1e-30))
            D0 = Dp[:, :, k]     * gate_prot[:, :, k]     * pm[:, :, k]
            D1 = Dp[:, :, k + 1] * gate_prot[:, :, k + 1] * pm[:, :, k + 1]
            Df = harm(D0, D1)
            psi = zp * (phi[:, :, k + 1] - phi[:, :, k]) / VT
            Jp = -(Df / hzf[k]) * (c1 * bern_cp(-psi) - c0 * bern_cp(psi))
            Jp = cp.nan_to_num(Jp, nan=0.0, posinf=0.0, neginf=0.0)
            Jp64 = Jp.astype(cp.float64) * area_z
            out_gpu[1 + pi] = cp.sum(Jp64)
            I_tot = I_tot + (zp * F) * cp.sum(Jp64)

        out_gpu[0] = I_tot
        return out_gpu
        
    def solve_poisson(use_wall, rtol=None, maxit=None, cold=False, label=""):
        nonlocal phi

        rtol_use = cfg.solver.cg_rtol if rtol is None else rtol
        maxit_use = cfg.solver.cg_maxit if maxit is None else maxit

        rtol_floor = 3.0e-6 if real_dtype == cp.float32 else 1.0e-12
        if rtol_use < rtol_floor:
            rtol_use = rtol_floor
        rho = cache.charge_density(mobile, rw, use_wall=use_wall)
        if not bool(cp.all(cp.isfinite(rho)).get()):
            raise FloatingPointError("rho non-finite before Poisson")

        phi, info = poisson.solve(rho, phi, rtol=rtol_use, maxit=maxit_use, cold=cold)

        if not bool(cp.all(cp.isfinite(phi)).get()):
            raise FloatingPointError(f"phi non-finite after Poisson {label}")

        pabs = float(cp.max(cp.abs(phi)).get())
        if info >= maxit_use:
            raise RuntimeError(f"Poisson PCG failed {label}: info={info}, max|phi|={pabs:.3e}")

        return info

    # ------------------------------------------------------------------
    # Osm monitor distance
    # ------------------------------------------------------------------
    plot_cfg = load_plot_limits(paths["out_dir"])
    osm_cfg = (plot_cfg.get("osm_monitor") or {}) if isinstance(plot_cfg, dict) else {}
    user_dist_nm = osm_cfg.get("local_distance_nm", None)

    def auto_detect_osm_local_distance_m():
        cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
        osm = cache.osmolarity(C, nprot_list)
        m = mobile.astype(real_dtype)
        w_xy = m * (dxc[:, None, None] * dyc[None, :, None])
        num = cp.sum(osm * w_xy, axis=(0, 1))
        den = cp.sum(w_xy, axis=(0, 1)) + real_t(1e-30)
        osm_z = (num / den).astype(cp.float64)
        z64 = z.astype(cp.float64)

        dosm = cp.zeros_like(osm_z)
        dosm[1:-1] = (osm_z[2:] - osm_z[:-2]) / (z64[2:] - z64[:-2])
        dosm[0] = (osm_z[1] - osm_z[0]) / (z64[1] - z64[0])
        dosm[-1] = (osm_z[-1] - osm_z[-2]) / (z64[-1] - z64[-2])

        valid = cp.sum(m, axis=(0, 1)) > 0
        adosm = cp.where(valid, cp.abs(dosm), 0.0)
        kmax = int(cp.argmax(adosm).get())
        return max(abs(float(z[kmax].get())), 2.0 * dz_min)

    if user_dist_nm is None:
        osm_local_dist_m = auto_detect_osm_local_distance_m()
        L(f"  osm local-monitor distance = {osm_local_dist_m*1e9:.4f} nm auto")
    else:
        osm_local_dist_m = float(user_dist_nm) * 1e-9
        L(f"  osm local-monitor distance = {osm_local_dist_m*1e9:.4f} nm from plot_limits.json")

    def osm_cyto_bulk_mM():
        osm = cache.osmolarity(C, nprot_list)
        d = real_t(osm_local_dist_m)
        mob_b = mobile > 0.5

        cyto_local = cyto & mob_b & (z3 >= -d)
        bulk_local = bulk & mob_b & (z3 <= d)

        if int(cp.sum(cyto_local).get()) == 0:
            cyto_local = cyto & mob_b
        if int(cp.sum(bulk_local).get()) == 0:
            bulk_local = bulk & mob_b

        return avg_region(osm, cyto_local), avg_region(osm, bulk_local)

    # ------------------------------------------------------------------
    # Initial Poisson and Vm shift
    # ------------------------------------------------------------------
 
    L(""); L("Imposing v_rest by multi-ion Debye-layer shift (Boltzmann-weighted), wall OFF:")
    rw[...] = 0.0
    slab = max(lD, 1.5 * dz_min)
    in_mask  = (z3 >= z_inner_face - slab) & (z3 <= z_inner_face) & mobile.astype(cp.bool_)
    out_mask = (z3 >= z_outer_face) & (z3 <= z_outer_face + slab) & mobile.astype(cp.bool_)
    Vin  = float(cp.sum(in_mask.astype(real_dtype) * vol_field).get())
    Vout = float(cp.sum(out_mask.astype(real_dtype) * vol_field).get())
    if Vin <= 0 or Vout <= 0:
        raise RuntimeError("Empty Debye layer.")

    target = cfg.physics.v_rest

    def Vm_after_shift():
        cache.update_ions(C)
        solve_poisson(use_wall=False, rtol=1e-9, maxit=5000, cold=True,
                    label="Vm_after_shift no-wall")
        return Vm()

    V_cur = Vm_after_shift()
    L(f"  Vm before shift = {V_cur*1e3:+.4f} mV   target = {target*1e3:+.4f} mV")

    # --- Build per-species Boltzmann weights and motion directions --------------
    # Weight w_i = z_i^2 * c_i_avg  (positive). Higher concentration / higher |z|
    # species absorb more of the layer charge, exactly as in linearized
    # Debye-Hückel theory.
    #
    # Direction sign: a species moves from cyto-side layer to bulk-side layer
    # (positive d_i) if sign(z_i)*direction > 0, otherwise vice versa.
    # This guarantees the shift always drives Vm toward the target monotonically.

    direction = +1.0 if (target - V_cur) < 0.0 else -1.0  # +1: lower Vm; -1: raise Vm

    species_shift = []  # list of (idx, z_i, w_i, dir_sign, Cbase)
    W_sum = 0.0
    for idx, sp in enumerate(ions):
        z_i = sp["z"]
        if z_i == 0:
            continue
        if sp["name"] in ("X", "Y"):     # balancer species: excluded from Vm shift
            continue
        c_avg = 0.5 * (sp["C_cyto"] + sp["C_bulk"])
        w_i = (z_i * z_i) * c_avg
        dir_sign = +1.0 if (z_i * direction > 0) else -1.0
        species_shift.append([idx, z_i, w_i, dir_sign, C[idx].copy()])
        W_sum += w_i

    if W_sum <= 0.0:
        raise RuntimeError("No mobile ions to build Debye layer with.")

    # Normalize weights
    for s in species_shift:
        s[2] = s[2] / W_sum

    L("  shift composition (z, weight, direction):")
    for idx, z_i, w_i, dir_sign, _ in species_shift:
        L(f"    {ions[idx]['name']:4s} z={z_i:+d}  w={w_i:.3f}  "
        f"{'in->out' if dir_sign>0 else 'out->in'}")

    # Maximum allowable d_total before any species concentration would go negative.
    # For each species: amount removed from its donor layer is w_i * d_total.
    # Donor layer = in_mask if dir_sign>0, else out_mask.
    max_d = float("inf")
    for idx, z_i, w_i, dir_sign, Cbase in species_shift:
        if w_i <= 0:
            continue
        donor = in_mask if dir_sign > 0 else out_mask
        src_min = float(cp.min(Cbase[donor]).get())
        if src_min <= 0:
            continue
        cap = 0.99 * src_min / w_i
        if cap < max_d:
            max_d = cap
    L(f"  max_d (Boltzmann-weighted total) = {max_d:.3e}")

    def apply_d(d):
        """Apply a Boltzmann-weighted shift of total magnitude d.
        For species i: move d*w_i from donor layer to acceptor layer (volume-balanced)."""
        for idx, z_i, w_i, dir_sign, Cbase in species_shift:
            Cn = Cbase.copy()
            di = w_i * d
            if dir_sign > 0:
                # cyto layer (in_mask) -> bulk layer (out_mask)
                Cn[in_mask]  -= di
                Cn[out_mask] += di * Vin / Vout
            else:
                Cn[in_mask]  += di * Vout / Vin
                Cn[out_mask] -= di
            if bool(cp.any(Cn < 0).get()):
                return None
            C[idx] = Cn
        return Vm_after_shift()

    # Bracket then bisect
    lo, hi = 0.0, min(max(1e-3, max_d * 0.05), max_d)
    Vh = apply_d(hi)
    Vl = apply_d(lo)
    if Vh is None:
        hi = 0.5 * max_d
        Vh = apply_d(hi)
    f_lo = Vl - target
    f_hi = Vh - target

    n = 0
    while f_lo * f_hi > 0 and hi < max_d and n < 30:
        hi = min(2 * hi, max_d)
        Vh = apply_d(hi)
        f_hi = Vh - target
        n += 1

    d_star = hi
    V_star = Vh

    if f_lo * f_hi <= 0:
        for it in range(60):
            mid = 0.5 * (lo + hi)
            Vm_mid = apply_d(mid)
            fm = Vm_mid - target
            d_star = mid
            V_star = Vm_mid
            if abs(fm * 1e3) < 0.05:
                break
            if f_lo * fm <= 0:
                hi = mid; f_hi = fm
            else:
                lo = mid; f_lo = fm

    V_star = apply_d(d_star)
    L(f"  shifted Vm = {V_star*1e3:+.4f} mV   d_total = {d_star:.3e}")
    cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list); update_gates(); enforce_ion_access()
    initialize_wall_counterion_screening("(after Vm shift)")
    cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
    solve_poisson(use_wall=True, cold=True, label="after Vm shift with-wall + screened wall")
    Vm_w = Vm()
    L(f"  Vm with wall after shift + screened wall = {Vm_w*1e3:+.4f} mV")

    cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
    rho_after = cache.charge_density(mobile, rw, use_wall=True)
    L(f"  rho after Vm shift: sum(rho*V)={float(cp.sum(rho_after*vol_field).get()):+.3e} C")
    
    # ------------------------------------------------------------------
    # ROI (Region of Interest)
    # ------------------------------------------------------------------
    roi_x = np.where(np.abs(x_np) <= cfg.output.roi_xy_nm * 1e-9)[0]
    roi_y = np.where(np.abs(y_np) <= cfg.output.roi_xy_nm * 1e-9)[0]
    roi_z = np.where(np.abs(z_np) <= cfg.output.roi_z_nm * 1e-9)[0]

    sl = (
        slice(roi_x[0], roi_x[-1] + 1),
        slice(roi_y[0], roi_y[-1] + 1),
        slice(roi_z[0], roi_z[-1] + 1),
    )
    L(f"  ROI: {roi_x.size}*{roi_y.size}*{roi_z.size}")

    state = SimpleNamespace(**locals())

    return state