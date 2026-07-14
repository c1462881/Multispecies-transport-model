#!/usr/bin/env python3
"""
time_loop.py

Production time loop, Case-D schedule/K-efflux trigger, NPZ/VTK saving.
"""

import os
import time

import cupy as cp
import numpy as np

from configuration import NA, VT, eps_l, eps_w
from output import write_vtr_binary, write_trajectory_csv, save_params_json
from solver import sg_update, grad_phi, PoissonPCG_LJZ


def run_time_loop(state, cfg, paths, timer, L):
    real_dtype = state.real_dtype
    real_t = state.real_t

    # Pull frequently-used variables from state namespace.
    C = state.C
    Ctmp = [cp.empty_like(Ci) for Ci in C]

    proteins = state.proteins
    nprot_list = state.nprot_list
    nprot_tmp_list = [cp.empty_like(n) for n in nprot_list]
    prot_mob_list = state.prot_mob_list
    Dprot_list = state.Dprot_list

    phi = state.phi
    phi_prev = phi.copy()
    rw = state.rw
    poisson = state.poisson

    pores = state.pores
    dynamic_events = state.dynamic_events
    ions = state.ions

    mobile = state.mobile
    lipid = state.lipid
    wall = state.wall
    pore = state.pore
    pore_block_zone = state.pore_block_zone
    eps = state.eps

    D_base = state.D_base

    cache = state.cache
    gate_one = state.gate_one
    gate_prot = state.gate_prot
    ion_mobile_eff = state.ion_mobile_eff

    vol_field = state.vol_field

    hxf, hyf, hzf = state.hxf, state.hyf, state.hzf
    dxc, dyc, dzc = state.dxc, state.dyc, state.dzc
    x, y, z = state.x, state.y, state.z
    x_np, y_np, z_np = state.x_np, state.y_np, state.z_np

    alpha = state.alpha

    build_masks = state.build_masks
    update_gates = state.update_gates
    enforce_ion_access = state.enforce_ion_access
    enforce_local_cl_electroneutrality = state.enforce_local_cl_electroneutrality
    initialize_wall_counterion_screening = state.initialize_wall_counterion_screening
    rho_wall_field = state.rho_wall_field

    Vm = state.Vm
    total_charge_C = state.total_charge_C
    all_z0_fluxes_gpu = state.all_z0_fluxes_gpu
    osm_cyto_bulk_mM = state.osm_cyto_bulk_mM

    sl = state.sl

    # ------------------------------------------------------------------
    # Saving functions
    # ------------------------------------------------------------------
    def save_npz_roi(step, t):
        cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
        update_gates()

        Ex, Ey, Ez = grad_phi(phi, x, y, z, hxf, hyf, hzf)
        Emag = cp.sqrt(Ex * Ex + Ey * Ey + Ez * Ez)
        osm = cache.osmolarity(C, nprot_list)
        rho = cache.charge_density(mobile, rw, use_wall=True)
        Qtot = float(cp.sum(rho * vol_field).get())

        def sub(A):
            return cp.asnumpy(A[sl]).astype(np.float32)

        prot_total = cp.zeros_like(nprot_list[0])
        for npi in nprot_list:
            prot_total += npi / NA

        d = dict(
            t_s=np.float32(t),
            Q_total_C=np.float32(Qtot),
            x_nm=(x_np[sl[0]] * 1e9).astype(np.float32),
            y_nm=(y_np[sl[1]] * 1e9).astype(np.float32),
            z_nm=(z_np[sl[2]] * 1e9).astype(np.float32),
            pores=np.array(pores),
            phi_mV=sub(phi * 1e3),
            protein_mM=sub(prot_total),
            osmolarity_mM=sub(osm),
            rho_Cpm3=sub(rho),
            Emag_Vpm=sub(Emag),
            Ex_Vpm=sub(Ex),
            Ey_Vpm=sub(Ey),
            Ez_Vpm=sub(Ez),
            steric=sub(cache.steric),
        )
        for p, npi in zip(proteins, nprot_list):
            d[f"{p['name']}_mM"] = sub(npi / NA)
        for Ci, s in zip(C, ions):
            d[s["name"]] = sub(Ci)

        np.savez(os.path.join(paths["npz_dir"], f"state_{step:07d}.npz"), **d)

    def save_vtr_full(step):
        cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
        update_gates()

        Ex, Ey, Ez = grad_phi(phi, x, y, z, hxf, hyf, hzf)
        Emag = cp.sqrt(Ex * Ex + Ey * Ey + Ez * Ez)
        osm = cache.osmolarity(C, nprot_list)
        rho = cache.charge_density(mobile, rw, use_wall=True)

        names = []
        arrs = []

        def add(n, a):
            names.append(n)
            arrs.append(a.astype(cp.float32))

        prot_total = cp.zeros_like(nprot_list[0])
        for npi in nprot_list:
            prot_total += npi / NA

        add("phi_mV", phi * 1e3)
        add("protein_mM", prot_total)
        add("osmolarity_mM", osm)
        add("steric", cache.steric)
        add("Emag_Vpm", Emag)
        add("rho_Cpm3", rho)
        add("pore_mask", pore.astype(cp.float32))
        add("wall_mask", wall.astype(cp.float32))

        for p, npi in zip(proteins, nprot_list):
            add(f"{p['name']}_mM", npi / NA)
        for Ci, s in zip(C, ions):
            add(s["name"], Ci)

        stacked = cp.asnumpy(cp.stack(arrs, axis=0))
        fields = {nm: stacked[i] for i, nm in enumerate(names)}

        write_vtr_binary(
            os.path.join(paths["vtk_dir"], f"fields_{step:07d}.vtr"),
            fields,
            x_np.astype(np.float32),
            y_np.astype(np.float32),
            z_np.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Optional protein mobility refresh at z=0
    # ------------------------------------------------------------------
    def refresh_prot_mob_on_open_z0_pores(reason=""):
        k = state.k_z0_face
        if k < 0 or k + 1 >= phi.shape[2]:
            return

        open_face_bool = cp.logical_and(
            mobile[:, :, k] > 0.5,
            mobile[:, :, k + 1] > 0.5,
        )
        n_open = int(cp.sum(open_face_bool).get())
        if n_open == 0:
            L(f"  refresh prot_mob z=0 ({reason}): n_open=0")
            return

        one = cp.asarray(1.0, dtype=real_dtype)
        for pi, p in enumerate(proteins):
            pm = prot_mob_list[pi]; Dp = Dprot_list[pi]; npi = nprot_list[pi]
            Dpc_gpu = cp.asarray(p["Dpc"], dtype=real_dtype)

            pm[:, :, k]     = cp.where(open_face_bool, one, pm[:, :, k]).astype(real_dtype)
            pm[:, :, k + 1] = cp.where(open_face_bool, one, pm[:, :, k + 1]).astype(real_dtype)
            Dp[:, :, k]     = cp.where(open_face_bool, Dpc_gpu, Dp[:, :, k]).astype(real_dtype)
            Dp[:, :, k + 1] = cp.where(open_face_bool, Dpc_gpu, Dp[:, :, k + 1]).astype(real_dtype)

            nref_k  = (p["Cprot_mM"] * NA) * alpha[:, :, k]
            nref_k1 = (p["Cprot_mM"] * NA) * alpha[:, :, k + 1]
            npi[:, :, k]     = cp.where(open_face_bool,
                                        cp.maximum(npi[:, :, k], nref_k.astype(real_dtype)),
                                        npi[:, :, k]).astype(real_dtype)
            npi[:, :, k + 1] = cp.where(open_face_bool,
                                        cp.maximum(npi[:, :, k + 1], nref_k1.astype(real_dtype)),
                                        npi[:, :, k + 1]).astype(real_dtype)
            npi[:, :, k]     *= pm[:, :, k]
            npi[:, :, k + 1] *= pm[:, :, k + 1]

        L(f"  refresh prot_mob z=0 ({reason}): n_open={n_open}")

    cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
    update_gates()
    refresh_prot_mob_on_open_z0_pores("initial geometry")
    cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
    update_gates()

    # ------------------------------------------------------------------
    # Adaptive dt schedule
    # ------------------------------------------------------------------
    t_final = float(cfg.time.t_final)
    dt_init = float(state.dt_init)
    dt_max  = float(state.dt_max)
    t_resolve = float(cfg.time.get("t_resolve_ns", 50.0)) * 1e-9
    growth = float(cfg.time.get("dt_growth", 1.03))

    # Uniform-time save grids 
    def build_event_times(n):
        if n <= 0:
            return np.array([], dtype=np.float64)
        return np.linspace(t_final / n, t_final, n)

    SAVE_INTERVAL_NS = 2.5
    n_saves = int(np.floor(t_final * 1e9 / SAVE_INTERVAL_NS + 1e-9))
    save_times = (np.arange(1, n_saves + 1) * SAVE_INTERVAL_NS) * 1e-9
    save_times = save_times[save_times <= t_final + 1e-15]
    vtk_times   = build_event_times(int(cfg.output.vtk_number))
    print_times = build_event_times(200)  # ~200 progress lines per run

    save_idx = 0
    vtk_idx = 0
    print_idx = 0

    L(f"  NPZ save: {save_times.size} frames, uniform in t ∈ (0, {t_final*1e9:.3g}] ns")
    L(f"  VTK save: {vtk_times.size} frames")
    L("")
    L(f"time loop: t_final={t_final:.3e} s   "
      f"dt_init={dt_init:.3e}   dt_max={dt_max:.3e}")

    extra_ns = cfg.output.get("extra_save_times_ns", []) or []
    if extra_ns:
        extra = np.array(extra_ns, dtype=np.float64) * 1e-9
        save_times = np.unique(np.concatenate([save_times, extra]))
    # ------------------------------------------------------------------
    # CSV header
    # ------------------------------------------------------------------
    prot_cols = "".join(
        f",{p['name']}_flux_z0_mol_s,{p['name']}_leakage_z0_mol,{p['name']}_leakage_z0_molecules"
        for p in proteins
    )
    ion_flux_cols = "".join(
        f",{sp['name']}_flux_z0_mol_s,{sp['name']}_efflux_z0_mol"
        for sp in ions
    )
    csv_header = (
        "step,frac,t_s,t_ns,Vm_mV,n_pores,"
        "osm_cyto_local_mM,osm_bulk_local_mM"
        + prot_cols +
        ",Q_total_C,I_z0_A,PCG_it,dt_s"
        + ion_flux_cols
    )
    log_rows = []
    with open(paths["csv_path"], "w") as _f:
        _f.write(csv_header + "\n")

    def _append_csv_row(row):
        with open(paths["csv_path"], "a") as _f:
            _f.write(",".join(repr(float(v)) if isinstance(v, float) else str(v)
                            for v in row) + "\n")

    # Initial save at t=0
    if save_times.size > 0:
        save_npz_roi(0, 0.0)
    if vtk_times.size > 0:
        save_vtr_full(0)

    # ------------------------------------------------------------------
    # Case-D trigger state 
    # ------------------------------------------------------------------
    event_i = 0
    protein_leakage_mol = [0.0 for _ in proteins]
    protein_flux_now    = [0.0 for _ in proteins]
    ion_efflux_z0_mol   = [0.0 for _ in ions]
    ion_flux_now        = [0.0 for _ in ions]
    k_idx_trigger = next((i for i, sp in enumerate(ions) if sp["name"] == "K"), None)
    if cfg.run.case == "D" and cfg.dynamic.dyn_trigger == "k_efflux":
        if k_idx_trigger is None:
            raise RuntimeError("dynamic.dyn_trigger=k_efflux requires K ion.")
        if not pores:
            L("  WARNING: D k_efflux trigger with no initial pore.")

    t0_wall = time.perf_counter()

    # ------------------------------------------------------------------
    # Adaptive time loop
    # ------------------------------------------------------------------
    t = 0.0
    step = 0
    dt_cur = dt_init

    EPS_T = 1e-15
    # ------------------------------------------------------------------
    # Convergence monitor (early stop on saturated ion efflux)
    # ------------------------------------------------------------------
    conv_cfg = cfg.get("convergence", {}) if hasattr(cfg, "get") else {}
    conv_enable = bool(conv_cfg.get("enable", False))
    conv_tol    = float(conv_cfg.get("efflux_rate_tol_mol_per_s", 1.0e-18))
    conv_N      = int(conv_cfg.get("consecutive_checks", 5))
    conv_t_min  = float(conv_cfg.get("min_time_ns", 50.0)) * 1e-9
    conv_emin   = float(conv_cfg.get("min_efflux_mol", 1.0e-22))
    conv_species = str(conv_cfg.get("species", "cations")).strip()

    conv_idx = list(range(len(ions)))

    if conv_enable:
        which = ",".join(ions[i]["name"] for i in conv_idx) or "(none)"
        L(f"  early-stop: max|d|efflux_i|/dt| < {conv_tol:.2e} mol/s "
          f"for {conv_N} prints, after t>={conv_t_min*1e9:.1f} ns, "
          f"species={which}")

    prev_efflux_vec = None
    prev_efflux_t   = None
    quiet_streak    = 0
    converged_early = False
    equilibrium_notified = False  
    # Mask for printing phi range away from the membrane.
    # Membrane spans |z| < Lmem/2; anything outside is "bulk side" or "cyto side".
    Lmem = float(cfg.domain.Lmem_nm) * 1e-9
    z_3d = z.reshape(1, 1, -1)
    cyto_mask = (z_3d < -Lmem)          # z < -Lmem/2  (cytosol half-space, away from mem)
    bulk_mask = (z_3d >  Lmem)          # z > +Lmem/2  (extracellular half-space)
    while t < t_final - EPS_T:
        step += 1

        # ---- Choose dt for this step --------------------------------
        if t >= t_resolve:
            dt_cur = min(dt_cur * growth, dt_max)

        # Snap to next scheduled event (save / vtk / print / final)
        next_event = t_final
        if save_idx  < save_times.size:  next_event = min(next_event, float(save_times[save_idx]))
        if vtk_idx   < vtk_times.size:   next_event = min(next_event, float(vtk_times[vtk_idx]))
        if print_idx < print_times.size: next_event = min(next_event, float(print_times[print_idx]))

        dt_step = min(dt_cur, next_event - t)
        if dt_step <= EPS_T:
            break

        # ---- Case-D pore insertion (uses real t) --------------------
        if cfg.run.case == "D":
            changed = False
            if cfg.dynamic.dyn_trigger == "k_efflux":
                cum_K = ion_efflux_z0_mol[k_idx_trigger]
                while (event_i < len(dynamic_events)
                       and cum_K >= (event_i + 1) * cfg.dynamic.dyn_k_threshold_mol):
                    ev = dynamic_events[event_i]
                    pores.append((ev["x"], ev["y"]))
                    event_i += 1
                    changed = True
                    L(f"  Case-D K-efflux: pore #{event_i} at t={t*1e9:.3f} ns "
                      f"cum K={cum_K:.3e} mol")
            else:
                while event_i < len(dynamic_events) and t >= dynamic_events[event_i]["t"]:
                    ev = dynamic_events[event_i]
                    pores.append((ev["x"], ev["y"]))
                    event_i += 1
                    changed = True
  
            if changed:
                old_mobile = mobile.copy()
                new_pore, new_lipid, new_mobile, new_wall, new_prot_mob_list, new_pbz = build_masks(pores)
                pore[...] = new_pore;   lipid[...] = new_lipid
                mobile[...] = new_mobile; wall[...] = new_wall
                for pi in range(len(proteins)):
                    prot_mob_list[pi][...] = new_prot_mob_list[pi]
                pore_block_zone[...] = new_pbz
                eps[...] = cp.where(lipid, eps_l, eps_w).astype(real_dtype)
                rw[...] = rho_wall_field()
                initialize_wall_counterion_screening("(dynamic pore insertion)")
                for q, sp in enumerate(ions):
                    D_base[q][...] = cp.where(lipid, 1e-24, sp["D"]).astype(real_dtype)
                for pi, p in enumerate(proteins):
                    Dprot_list[pi][...] = cp.where(lipid, 1e-24, p["Dpc"]).astype(real_dtype)
                cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
                update_gates(); enforce_ion_access()
                refresh_prot_mob_on_open_z0_pores("dynamic insertion")

                newly_open = (mobile > 0.5) & (old_mobile < 0.5)
                for q, sp in enumerate(ions):
                    Ci_ref = alpha * sp["C_cyto"] + (1.0 - alpha) * sp["C_bulk"]
                    C[q][...] = cp.where(newly_open, Ci_ref, C[q])
                    C[q] *= mobile
                    C[q] += real_t(1e-30) * (1.0 - mobile)
                for pi, p in enumerate(proteins):
                    nref = (p["Cprot_mM"] * NA) * alpha * prot_mob_list[pi]
                    nprot_list[pi][...] = cp.where(newly_open, nref, nprot_list[pi])
                    nprot_list[pi] *= prot_mob_list[pi]
                    cp.minimum(nprot_list[pi], p["nprot_max"], out=nprot_list[pi])
                enforce_local_cl_electroneutrality(
                    "(after pore insertion)", where_mask=newly_open
                )
                timer.tic("solver_rebuild")
                poisson = PoissonPCG_LJZ(eps, hxf, hyf, hzf, dxc, dyc, dzc)
                timer.toc("solver_rebuild")
        # ---- Cache + Poisson ----------------------------------------
        timer.tic("cache")
        cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
        update_gates()
        timer.toc("cache")

        timer.tic("poisson")
        rho = cache.charge_density(mobile, rw, use_wall=True)
        if step >= 2:
            phi_predict = 2.0 * phi - phi_prev
        else:
            phi_predict = phi
        phi_prev = phi.copy()
        phi, info = poisson.solve(
            rho, phi_predict,
            rtol=cfg.solver.cg_rtol, maxit=cfg.solver.cg_maxit, cold=False,
        )
        if info >= cfg.solver.cg_maxit:
            raise RuntimeError(f"Poisson PCG failed step={step} info={info}")
        timer.toc("poisson")
        state.phi = phi

        # ---- Flux diagnostics ---------------------------------------
        fluxes_cpu = cp.asnumpy(all_z0_fluxes_gpu())
        I_z0_A = float(fluxes_cpu[0])
        n_prot = len(proteins)
        for pi in range(n_prot):
            fp = float(fluxes_cpu[1 + pi])
            protein_flux_now[pi] = fp
            protein_leakage_mol[pi] += max(fp, 0.0) * dt_step
        for qi in range(len(ions)):
            Jq = float(fluxes_cpu[1 + n_prot + qi])
            ion_flux_now[qi] = Jq
            ion_efflux_z0_mol[qi] += max(Jq, 0.0) * dt_step

        # ---- SG updates with adaptive dt ----------------------------
        timer.tic("ion_sg")
        for i, s in enumerate(ions):
            sg_update(C[i], D_base[i], gate_prot, s["z"], phi,
                      hxf, hyf, hzf, dxc, dyc, dzc,
                      dt_step, ion_mobile_eff, Ctmp[i])
        for qi in range(len(C)):
            C[qi], Ctmp[qi] = Ctmp[qi], C[qi]
        timer.toc("ion_sg")

        timer.tic("prot_sg")
        for pi, p in enumerate(proteins):
            sg_update(nprot_list[pi], Dprot_list[pi], gate_prot, p["zprot"], phi,
                      hxf, hyf, hzf, dxc, dyc, dzc,
                      dt_step, prot_mob_list[pi], nprot_tmp_list[pi])
            nprot_list[pi][...] = nprot_tmp_list[pi]
            cp.clip(nprot_list[pi], 0.0, p["nprot_max"], out=nprot_list[pi])
            nprot_list[pi] *= prot_mob_list[pi]
        timer.toc("prot_sg")
        # ---- Advance time -------------------------------------------
        t += dt_step

        # ---- Periodic safety check ----------------------------------
        if step <= 100 or step % 200 == 0:
            if not bool(cp.all(cp.isfinite(phi)).get()):
                raise FloatingPointError(f"phi non-finite step={step}")
            for pi, npi in enumerate(nprot_list):
                if not bool(cp.all(cp.isfinite(npi)).get()):
                    raise FloatingPointError(
                        f"protein '{proteins[pi]['name']}' non-finite step={step}") 
            for qi, Ci in enumerate(C):
                if not bool(cp.all(cp.isfinite(Ci)).get()):
                    raise FloatingPointError(f"ion {ions[qi]['name']} non-finite step={step}")

        # ---- Event triggers (compare against snapped t) -------------
        do_save  = (save_idx  < save_times.size)  and (t >= save_times[save_idx]  - EPS_T)
        do_vtk   = (vtk_idx   < vtk_times.size)   and (t >= vtk_times[vtk_idx]   - EPS_T)
        do_print = (print_idx < print_times.size) and (t >= print_times[print_idx] - EPS_T)
        do_step_print = (step % 200 == 0) and not do_print   # heartbeat
        do_final = (t >= t_final - EPS_T)

        if do_print or do_step_print or do_final:
            cache.update_all(C, nprot_list, prot_mob_list=prot_mob_list)
            update_gates()
            osm_cyto, osm_bulk = osm_cyto_bulk_mM()
            Qtot = total_charge_C()
            vmnow = Vm() * 1e3
            elapsed = time.perf_counter() - t0_wall
            frac = t / t_final
            eta_t = elapsed * (1.0 - frac) / max(frac, 1e-6)
            # phi range away from membrane (membrane interior can spike harmlessly)
            phi_c_min = float(cp.min(cp.where(cyto_mask, phi,  cp.inf)).get())
            phi_c_max = float(cp.max(cp.where(cyto_mask, phi, -cp.inf)).get())
            phi_b_min = float(cp.min(cp.where(bulk_mask, phi,  cp.inf)).get())
            phi_b_max = float(cp.max(cp.where(bulk_mask, phi, -cp.inf)).get())
            ion_short = " ".join(
                f"{sp['name']}J={ion_flux_now[i]:+.2e}" for i, sp in enumerate(ions))
            # Stable total estimate: rate = step / elapsed_sim_time, scale to t_final
            est_total = max(step, int(round(step * t_final / max(t, EPS_T))))
            print(
                f"  {step:>7d}/{est_total:<7d}  t={t*1e9:7.2f}/{t_final*1e9:.0f} ns  "
                f"dt_cur={dt_cur:.2e}  Vm={vmnow:+.2f} mV  "
                f"phi_c=[{phi_c_min*1e3:+.1f},{phi_c_max*1e3:+.1f}] "
                f"phi_b=[{phi_b_min*1e3:+.1f},{phi_b_max*1e3:+.1f}] mV  "
                f"np={len(pores)}  "
                f"osm c/b={osm_cyto:.2f}/{osm_bulk:.2f} mM  "
                f"Q={Qtot:+.1e} I={I_z0_A:+.3e} {ion_short}  "
                f"| wall {elapsed:.0f}s ETA {eta_t:.0f}s PCG={info}",
                flush=True,
            )
            row = [step, frac, t, t*1e9, vmnow, len(pores),
                   osm_cyto, osm_bulk]
            for pi in range(len(proteins)):
                row.extend([protein_flux_now[pi],
                            protein_leakage_mol[pi],
                            protein_leakage_mol[pi] * NA])
            row.extend([Qtot, I_z0_A, info, dt_step])
            for qi in range(len(ions)):
                row.extend([ion_flux_now[qi], ion_efflux_z0_mol[qi]])
            log_rows.append(row)
            _append_csv_row(row) 
            # ---- Convergence check: each ion species must be quiet ----
            if conv_enable and t >= conv_t_min and conv_idx:
                # |cumulative efflux| per species [mol]
                cur_vec = np.array(
                    [abs(ion_efflux_z0_mol[i]) for i in conv_idx],
                    dtype=np.float64,
                )
                if prev_efflux_vec is not None and prev_efflux_t is not None:
                    dt_obs = t - prev_efflux_t
                    if dt_obs > 0:
                        rates = (cur_vec - prev_efflux_vec) / dt_obs   # per species
                        max_rate = float(np.max(np.abs(rates)))
                        tot_abs  = float(np.sum(cur_vec))
                        worst_i  = int(np.argmax(np.abs(rates)))
                        worst_nm = ions[conv_idx[worst_i]]["name"]
                        is_quiet = (max_rate < conv_tol) and (tot_abs >= conv_emin)
                        if is_quiet:
                            quiet_streak += 1
                        else:
                            quiet_streak = 0
                        L(f"    [conv] max|rate|={max_rate:.3e} mol/s "
                          f"(worst={worst_nm})  Σ|efflux|={tot_abs:.3e} mol  "
                          f"quiet={quiet_streak}/{conv_N}")
                        if quiet_streak >= conv_N and not equilibrium_notified:
                            equilibrium_notified = True
                            L(f"  NOTE: equilibrium likely reached at t={t*1e9:.3f} ns "
                              f"(max|rate|={max_rate:.3e}<{conv_tol:.3e}, worst={worst_nm}). "
                              f"Early stop is DISABLED; simulation continues.")
                prev_efflux_vec = cur_vec
                prev_efflux_t   = t
            if do_print:
                print_idx += 1

        if do_save:
            timer.tic("save_npz")
            save_npz_roi(step, t)
            timer.toc("save_npz")
            save_idx += 1

        if do_vtk:
            timer.tic("save_vtk")
            save_vtr_full(step)
            timer.toc("save_vtk")
            vtk_idx += 1
        
    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------
    write_trajectory_csv(paths["csv_path"], csv_header, log_rows)

    state.C = C
    state.nprot_list = nprot_list
    state.phi = phi
    state.rw = rw

    save_params_json(state, cfg, paths)

    L("")
    L(f"  csv : {paths['csv_path']}")
    L(f"  vtk : {paths['vtk_dir']}")
    L(f"  npz : {paths['npz_dir']}")
