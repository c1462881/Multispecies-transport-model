import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

from configuration import *
from initialization import (
    z, N, dz, mem_mask, cyto_mask, bulk_mask, lambda_D,
    N_ions, N_aux, C_don_cyto, C_don_bulk, dphi_don,
    Pi_init_cyto, Pi_init_bulk, Pi_don_cyto, Pi_don_bulk,
    C_PROTEIN_CYTO,
)
from solver import get_phi

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'legend.fontsize': 7, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

D_NM  = D_HALF * 1e9
L_DNM = lambda_D * 1e9

# ---- temporal "in-side" colours ----
TEMPORAL_IN_COLOR = {
    'phi': 'purple',
    'Pi':  'seagreen',
    'protein': SPATIAL_COLORS['protein'],
}

def _style(ax):
    if ax.get_xscale() != 'log': ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ax.get_yscale() != 'log': ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='minor', length=2); ax.tick_params(which='major', length=4)

def _shade_mem(ax, debye=True):
    ax.axvspan(-D_NM, D_NM, color='slategray', alpha=0.25, zorder=1)
    if debye:
        for s_ in [-1, 1]:
            lo, hi = sorted([s_*D_NM, s_*(D_NM+L_DNM)])
            ax.axvspan(lo, hi, color='gold', alpha=0.20, zorder=0)

def _init_profile(v_c, v_b, z_arr):
    p = np.empty_like(z_arr)
    p[z_arr < -D_HALF] = v_c
    p[z_arr >  D_HALF] = v_b
    p[np.abs(z_arr) <= D_HALF] = 0.5*(v_c+v_b)
    return p

def _don_seg(ax, val, side, **kw):
    if side == 'cyto':
        ax.plot([-SIM_L, -D_NM], [val, val], **kw)
    else:  # bulk / extracellular
        ax.plot([D_NM, SIM_L], [val, val], **kw)

# ================================================================
def _monitor_indices():
    box_mask = np.abs(z) <= SIM_L*1e-9
    box_idx  = np.where(box_mask)[0]
    L = SIM_L*1e-9
    idx_ni = np.argmin(np.abs(z - (-FRAC_NEAR*L)))
    idx_no = np.argmin(np.abs(z - ( FRAC_NEAR*L)))
    idx_fi = np.argmin(np.abs(z - (-FRAC_FAR *L)))
    idx_fo = np.argmin(np.abs(z - ( FRAC_FAR *L)))
    return box_idx, idx_ni, idx_no, idx_fi, idx_fo

# ================================================================
def plot_all(sol):
    box_idx, idx_ni, idx_no, idx_fi, idx_fo = _monitor_indices()
    z_box = z[box_idx]; z_nm = z_box*1e9
    t_ns  = sol.t*1e9
    to_l  = {gi: li for li, gi in enumerate(box_idx)}
    li_ni, li_no = to_l[idx_ni], to_l[idx_no]
    li_fi, li_fo = to_l[idx_fi], to_l[idx_fo]

    ion_hist = [sol.y[k*N + box_idx, :].T for k in range(N_ions)]
    p_hist   = sol.y[N_ions*N + box_idx, :].T
    n_t = len(t_ns)
    phi_hist = np.zeros((n_t, len(box_idx)))
    Pi_hist  = np.zeros((n_t, len(box_idx)))

    for i in range(n_t):
        phi_hist[i] = get_phi(sol.y[:, i])[box_idx] * 1e3
        Csum = sum(sol.y[k*N + box_idx, i] for k in range(N_ions))
        Cp   = sol.y[N_ions*N + box_idx, i]
        base = (N_ions+1)*N
        Caux = sum(sol.y[base+j*N + box_idx, i] for j in range(N_aux))
        Pi_hist[i] = R_GAS*T_K*(Csum + Cp + Caux)/1e3

    from configuration import SNAP_FRACS, PLOT_PI_SNAPSHOTS
    snap_idx = [int(np.argmin(np.abs(t_ns - f*t_ns[-1]))) for f in SNAP_FRACS]

    # ============ FIGURE 1: FINAL SPATIAL PROFILES ============
    fig1, (ax_ion, ax_p, ax_b, ax_c) = plt.subplots(
        1, 4, figsize=(22, 5.4), gridspec_kw={'wspace': 0.45})
    ax_ca = ax_ion.twinx()
    for ax in (ax_ion, ax_p, ax_b, ax_c):
        _shade_mem(ax); ax.set_xlim(-SIM_L, SIM_L); _style(ax)
        ax.set_box_aspect(1)                 

    name2k = {s['name']: k for k, s in enumerate(ION_SPECIES)}
    monov  = [s for s in ION_SPECIES if abs(s['z']) == 1]
    divv   = [s for s in ION_SPECIES if abs(s['z']) == 2]

    # --- Panel 1: ions (Na,K,Cl left ; Ca right) ---
    for s in monov:
        k = name2k[s['name']]; col = SPATIAL_COLORS[s['name']]
        ax_ion.plot(z_nm, ion_hist[k][-1], color=col, lw=1.8, label=s['name'])
        ax_ion.plot(z_nm, _init_profile(s['C_cyto'], s['C_bulk'], z_box),
                    color=col, lw=0.9, alpha=0.5)
        _don_seg(ax_ion, C_don_cyto[k], 'cyto', color=col, ls='--', lw=0.8)
        _don_seg(ax_ion, C_don_bulk[k], 'bulk', color=col, ls=':',  lw=0.8)
    ax_ion.set_ylabel(r'$C_{\mathrm{Na,K,Cl}}$ [mM]')
    ax_ion.set_xlabel('z [nm]'); ax_ion.set_title('Ion concentrations')
    for s in divv:
        k = name2k[s['name']]; col = SPATIAL_COLORS[s['name']]
        ax_ca.plot(z_nm, ion_hist[k][-1], color=col, lw=1.8, label=s['name'])
        ax_ca.plot(z_nm, _init_profile(s['C_cyto'], s['C_bulk'], z_box),
                   color=col, lw=0.9, alpha=0.5)
        _don_seg(ax_ca, C_don_cyto[k], 'cyto', color=col, ls='--', lw=0.8)
        _don_seg(ax_ca, C_don_bulk[k], 'bulk', color=col, ls=':',  lw=0.8)
    ax_ca.set_ylabel(r'$C_{\mathrm{Ca}}$ [mM]', color=SPATIAL_COLORS['Ca2+'])
    ax_ca.tick_params(axis='y', colors=SPATIAL_COLORS['Ca2+'])
    ax_ca.set_box_aspect(1)

    from matplotlib.lines import Line2D
    # colour legend (species) — one entry per species, solid swatch
    species_handles = [Line2D([0],[0], color=SPATIAL_COLORS[s['name']], lw=1.8,
                              label=s['name']) for s in ION_SPECIES]
    leg1 = ax_ion.legend(handles=species_handles, fontsize=6, ncol=2,
                         loc='upper left', framealpha=0.9, title='species')

    # style legend (meaning) — black proxies: bold=final, thin=initial, dashed=donnan
    style_handles = [
        Line2D([0],[0], color='k', lw=1.8, ls='-',  label='final'),
        Line2D([0],[0], color='k', lw=0.9, ls='-',  label='initial'),
        Line2D([0],[0], color='k', lw=0.9, ls='--', label='Donnan (cyto)'),
        Line2D([0],[0], color='k', lw=0.9, ls=':',  label='Donnan (bulk)'),
    ]
    ax_ion.legend(handles=style_handles, fontsize=6, ncol=1,
                  loc='upper right', framealpha=0.9, title='line style')
    ax_ion.add_artist(leg1)                     # keep BOTH legends

    # --- Panel 2: protein ---
    col = SPATIAL_COLORS['protein']
    ax_p.plot(z_nm, p_hist[-1], color=col, lw=1.8, label=r'$C_p$ final')
    ax_p.plot(z_nm, _init_profile(C_PROTEIN_CYTO, 0.0, z_box),
              color=col, lw=0.9, alpha=0.6, label='Initial')
    ax_p.set_ylabel(r'$C_p$ [mM]'); ax_p.set_xlabel('z [nm]')
    ax_p.set_title('Protein'); ax_p.legend(fontsize=6, loc='best', framealpha=0.9)

    # --- Panel 3: potential ---
    ax_b.plot(z_nm, phi_hist[-1], color=SPATIAL_COLORS['phi'], lw=1.8, label='φ final')
    ax_b.plot(z_nm, get_phi(sol.y[:,0])[box_idx]*1e3,
              color=SPATIAL_COLORS['phi'], lw=1.0, alpha=0.6, label='Initial')
    _don_seg(ax_b, dphi_don*1e3, 'cyto', color='gray', ls='--', lw=1.0,
             label=f'Donnan cyto = {dphi_don*1e3:.2f}')
    _don_seg(ax_b, 0.0, 'bulk', color='gray', ls=':', lw=1.0, label='Donnan bulk = 0')
    ax_b.set_xlabel('z [nm]'); ax_b.set_ylabel('φ [mV]')
    ax_b.set_title('Electric potential'); ax_b.legend(fontsize=6.5, framealpha=0.9)

    # --- Panel 4: osmolarity (optional all-timepoint overlay) ---
    if PLOT_PI_SNAPSHOTS:
        cols = plt.cm.viridis(np.linspace(0, 0.9, len(snap_idx)))
        for c, ti in zip(cols, snap_idx):
            ax_c.plot(z_nm, Pi_hist[ti], color=c, lw=1.3,
                      label=f't = {t_ns[ti]:.3g} ns')
    else:
        ax_c.plot(z_nm, Pi_hist[-1], color=SPATIAL_COLORS['Pi'], lw=1.8, label='Π final')
        ax_c.plot(z_nm, _init_profile(Pi_init_cyto, Pi_init_bulk, z_box),
                  color=SPATIAL_COLORS['Pi'], lw=1.0, alpha=0.6, label='Initial')
    _don_seg(ax_c, Pi_don_cyto, 'cyto', color='gray', ls='--', lw=1.0)
    _don_seg(ax_c, Pi_don_bulk, 'bulk', color='gray', ls=':',  lw=1.0)
    ax_c.set_xlabel('z [nm]'); ax_c.set_ylabel('Π [kPa]')
    ax_c.set_title('Osmotic pressure'); ax_c.legend(fontsize=6, framealpha=0.9)

    fig1.suptitle(fr'Final Spatial Profiles  ($t = {SIM_T}$ ns)', y=1.02)
    for fmt in FORMATS:
        fig1.savefig(OUTPUT_DIR/f'Spatial_final.{fmt}', bbox_inches='tight', dpi=300)

    # ============ FIGURE 2: ALL TEMPORAL PROFILES ============
    t_plot = t_ns[1:]                            # drop t=0
    p_ni, p_no = p_hist[1:, li_ni], p_hist[1:, li_no]
    p_fi, p_fo = p_hist[1:, li_fi], p_hist[1:, li_fo]
    cyto_max = max(p_ni.max(), p_fi.max())
    bulk_max = max(p_no.max(), p_fo.max())
    protein_needs_break = (max(cyto_max, bulk_max) >
                            20*max(min(cyto_max, bulk_max), 1e-30))
    n_prot_rows = 2 if protein_needs_break else 1
    total_rows = N_ions + n_prot_rows + 2

    fig2 = plt.figure(figsize=(11, 2.2*total_rows))
    gs2 = gridspec.GridSpec(total_rows, 1, figure=fig2, hspace=0.35)

    def _plot_ion(ax, k, s):
        col = SPATIAL_COLORS[s['name']]             
        y = ion_hist[k]
        ax.plot(t_plot, y[1:, li_fi], color=col,  ls=':', lw=1.5, label=f"far_in ({z_nm[li_fi]:.1f} nm)")
        ax.plot(t_plot, y[1:, li_ni], color=col,  ls='-', lw=1.8, label=f"near_in ({z_nm[li_ni]:.1f} nm)")
        ax.plot(t_plot, y[1:, li_fo], color='navy', ls=':', lw=1.5, label=f"far_out ({z_nm[li_fo]:.1f} nm)")
        ax.plot(t_plot, y[1:, li_no], color='navy', ls='-', lw=1.8, label=f"near_out ({z_nm[li_no]:.1f} nm)")
        _don_seg_time_c = C_don_cyto[k]; _don_seg_time_b = C_don_bulk[k]
        ax.axhline(_don_seg_time_c, color='gray', ls='-', lw=1.0, label=f"Donnan cyto={_don_seg_time_c:.2f}")
        ax.axhline(_don_seg_time_b, color='gray', ls=':', lw=1.0, label=f"Donnan bulk={_don_seg_time_b:.2f}")
        ax.set_ylabel(f"{s['name']} [mM]")
        ax.legend(fontsize=5.5, ncol=3, loc='best', framealpha=0.9)
        _style(ax)

    row = 0
    for k, s in enumerate(ION_SPECIES):
        ax = fig2.add_subplot(gs2[row]); _plot_ion(ax, k, s); row += 1
    pcol = TEMPORAL_IN_COLOR['protein']
    if protein_needs_break:
        cyto_side_larger = cyto_max >= bulk_max
        ax_top = fig2.add_subplot(gs2[row]);       row += 1
        ax_bot = fig2.add_subplot(gs2[row]);       row += 1
        for a in (ax_top, ax_bot):
            a.plot(t_plot, p_fi, color=pcol,  ls=':', lw=1.5, label='far_in')
            a.plot(t_plot, p_ni, color=pcol,  ls='-', lw=1.8, label='near_in')
            a.plot(t_plot, p_fo, color='navy', ls=':', lw=1.5, label='far_out')
            a.plot(t_plot, p_no, color='navy', ls='-', lw=1.8, label='near_out')
            _style(a)
        if cyto_side_larger:
            ax_top.set_ylim(0.9*min(p_ni.min(), p_fi.min()), 1.05*cyto_max)
            ax_bot.set_ylim(0.0, 1.5*bulk_max + 1e-30)
        else:
            ax_top.set_ylim(0.9*min(p_no.min(), p_fo.min()), 1.05*bulk_max)
            ax_bot.set_ylim(0.0, 1.5*cyto_max + 1e-30)
        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_top.tick_params(labelbottom=False, bottom=False)
        d_ = .015
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, lw=1)
        ax_top.plot((-d_, +d_), (-d_, +d_), **kwargs)
        ax_top.plot((1-d_, 1+d_), (-d_, +d_), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d_, +d_), (1-d_, 1+d_), **kwargs)
        ax_bot.plot((1-d_, 1+d_), (1-d_, 1+d_), **kwargs)
        ax_top.set_ylabel('')
        ax_bot.set_ylabel('Protein [mM]')
        ax_top.legend(fontsize=5.5, ncol=4, loc='best', framealpha=0.9)
    else:
        ax = fig2.add_subplot(gs2[row]); row += 1
        ax.plot(t_plot, p_fi, color=pcol,  ls=':', lw=1.5, label='far_in')
        ax.plot(t_plot, p_ni, color=pcol,  ls='-', lw=1.8, label='near_in')
        ax.plot(t_plot, p_fo, color='navy', ls=':', lw=1.5, label='far_out')
        ax.plot(t_plot, p_no, color='navy', ls='-', lw=1.8, label='near_out')
        ax.set_ylabel('Protein [mM]')
        ax.legend(fontsize=5.5, ncol=4, loc='best', framealpha=0.9); _style(ax)

    pcol = TEMPORAL_IN_COLOR['phi']
    ax = fig2.add_subplot(gs2[row]); row += 1
    ax.plot(t_plot, phi_hist[1:, li_fi], color=pcol,  ls=':', lw=1.5, label='far_in')
    ax.plot(t_plot, phi_hist[1:, li_ni], color=pcol,  ls='-', lw=1.8, label='near_in')
    ax.plot(t_plot, phi_hist[1:, li_fo], color='navy', ls=':', lw=1.5, label='far_out')
    ax.plot(t_plot, phi_hist[1:, li_no], color='navy', ls='-', lw=1.8, label='near_out')
    ax.axhline(dphi_don*1e3, color='gray', ls='-', lw=1.0, label=f'Donnan cyto={dphi_don*1e3:.2f}')
    ax.axhline(0.0,          color='gray', ls=':', lw=1.0, label='Donnan bulk=0')
    ax.set_ylabel('φ [mV]')
    ax.legend(fontsize=5.5, ncol=3, loc='best', framealpha=0.9); _style(ax)

    pcol = TEMPORAL_IN_COLOR['Pi']
    ax = fig2.add_subplot(gs2[row]); row += 1
    ax.plot(t_plot, Pi_hist[1:, li_fi], color=pcol,  ls=':', lw=1.5, label='far_in')
    ax.plot(t_plot, Pi_hist[1:, li_ni], color=pcol,  ls='-', lw=1.8, label='near_in')
    ax.plot(t_plot, Pi_hist[1:, li_fo], color='navy', ls=':', lw=1.5, label='far_out')
    ax.plot(t_plot, Pi_hist[1:, li_no], color='navy', ls='-', lw=1.8, label='near_out')
    ax.axhline(Pi_don_cyto, color='gray', ls='-', lw=1.0, label=f'Donnan cyto={Pi_don_cyto:.1f}')
    ax.axhline(Pi_don_bulk, color='gray', ls=':', lw=1.0, label=f'Donnan bulk={Pi_don_bulk:.1f}')
    ax.set_xlabel('Time [ns]'); ax.set_ylabel('Π [kPa]')
    ax.legend(fontsize=5.5, ncol=3, loc='best', framealpha=0.9); _style(ax)

    fig2.suptitle('Temporal Profiles at Monitoring Points', y=1.002)
    for fmt in FORMATS:
        fig2.savefig(OUTPUT_DIR/f'Temporal_all.{fmt}', bbox_inches='tight', dpi=300)

    # ============ FIGURE 3: DONNAN CONVERGENCE ============
    fig3 = plt.figure(figsize=(15, 8))
    ncols = 2; nrows = int(np.ceil(N_ions/ncols))
    gs3 = gridspec.GridSpec(nrows+1, ncols, figure=fig3, hspace=0.35, wspace=0.35)
    for k, s in enumerate(ION_SPECIES):
        ax = fig3.add_subplot(gs3[k//ncols, k%ncols])
        col = SPATIAL_COLORS[s['name']]
        ax.plot(t_ns, ion_hist[k][:, li_ni]-C_don_cyto[k], color=col, ls='-', lw=1.5, label='near_in - Don_cyto')
        ax.plot(t_ns, ion_hist[k][:, li_fi]-C_don_cyto[k], color=col, ls=':', lw=1.5, label='far_in - Don_cyto')
        ax.plot(t_ns, ion_hist[k][:, li_no]-C_don_bulk[k], color='navy', ls='-', lw=1.5, label='near_out - Don_bulk')
        ax.plot(t_ns, ion_hist[k][:, li_fo]-C_don_bulk[k], color='navy', ls=':', lw=1.5, label='far_out - Don_bulk')
        ax.axhline(0.0, color='gray', lw=0.6)
        ax.set_xlabel('Time [ns]'); ax.set_ylabel(r'$C - C_{\mathrm{don}}$ [mM]')
        ax.set_title(s['name']); ax.legend(fontsize=6, framealpha=0.9); _style(ax)

    ax = fig3.add_subplot(gs3[nrows, 0])
    ax.semilogy(t_ns, np.clip(np.abs(phi_hist[:, li_ni]-dphi_don*1e3), 1e-10, None),
                color=SPATIAL_COLORS['phi'], ls='-', lw=1.5, label='near_in')
    ax.semilogy(t_ns, np.clip(np.abs(phi_hist[:, li_fi]-dphi_don*1e3), 1e-10, None),
                color=SPATIAL_COLORS['phi'], ls=':', lw=1.5, label='far_in')
    ax.set_xlabel('Time [ns]'); ax.set_ylabel(r'$|\phi - \phi_{\mathrm{don,cyto}}|$ [mV]')
    ax.set_title('Potential'); ax.legend(fontsize=7); _style(ax)

    ax = fig3.add_subplot(gs3[nrows, 1])
    ax.semilogy(t_ns, np.clip(np.abs(Pi_hist[:, li_ni]-Pi_don_cyto), 1e-10, None),
                color=SPATIAL_COLORS['Pi'], ls='-', lw=1.5, label='near_in')
    ax.semilogy(t_ns, np.clip(np.abs(Pi_hist[:, li_fi]-Pi_don_cyto), 1e-10, None),
                color=SPATIAL_COLORS['Pi'], ls=':', lw=1.5, label='far_in')
    ax.set_xlabel('Time [ns]'); ax.set_ylabel(r'$|\Pi - \Pi_{\mathrm{don,cyto}}|$ [kPa]')
    ax.set_title('Osmotic Pressure'); ax.legend(fontsize=7); _style(ax)

    fig3.suptitle('Convergence to Donnan Reference', y=0.98)
    for fmt in FORMATS:
        fig3.savefig(OUTPUT_DIR/f'Donnan_convergence.{fmt}', bbox_inches='tight', dpi=300)

    # ============ FIGURE 5: TIMESCALE SEPARATION (extracellular, near_out) ============
    from timescale_fit import relax, build_diagnostics

    tau_D_th = lambda_D**2 / max(s['D0'] for s in ION_SPECIES)

    p_hist_arg = p_hist 
    diagnostics = build_diagnostics(t_ns, ion_hist, p_hist_arg, li_no, SPATIAL_COLORS)
    tau_lb = t_ns[-1]

    fig5, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.5),
                                    gridspec_kw={'wspace': 0.32,
                                                 'width_ratios': [1.0, 1.15]})

    # ---------- PANEL A: tau bar chart (linear) ----------
    labels = [d['label'] for d in diagnostics]
    ypos   = np.arange(len(labels))
    fit_taus = [d['tau'] for d in diagnostics if d['ok'] and d['tau'] > 0]
    ths = [v for d in diagnostics for v in (d['tau_mem'], d['tau_bulk'])
           if np.isfinite(v) and v > 0]
    tau_floor = min(fit_taus + [tau_lb] + ths + [tau_D_th*1e9]) * 0.3
    tau_ceil  = max(fit_taus + [tau_lb] + ths + [tau_D_th*1e9]) * 3.0

    for i, d in enumerate(diagnostics):
        if d['ok'] and d['tau'] > 0:
            axA.barh(i, d['tau']-tau_floor, left=tau_floor, height=0.55,
                     color=d['color'], edgecolor='k', lw=0.6, alpha=0.85, zorder=2)
            axA.text(d['tau'], i, f"  {d['tau']:.0f} ns (R²={d['r2']:.3f})",
                     fontsize=6.5, va='center', ha='left', zorder=4)
        else:
            # unconverged: hatched lower-bound bar + arrow to the right
            axA.barh(i, tau_lb-tau_floor, left=tau_floor, height=0.55,
                     color=d['color'], edgecolor='k', lw=0.6, alpha=0.45,
                     hatch='///', zorder=2)
            axA.annotate('', xy=(tau_lb*2.2, i), xytext=(tau_lb, i),
                         arrowprops=dict(arrowstyle='-|>', color=d['color'], lw=1.6),
                         zorder=4)
            axA.text(tau_lb, i-0.32, f"  ≥{tau_lb:.0f} ns (lower bound)",
                     fontsize=6.5, va='center', ha='left', zorder=4)
    for i, d in enumerate(diagnostics):
        axA.plot(d['tau_bulk'], i, marker='o', color='dimgray', markersize=9,
                 markerfacecolor='none', markeredgewidth=1.4,
                 label=r'$\tau_{\mathrm{bulk}}=\ell^2/D_{\mathrm{ext}}$' if i == 0 else None,
                 zorder=5)
        axA.plot(d['tau_mem'], i, marker='D', color='k', markersize=7,
                 markerfacecolor='none', markeredgewidth=1.2,
                 label=r'$\tau_{\mathrm{mem}}$' if i == 0 else None, zorder=5)
    axA.axvline(tau_D_th*1e9, color='crimson', ls='--', lw=1.1, alpha=0.7,
                label=fr'$\tau_D={tau_D_th*1e9:.2f}$ ns')
    # axA.set_xscale('linear'); axA.set_xlim(tau_floor, tau_ceil)
    axA.set_xscale('linear'); axA.set_xlim(0.0, 20000.0)
    axA.set_yticks(ypos); axA.set_yticklabels(labels)
    axA.set_ylim(-0.6, len(labels)-0.4); axA.invert_yaxis()
    axA.set_xlabel('Characteristic time τ [ns]')
    axA.yaxis.set_ticks_position('none')
    axA.legend(fontsize=6.5, framealpha=0.9, loc='lower right')
    axA.set_title('First-order relaxation τ (fit) vs theory', fontsize=9)
    _style(axA)

    # ---------- PANEL B: normalized relaxation curves ----------
    for d in diagnostics:
        y = d['sig']
        C0 = y[0]
        Cinf = d['Cinf'] if (d['ok'] and np.isfinite(d['Cinf'])) else y[-1]
        denom = Cinf - C0
        if abs(denom) < 1e-30:
            continue
        u = (y - C0) / denom
        axB.plot(t_ns, np.clip(u, -0.05, 1.25), color=d['color'], lw=1.8, label=d['label'])
        if d['ok'] and d['tau'] > 0:
            u_fit = (relax(t_ns, d['Cinf'], d['Cinf']-C0, d['tau']) - C0)/denom
            axB.plot(t_ns[1:], u_fit[1:], color=d['color'], ls='--', lw=1.0, alpha=0.8)
            axB.plot(d['tau'], 1.0-1.0/np.e, marker='o', color=d['color'],
                     markersize=6, markeredgecolor='k', markeredgewidth=0.6, zorder=5)
    axB.axhline(1.0-1.0/np.e, color='gray', ls=':', lw=1.0)
    axB.text(0.02*t_ns[-1], 1.0-1.0/np.e+0.02, '63% (τ)', fontsize=7, color='gray')
    axB.set_xscale('linear')
    axB.set_xlim(0.0, t_ns[-1])
    axB.set_ylim(-0.05, 1.15)
    axB.set_xlabel('Time [ns]')
    axB.set_ylabel(r'$u(t)=(C-C_0)/(C_\infty-C_0)$')
    axB.set_title('Normalized relaxation (dashed = exponential fit)', fontsize=9)
    axB.legend(fontsize=7, framealpha=0.9, loc='center right')
    _style(axB)

    fig5.suptitle('Timescale Separation (extracellular side, near_out)', y=1.005)
    for fmt in FORMATS:
        fig5.savefig(OUTPUT_DIR/f'Timescales_difference.{fmt}',
                     bbox_inches='tight', dpi=300)

    plt.show()
    return {
        't_ns': t_ns, 'z_nm': z_nm,
        'ion_hist': ion_hist, 'p_hist': p_hist,
        'phi_hist': phi_hist, 'Pi_hist': Pi_hist,
        'monitor': dict(li_ni=li_ni, li_no=li_no, li_fi=li_fi, li_fo=li_fo),
    }


