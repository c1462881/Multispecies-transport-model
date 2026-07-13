"""Shared first-order relaxation fitting for timescale diagnostics.
Imported by both main.py (stdout table) and visualization.py."""
import numpy as np
from scipy.optimize import curve_fit

from initialization import (ION_SPECIES, ETA_W, ETA_M, D_MEM,
                            FRAC_FAR, SIM_L, D_p_M, D_p_w, )


def relax(t, Cinf, dC, tau):
    """C(t) = Cinf - (Cinf - C0) exp(-t/tau)."""
    return Cinf - dC*np.exp(-t/tau)


def fit_tau(tt, yy):
    """Fit first-order relaxation with C_inf free.
    ok=True only if the trace has genuine plateau-curvature:
    good R^2, positive tau, tau not pinned to the window edge."""
    C0 = yy[0]; span = yy[-1] - yy[0]
    if abs(span) < 1e-30:
        return dict(tau=np.nan, Cinf=yy[-1], r2=0.0, ok=False)
    guess = (yy[-1] + 0.5*span, yy[-1] - yy[0], 0.3*tt[-1])
    try:
        p, _ = curve_fit(relax, tt, yy, p0=guess, maxfev=40000)
        Cinf, dC, tau = p
        resid = yy - relax(tt, *p)
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((yy - yy.mean())**2) + 1e-30
        r2 = 1.0 - ss_res/ss_tot
        ok = (r2 > 0.98) and (tau > 0) and (tau < 5.0*tt[-1])
        return dict(tau=tau, Cinf=Cinf, r2=r2, ok=bool(ok))
    except Exception:
        return dict(tau=np.nan, Cinf=np.nan, r2=0.0, ok=False)


def _theory_times():
    """Extracellular bulk (l^2/D) and membrane-crossing timescales [ns]."""
    L_mon = FRAC_FAR * SIM_L * 1e-9
    tau_mem, tau_bulk = {}, {}
    for s in ION_SPECIES:
        D_mem_i = s['D0'] * (ETA_W/ETA_M)
        tau_mem[s['name']]  = L_mon * D_MEM / (2*D_mem_i) * 1e9
        tau_bulk[s['name']] = L_mon**2 / s['D0'] * 1e9
    tau_mem['Protein']  = L_mon * D_MEM / (2*D_p_M) * 1e9
    tau_bulk['Protein'] = L_mon**2 / D_p_w * 1e9
    return tau_mem, tau_bulk


def build_diagnostics(t_ns, ion_hist, p_hist, li_no, colors):
    tau_mem, tau_bulk = _theory_times()
    diags = []
    for k, s in enumerate(ION_SPECIES):
        sig = ion_hist[k][:, li_no]
        fit = fit_tau(t_ns, sig)
        diags.append(dict(label=s['name'], color=colors[s['name']],
                          t=t_ns, sig=sig,
                          tau=fit['tau'], Cinf=fit['Cinf'],
                          r2=fit['r2'], ok=fit['ok'],
                          tau_mem=tau_mem[s['name']],
                          tau_bulk=tau_bulk[s['name']]))
    if p_hist is not None:
        sig = p_hist[:, li_no]
        fit = fit_tau(t_ns, sig)
        diags.append(dict(label='Protein', color=colors['protein'],
                          t=t_ns, sig=sig,
                          tau=fit['tau'], Cinf=fit['Cinf'],
                          r2=fit['r2'], ok=fit['ok'],
                          tau_mem=tau_mem['Protein'],
                          tau_bulk=tau_bulk['Protein']))
    return diags


def print_summary(diags, t_window_ns):
    """Log the fit table to stdout (called from main.py)."""
    print("\n" + "="*72)
    print("  TIMESCALE FIT SUMMARY  (extracellular near_out, first-order relaxation)")
    print("="*72)
    print(f"  {'species':<9}{'tau[ns]':>12}{'C_inf':>12}"
          f"{'R^2':>9}{'ok':>5}   note")
    print("  " + "-"*68)
    for d in diags:
        if d['ok'] and np.isfinite(d['tau']):
            tau_s = f"{d['tau']:.1f}"; note = ""
        else:
            tau_s = f">={t_window_ns:.0f}"
            note = "lower bound (unconverged / accelerating)"
        cinf = f"{d['Cinf']:.4g}" if np.isfinite(d['Cinf']) else "--"
        print(f"  {d['label']:<9}{tau_s:>12}{cinf:>12}"
              f"{d['r2']:>9.4f}{str(d['ok']):>5}   {note}")
    print("="*72 + "\n")