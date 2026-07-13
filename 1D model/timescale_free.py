# coding=utf-8
"""Usage:  python timescale_free.py  <path_to_run_folder>
Reads Temporal_datapoints.csv (near_out columns) and reports, per species:
  t10/t50/t90  (fractional-crossing times toward the MEASURED end-state)
  tau_int      (integral/spectral relaxation time, exact = tau for 1-exp)
Produces Timescales_free.pdf/svg with a linear time axis.
"""
import sys
# coding=utf-8
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from configuration import SPATIAL_COLORS, ION_SPECIES, FORMATS

def _cross(t, u, level):
    """First time u(t) crosses `level` (linear interp). NaN if never."""
    idx = np.where(u >= level)[0]
    if idx.size == 0 or idx[0] == 0:
        return np.nan, bool(idx.size)
    i = idx[0]
    return t[i-1] + (level-u[i-1])/(u[i]-u[i-1])*(t[i]-t[i-1]), True

def metrics(t, y):
    C0, Cend = y[0], y[-1]; span = Cend - C0
    out = dict(t10=np.nan, t50=np.nan, t90=np.nan, tau_int=np.nan,
               C0=C0, Cend=Cend, reached90=False, span=span)
    if abs(span) < 1e-30:
        return out
    u = (y - C0)/span
    out['t10'], _              = _cross(t, u, 0.10)
    out['t50'], _              = _cross(t, u, 0.50)
    out['t90'], out['reached90'] = _cross(t, u, 0.90)
    out['tau_int'] = np.trapz(1.0 - u, t)      # spectral mean time
    return out

def main(folder):
    folder = Path(folder)
    df = pd.read_csv(folder/'Temporal_datapoints.csv')
    t  = df['Time_ns'].to_numpy()
    T  = t[-1]

    species = [s['name'] for s in ION_SPECIES] + ['protein']
    print("\n" + "="*82)
    print(f"  FREE MODELTIMESCALE SEPARATION  (near_out, window = {T:.4g} ns)")
    print("="*82)
    print(f"  {'species':<9}{'t10[ns]':>13}{'t50[ns]':>13}{'t90[ns]':>13}"
          f"{'tau_int[ns]':>15}   note")
    print("  " + "-"*78)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    res = {}
    for name in species:
        col_name = f"C_{name}_near_out_mM"
        if col_name not in df.columns:
            continue
        y = df[col_name].to_numpy()
        m = metrics(t, y); res[name] = m
        color = SPATIAL_COLORS.get(name, SPATIAL_COLORS.get('protein'))
        span = m['span']
        if abs(span) > 1e-30:
            u = (y - m['C0'])/span
            ax.plot(t, np.clip(u, -0.05, 1.15), color=color, lw=1.8, label=name)
            if np.isfinite(m['t50']):
                ax.plot(m['t50'], 0.5, 'o', color=color,
                        markeredgecolor='k', markeredgewidth=0.6, zorder=5)
        note = "" if m['reached90'] else f"90% NOT reached (>{T:.3g} ns)"
        fmt = lambda v: f"{v:.4g}" if np.isfinite(v) else f">{T:.3g}"
        print(f"  {name:<9}{fmt(m['t10']):>13}{fmt(m['t50']):>13}"
              f"{fmt(m['t90']):>13}{m['tau_int']:>15.4g}   {note}")
    print("="*82 + "\n")

    ax.axhline(0.5, color='gray', ls=':', lw=1.0)
    ax.axhline(0.9, color='gray', ls='--', lw=0.8)
    ax.set_xlim(0.0, T); ax.set_ylim(-0.05, 1.15)         
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel(r'$u(t)=(C-C_0)/(C_{\mathrm{end}}-C_0)$')
    ax.set_title('Model-free fractional times')
    ax.legend(fontsize=8, framealpha=0.9)
    for fmt in FORMATS:
        fig.savefig(folder/f'Timescales_free.{fmt}', bbox_inches='tight', dpi=300)
    plt.show()
    return res

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
