# coding=utf-8
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import pandas as pd
from configuration import *
from initialization import (
    lambda_D, X_cyto, X_bulk, Y_cyto, Y_bulk, Z_X, C_PROTEIN_CYTO,
    C_don_cyto, C_don_bulk, dphi_don,
    C_don_cyto_prot, dphi_don_prot,
    sigma_protein, sigma_X, sigma_total,
    Pi_init_cyto, Pi_init_bulk, Pi_don_cyto, Pi_don_bulk,
    N_ions, dz, _INIT_ITER, _INIT_VM,
)
import initialization as ini
from solver import run_simulation, get_phi
from visualization import plot_all

import sys
class _Tee:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.f.write(s)
    def flush(self):    self.stdout.flush();  self.f.flush()
sys.stdout = _Tee(OUTPUT_DIR / 'run_log.txt')

# ---------------- Diagnostics ----------------
print("="*64)
print(f"Domain ±{EXT_L} nm  (analysis ±{SIM_L} nm),  sim_t = {SIM_T} ns")
print(f"Debye length λ_D = {lambda_D*1e9:.3f} nm")
print(f"\nProtein set by cytosolic electroneutrality: C_p_cyto = {C_PROTEIN_CYTO:.4f} mM")
print("Initial profiles of mobile aux species X, Y (t=0 EN + osmolarity):")
print(f"  X (z={Z_X:+d}):  cyto = {X_cyto:.4f} mM,  bulk = {X_bulk:.4f} mM")
print(f"  Y (z= 0):     cyto = {Y_cyto:.4f} mM,  bulk = {Y_bulk:.4f} mM")
print(f"\nInitial osmolarity (should match exactly by construction of Y):")
print(f"  Π_cyto = {Pi_init_cyto:.4f} kPa,  Π_bulk = {Pi_init_bulk:.4f} kPa")

print(f"\nResting-potential IC (target V_rest = {V_REST_MV:.1f} mV):")
print(f"  Converged in {_INIT_ITER} iteration(s)")
print(f"  V_m(0)  = {_INIT_VM:.4f} mV   (tol = ±{V_REST_TOL_MV} mV)")

print("\nDonnan reference (reservoir / single-ratio, protein-driven):")
print(f"  trapped charge: sigma_protein = {sigma_protein:+.3f} mM  (X,W,Y mobile)")
print(f"  residual mobile charge at equil (sigma_X,W,Y) = {sigma_X:+.3f} mM")
for k, s in enumerate(ION_SPECIES):
    print(f"  {s['name']:5s}: cyto {s['C_cyto']:8.4f} → {C_don_cyto[k]:8.4f} mM"
          f"  (bulk fixed {C_don_bulk[k]:8.4f} mM)")
print(f"  Δφ_don = {dphi_don*1e3:8.4f} mV   (protein-only)")
print(f"  Π_don: cyto = {Pi_don_cyto:.3f} kPa, bulk = {Pi_don_bulk:.3f} kPa")

# ------------- Mass-conservation check of Donnan (theoretical) -------------
print("\nDonnan mass-conservation check (theoretical):")
for k, s in enumerate(ION_SPECIES):
    S_init = s['C_cyto'] + s['C_bulk']
    S_don  = C_don_cyto[k] + C_don_bulk[k]
    print(f"  {s['name']:5s}: S_init = {S_init:8.4f},  S_don = {S_don:8.4f},"
          f"  Δ = {S_don-S_init:+.2e}")

# ---------------- Report initial membrane potential ----------------
print("\n" + "="*64)
print(f">>> Initial membrane potential  V_m(0) = φ_cyto − φ_bulk = {_INIT_VM:.4f} mV")
print("="*64)

import pickle
import os

RELOAD_FROM_CHECKPOINT = False 
CHECKPOINT_PATH = OUTPUT_DIR / 'sol_checkpoint.pkl'

# ---------------- Run / Load ----------------
if RELOAD_FROM_CHECKPOINT and CHECKPOINT_PATH.exists():
    print(f"\n[!] Loading previous simulation data from {CHECKPOINT_PATH}...")
    with open(CHECKPOINT_PATH, 'rb') as f:
        sol = pickle.load(f)
    print("✓ Successfully loaded.")
else:
    print("\n[!] Starting 1D PNP Simulation...")
    sol = run_simulation()
    print(f"\n[!] Saving safety checkpoint to {CHECKPOINT_PATH}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CHECKPOINT_PATH, 'wb') as f:
        pickle.dump(sol, f)
    print("✓ Checkpoint saved securely.")
    
# ---------------- Numerical mass-conservation check ----------------
print("\nNumerical mass conservation (∫C dz over full domain, ion + protein):")
for k, s in enumerate(ION_SPECIES):
    m0 = np.trapz(sol.y[k*ini.N:(k+1)*ini.N, 0],  dx=dz)
    mF = np.trapz(sol.y[k*ini.N:(k+1)*ini.N, -1], dx=dz)
    rel = (mF-m0)/max(abs(m0), 1e-30)
    flag = "✓" if abs(rel) < 1e-3 else "⚠"
    print(f"  {s['name']:5s}: m0 = {m0:10.4e},  mF = {mF:10.4e},  Δrel = {rel:+.2e}  {flag}")
psl = slice(N_ions*ini.N, (N_ions+1)*ini.N)
m0p, mFp = np.trapz(sol.y[psl,0],dx=dz), np.trapz(sol.y[psl,-1],dx=dz)
relp = (mFp-m0p)/max(abs(m0p),1e-30)
print(f"  Protein: m0 = {m0p:10.4e}, mF = {mFp:10.4e}, Δrel = {relp:+.2e} "
      f"{'✓' if abs(relp)<1e-3 else '⚠'}")

base = (N_ions+1)*ini.N
for j, a in enumerate(ini.AUX_SPECIES):
    asl = slice(base+j*ini.N, base+(j+1)*ini.N)
    print(f"  {a['name']} (mobile, reservoir): min = {sol.y[asl,:].min():+.3e} mM")
print(f"  Protein: min = {sol.y[psl,:].min():+.3e} mM")

# ---------------- Positivity ----------------
print("\nPositivity:")
ok = True
for k, s in enumerate(ION_SPECIES):
    cmin = sol.y[k*ini.N:(k+1)*ini.N, :].min()
    ok &= cmin >= -1e-10
    print(f"  {s['name']:5s}: min = {cmin:+.3e} mM")
cpmin = sol.y[N_ions*ini.N:, :].min()
print(f"  Protein: min = {cpmin:+.3e} mM")

# ---------------- Final V_m + summary ----------------
phi_f = get_phi(sol.y[:, -1])
V_m_final = (np.mean(phi_f[ini.cyto_mask]) - np.mean(phi_f[ini.bulk_mask])) * 1e3
print(f"\nFinal membrane potential V_m = {V_m_final:.4f} mV   (Donnan target: {dphi_don*1e3:.4f} mV)")

# ---------------- Plots ----------------
results = plot_all(sol)

# --- timescale fit summary ---
from timescale_fit import build_diagnostics, print_summary
from configuration import SPATIAL_COLORS

p_hist_arg  = results['p_hist'] 
li_no       = results['monitor']['li_no']
_diags = build_diagnostics(results['t_ns'], results['ion_hist'],
                           p_hist_arg, li_no, SPATIAL_COLORS)
print_summary(_diags, t_window_ns=results['t_ns'][-1])

# ---------------- CSV exports ----------------
mon = results['monitor']
t_ns = results['t_ns']; z_nm = results['z_nm']

# Temporal
tdat = {'Time_ns': t_ns}
for k, s in enumerate(ION_SPECIES):
    tdat[f"C_{s['name']}_near_in_mM"]  = results['ion_hist'][k][:, mon['li_ni']]
    tdat[f"C_{s['name']}_near_out_mM"] = results['ion_hist'][k][:, mon['li_no']]
    tdat[f"C_{s['name']}_far_in_mM"]   = results['ion_hist'][k][:, mon['li_fi']]
    tdat[f"C_{s['name']}_far_out_mM"]  = results['ion_hist'][k][:, mon['li_fo']]
tdat['C_protein_near_in_mM']  = results['p_hist'][:, mon['li_ni']]
tdat['C_protein_near_out_mM'] = results['p_hist'][:, mon['li_no']]
tdat['C_protein_far_in_mM']   = results['p_hist'][:, mon['li_fi']]
tdat['C_protein_far_out_mM']  = results['p_hist'][:, mon['li_fo']]
for label, li in mon.items():
    tdat[f"phi_{label}_mV"] = results['phi_hist'][:, li]
    tdat[f"Pi_{label}_kPa"] = results['Pi_hist'][:, li]
pd.DataFrame(tdat).set_index('Time_ns').to_csv(OUTPUT_DIR/'Temporal_datapoints.csv')

# Spatial (final)
sdat = {'z_nm': z_nm}
for k, s in enumerate(ION_SPECIES):
    sdat[f"C_{s['name']}_final_mM"] = results['ion_hist'][k][-1]
sdat['C_protein_final_mM'] = results['p_hist'][-1]
sdat['phi_final_mV']       = results['phi_hist'][-1]
sdat['Pi_final_kPa']       = results['Pi_hist'][-1]
pd.DataFrame(sdat).set_index('z_nm').to_csv(OUTPUT_DIR/'Spatial_datapoints.csv')

# Parameters
params = [
    ('SIM', 'sim_t (ns)', SIM_T), ('SIM', 'sim_L (nm)', SIM_L),
    ('SIM', 'ext_L (nm)', EXT_L), ('SIM', 'N_grid', N_GRID),
    ('MAT', 'eps_r_w', EPS_R_W), ('MAT', 'eps_r_M', EPS_R_M),
    ('MAT', 'eta_w', ETA_W), ('MAT', 'eta_c', ETA_C), ('MAT', 'eta_M', ETA_M),
    ('MEM', 'd_mem (nm)', D_MEM*1e9),
    ('PROT', 'C_p_cyto (mM)', C_PROTEIN_CYTO), ('PROT', 'z_protein', Z_PROTEIN),
    ('PROT', 'a_p (nm)', A_P*1e9), ('PROT', 'R_H (nm)', R_H*1e9),
    ('PROT', 'alpha_block', ALPHA_BLOCK),
    ('IMMO', 'Z_X', Z_X), ('IMMO', 'X_cyto', X_cyto), ('IMMO', 'X_bulk', X_bulk),
    ('IMMO', 'Y_cyto', Y_cyto), ('IMMO', 'Y_bulk', Y_bulk),
    ('REST', 'V_rest target (mV)', V_REST_MV),
    ('REST', 'V_m(0) numerical (mV)', _INIT_VM),
    ('REST', 'V_m tolerance (mV)', V_REST_TOL_MV),
    ('REST', 'iterations', _INIT_ITER),
    ('REST', 'N_Debye_layers', N_DEBYE_LAYERS),
    ('DERIV', 'lambda_D (nm)', lambda_D*1e9),
    ('DERIV', 'dphi_don (mV)', dphi_don*1e3),
    ('DERIV', 'Pi_don_cyto (kPa)', Pi_don_cyto),
    ('DERIV', 'Pi_don_bulk (kPa)', Pi_don_bulk),
    ('DERIV', 'V_m_final (mV)', V_m_final),
]
for s in ION_SPECIES:
    params.append(('IONS', f"{s['name']}_C_cyto",  s['C_cyto']))
    params.append(('IONS', f"{s['name']}_C_bulk",  s['C_bulk']))
    params.append(('IONS', f"{s['name']}_D0",      s['D0']))
pd.DataFrame(params, columns=['Category','Parameter','Value']).to_csv(
    OUTPUT_DIR/'Parameters_and_Data.csv', index=False)

from configuration import SNAP_FRACS
snap = {'z_nm': z_nm}
tfin = t_ns[-1]
for f in SNAP_FRACS:
    ti  = int(np.argmin(np.abs(t_ns - f*tfin)))
    tag = f"{t_ns[ti]:.4g}ns"
    for k, s in enumerate(ION_SPECIES):
        snap[f"C_{s['name']}_{tag}_mM"] = results['ion_hist'][k][ti]
    snap[f"C_protein_{tag}_mM"] = results['p_hist'][ti]
    snap[f"phi_{tag}_mV"]       = results['phi_hist'][ti]
    snap[f"Pi_{tag}_kPa"]       = results['Pi_hist'][ti]
pd.DataFrame(snap).set_index('z_nm').to_csv(OUTPUT_DIR/'Spatial_snapshots.csv')

print(f"\nAll outputs written to: {OUTPUT_DIR}")
print("Done.\n")