# coding=utf-8
"""Grid, immobile species, IC (with resting-potential shift), Donnan reference."""
import numpy as np
import sys
from scipy.optimize import fsolve
from scipy.linalg import solve_banded
from configuration import *

    
# ---------------- Grid ----------------
L_ext = EXT_L * 1e-9
L_box = SIM_L * 1e-9
N     = N_GRID
dz    = 2 * L_ext / (N - 1)
z     = np.linspace(-L_ext, L_ext, N)

mem_mask  = np.abs(z) <= D_HALF
bulk_mask = z >  D_HALF
cyto_mask = z < -D_HALF

# ---------------- Debye length ----------------
lambda_D = np.sqrt(
    (EPS_R_W * EPS_0 * K_B * T_K) /
    (E_C**2 * sum(s['z']**2 * s['C_bulk'] * N_A for s in ION_SPECIES))
)

# ---------------- Permittivity ----------------
eps_profile = np.where(mem_mask, EPS_R_M*EPS_0, EPS_R_W*EPS_0)
eps_half = (2.0*eps_profile[:-1]*eps_profile[1:]
            /(eps_profile[:-1]+eps_profile[1:]))

def step_profile(v_cyto, v_bulk):
    p = np.zeros(N)
    p[cyto_mask] = v_cyto
    p[bulk_mask] = v_bulk
    p[mem_mask]  = 0.5 * (v_cyto + v_bulk)
    return p

# ---------------- cytosol protein FROM electroneutrality ----------------
# Neutralise the cytosolic ionic net charge with protein (z = Z_PROTEIN):
#     Z_PROTEIN * C_p_cyto + sum_i z_i C_i^cyto = 0
sum_zC_cyto_ions = sum(s['z'] * s['C_cyto'] for s in ION_SPECIES)
C_PROTEIN_CYTO   = -sum_zC_cyto_ions / Z_PROTEIN
if C_PROTEIN_CYTO <= 0.0:
    print("[FATAL] Cytosolic protein concentration required for electroneutrality "
          f"is non-positive: C_PROTEIN_CYTO = {C_PROTEIN_CYTO:.6g} mM")
    print(f"        cyto ionic net charge = {sum_zC_cyto_ions:+.6g} mM, "
          f"Z_PROTEIN = {Z_PROTEIN}.  Cannot enforce cytosolic electroneutrality.")
    sys.exit(1)

# ---------------- extracellular X FROM electroneutrality ----------------
# X lives ONLY on the extracellular side; its SIGN is chosen to cancel the bulk
# ionic net charge:  z_X * C_X^bulk + sum_i z_i C_i^bulk = 0,  C_X^bulk >= 0.
sum_zC_bulk_ions = sum(s['z'] * s['C_bulk'] for s in ION_SPECIES)
if   sum_zC_bulk_ions > 0:   Z_X, X_bulk = -1,  sum_zC_bulk_ions
elif sum_zC_bulk_ions < 0:   Z_X, X_bulk = +1, -sum_zC_bulk_ions
else:                        Z_X, X_bulk = -1,  0.0
X_cyto = 0.0                      # X exists ONLY extracellularly

# ---------------- Y (neutral) → equal total concentration across membrane ----------------
sum_C_bulk = sum(s['C_bulk'] for s in ION_SPECIES) + X_bulk          # X_cyto = 0
sum_C_cyto = sum(s['C_cyto'] for s in ION_SPECIES) + C_PROTEIN_CYTO
if sum_C_cyto > sum_C_bulk:
    Y_bulk, Y_cyto = sum_C_cyto - sum_C_bulk, 0.0
else:
    Y_cyto, Y_bulk = sum_C_bulk - sum_C_cyto, 0.0
tot_cyto = sum_C_cyto + Y_cyto
tot_bulk = sum_C_bulk + Y_bulk
Pi_init_cyto = R_GAS * T_K * tot_cyto / 1e3
Pi_init_bulk = R_GAS * T_K * tot_bulk / 1e3

# ---------------- Diffusivities ----------------
N_ions = len(ION_SPECIES)
def build_ion_D(D0):
    D = np.full(N, D0)
    D[mem_mask] = (ETA_W/ETA_M)*D0
    return D
ion_D_profiles = [build_ion_D(s['D0']) for s in ION_SPECIES]

D_p_w = K_B*T_K/(6.0*np.pi*ETA_W*R_H)
D_p_c = (ETA_W/ETA_C)*D_p_w
D_p_M = (ETA_W/ETA_M)*D_p_w*ALPHA_BLOCK
D_p_profile = np.where(bulk_mask, D_p_w, np.where(mem_mask, D_p_M, D_p_c))

zeta_p  = (Z_PROTEIN*E_C)/(4.0*np.pi*EPS_R_W*EPS_0*R_H)
kappa_a = A_P/lambda_D
f_henry = 1.0 + 0.5/(1.0 + 2.5/(kappa_a*(1.0+2.0*np.exp(-kappa_a))))**3
mu_p_w = (2.0*EPS_R_W*EPS_0*zeta_p*f_henry)/(3.0*ETA_W)
mu_p_c = (2.0*EPS_R_W*EPS_0*zeta_p*f_henry)/(3.0*ETA_C)
mu_p_M = (2.0*EPS_R_M*EPS_0*zeta_p*f_henry)/(3.0*ETA_M)*ALPHA_BLOCK
mu_p_profile = np.where(bulk_mask, mu_p_w, np.where(mem_mask, mu_p_M, mu_p_c))

# ---------------- Base (electroneutral) IC ----------------
C0_ions_base = [step_profile(s['C_cyto'], s['C_bulk']) for s in ION_SPECIES]
for Ci in C0_ions_base:
    Ci[mem_mask] = 0.0          # ion-excluding lipid: no mobile ions in membrane at t=0
C_p0 = np.zeros(N)
C_p0[cyto_mask] = C_PROTEIN_CYTO           # protein neutralises cyto ions

# ---------------- X (extracellular only) & Y (neutral) profiles ----------------
C_X_profile = np.zeros(N)
C_X_profile[bulk_mask] = X_bulk            # X neutralises bulk ions; zero elsewhere
C_Y_profile = step_profile(Y_cyto, Y_bulk) # neutral osmolyte (osmolarity only)

# ---------------- Immobile membrane background charge (V_m(0)=0) ----------------
# Protein neutralises the cytosol, X neutralises the bulk, but the membrane
# (|z|<=D_HALF) holds interpolated ion concentrations whose net charge is
# uncompensated (protein/X live only on their own sides). Cancel that ionic net
# charge with a FIXED immobile background so rho(z)=0 EVERYWHERE at baseline and
# V_m(baseline)=0. Not evolved (represents fixed membrane charge).
_ion_charge0 = np.zeros(N)
for k, s in enumerate(ION_SPECIES):
    _ion_charge0 += s['z'] * C0_ions_base[k]
rho_fixed = np.zeros(N)
rho_fixed[mem_mask] = -_ion_charge0[mem_mask]      # mM (charge-equivalent)

# ---- X, Y are mobile & membrane-permeant
# Their initial profiles (below) enforce local electroneutrality / osmotic
# balance at t=0, but they are free to Donnan-partition afterward, so the ONLY
# trapped charge that sets the equilibrium Donnan ratio is the protein.
AUX_D0 = 2.0e-9
def _build_aux_D(D0):
    D = np.full(N, D0)
    D[mem_mask] = (ETA_W/ETA_M) * D0
    return D

AUX_SPECIES = [
    {'name': 'X', 'z': Z_X, 'D0': AUX_D0, 'C0': C_X_profile.copy(),
     'C_bulk': float(C_X_profile[bulk_mask][-1])},
    {'name': 'Y', 'z': 0,   'D0': AUX_D0, 'C0': C_Y_profile.copy(),
     'C_bulk': float(C_Y_profile[bulk_mask][-1])},
]
aux_D_profiles = [_build_aux_D(a['D0']) for a in AUX_SPECIES]
N_aux = len(AUX_SPECIES)

# ---------------- Local Poisson solver (for IC verification only) ----------------
from scipy.linalg import solve_banded
def _solve_poisson(rho):
    rhs = np.zeros(N); rhs[1:-1] = -(F_C * rho[1:-1])
    lower = eps_half[:-1]/dz**2
    upper = eps_half[1:] /dz**2
    diag  = -(eps_half[:-1] + eps_half[1:])/dz**2
    d = np.empty(N); o_up = np.empty(N-1); o_lo = np.empty(N-1)
    d[1:-1] = diag; o_up[1:] = upper; o_lo[:-1] = lower
    d[0]  = -1.0; o_up[0]  = 1.0; rhs[0]  = 0.0    # Neumann at left
    d[-1] =  1.0; o_lo[-1] = 0.0; rhs[-1] = 0.0    # Dirichlet at right (phi=0)
    ab = np.zeros((3, N)); ab[0,1:] = o_up; ab[1] = d; ab[2,:-1] = o_lo
    return solve_banded((1,1), ab, rhs)
def _rho(C_ion_list, Cp):
    r = Z_X*C_X_profile + Z_PROTEIN*Cp
    for k, s in enumerate(ION_SPECIES):
        r = r + s['z'] * C_ion_list[k]
    return r
# ---------------- Impose V_rest via multi-ion Boltzmann-weighted Debye shift ----
# build cyto-side and bulk-side Debye slabs, then move
# a per-species amount d_i = w_i * d from donor slab to acceptor slab, where
#   w_i = z_i^2 * <c_i>  (normalized),
# and the acceptor is chosen so that shifting always drives Vm toward the target.
# d is found by bracket + bisection on a Poisson-resolved Vm(d).

V_target = V_REST_MV * 1e-3
slab     = max(lambda_D, 1.5 * dz)

# Debye-layer masks. Cyto side sits at z < -D_HALF, bulk side at z > +D_HALF.
in_mask  = (z >= -D_HALF - slab) & (z <  -D_HALF)   # cyto-side Debye layer
out_mask = (z >   D_HALF)        & (z <=  D_HALF + slab)  # bulk-side Debye layer
if not in_mask.any() or not out_mask.any():
    raise RuntimeError("Empty Debye slab; check mesh vs membrane thickness.")

# 1D "volumes" (line-integrated weights) so mass moved from cyto layer equals
# mass gained by bulk layer, with different slab widths handled correctly.
Vin  = in_mask.sum()  * dz
Vout = out_mask.sum() * dz

# ----- Baseline C0_ions (electroneutral) --------------------------------------
C0_ions = [Ci.copy() for Ci in C0_ions_base]

def _Vm_from_C(C_list):
    phi = _solve_poisson(_rho(C_list, C_p0))
    return (np.mean(phi[cyto_mask]) - np.mean(phi[bulk_mask]))  # in Volts

V_cur = _Vm_from_C(C0_ions)

# ----- Per-species Boltzmann weights + motion direction -----------------------
# direction=+1 lowers Vm (cations cyto->bulk, anions bulk->cyto);
# direction=-1 raises Vm.
direction = +1.0 if (V_target - V_cur) < 0.0 else -1.0

species_shift = []   # each entry: [idx, z_i, w_i, dir_sign, Cbase]
W_sum = 0.0
for idx, s in enumerate(ION_SPECIES):
    z_i = s['z']
    if z_i == 0:
        continue
    c_avg    = 0.5 * (s['C_cyto'] + s['C_bulk'])
    w_i      = (z_i * z_i) * c_avg
    dir_sign = +1.0 if (z_i * direction > 0) else -1.0    # +1: in->out
    species_shift.append([idx, z_i, w_i, dir_sign, C0_ions[idx].copy()])
    W_sum += w_i
if W_sum <= 0.0:
    raise RuntimeError("No mobile ions available to build Debye layer.")
for s in species_shift:
    s[2] /= W_sum                                          # normalize weights

# ----- Per-species donor cap; trace ions must NOT throttle the global shift ---
# A species with a nearly-empty donor slab (e.g. cytosolic Ca) contributes
# negligible charge but would otherwise cap the shared scalar d. So:
#   (a) exclude tiny-donor species from the *global* max_d, and
#   (b) clip each species' own move inside _apply_d so it never goes negative.
C_FLOOR = 1.0   # mM: donors below this don't participate in global cap
max_d = np.inf
for idx, z_i, w_i, dir_sign, Cbase in species_shift:
    if w_i <= 0:
        continue
    donor   = in_mask if dir_sign > 0 else out_mask
    src_min = float(Cbase[donor].min())
    if src_min < C_FLOOR:            # trace donor: skip from GLOBAL cap
        continue
    max_d = min(max_d, 0.99 * src_min / w_i)
if not np.isfinite(max_d):
    # fallback: use the largest abundant donor
    max_d = 10.0 * max(s['C_cyto'] + s['C_bulk'] for s in ION_SPECIES)

def _apply_d(d):
    """Apply Boltzmann-weighted total shift d.
    Each species is clipped to its OWN donor limit, so trace ions saturate
    early and abundant ions keep driving Vm toward the target."""
    C_try = [Ci.copy() for Ci in C0_ions_base]
    for idx, z_i, w_i, dir_sign, Cbase in species_shift:
        Cn    = Cbase.copy()
        donor = in_mask if dir_sign > 0 else out_mask
        src_min = float(Cbase[donor].min())
        di = min(w_i * d, max(0.0, 0.99 * src_min))   # per-species clip
        if dir_sign > 0:                               # cyto layer -> bulk layer
            Cn[in_mask]  -= di
            Cn[out_mask] += di * Vin / Vout
        else:                                          # bulk layer -> cyto layer
            Cn[in_mask]  += di * Vout / Vin
            Cn[out_mask] -= di
        C_try[idx] = Cn
    for k_, Ck in enumerate(C_try):
        C0_ions[k_] = Ck
    return _Vm_from_C(C0_ions)                      

# ----- Bracket then bisect on d -----------------------------------------------
lo, hi = 0.0, min(max(1e-3, 0.05 * max_d), max_d)
V_lo = _apply_d(lo)
V_hi = _apply_d(hi)
if V_hi is None:
    hi = 0.5 * max_d
    V_hi = _apply_d(hi)

f_lo = V_lo - V_target
f_hi = V_hi - V_target
n_expand = 0
while f_lo * f_hi > 0 and hi < max_d and n_expand < 30:
    hi = min(2.0 * hi, max_d)
    V_hi = _apply_d(hi)
    if V_hi is None:
        break
    f_hi = V_hi - V_target
    n_expand += 1

d_star, V_star = hi, V_hi
_INIT_ITER = 1

if f_lo * f_hi <= 0:
    for it in range(60):
        mid  = 0.5 * (lo + hi)
        V_mid = _apply_d(mid)
        if V_mid is None:                                 # retreat on failure
            hi = mid
            continue
        fm = V_mid - V_target
        d_star, V_star = mid, V_mid
        _INIT_ITER = it + 1
        if abs(fm) * 1e3 < V_REST_TOL_MV:
            break
        if f_lo * fm <= 0:
            hi, f_hi = mid, fm
        else:
            lo, f_lo = mid, fm

# Final commit
V_star = _apply_d(d_star)

# ----- Package for the rest of initialization ---------------------------------
y0 = np.concatenate(C0_ions + [C_p0] + [a['C0'] for a in AUX_SPECIES])
_phi_ic    = _solve_poisson(_rho(C0_ions, C_p0))
_INIT_VM   = (np.mean(_phi_ic[cyto_mask]) - np.mean(_phi_ic[bulk_mask])) * 1e3
_INIT_OK   = abs(_INIT_VM - V_REST_MV) < V_REST_TOL_MV

# ---------------- Donnan reference (reservoir / single-ratio) ----------------
from scipy.optimize import brentq

def _reservoir_ratio(sigma_trapped):
    """Cytosolic EN vs FIXED reservoir; every mobile species Boltzmann-partitions:
       sum_i z_i C_bulk_i r^z_i + sum_aux z_a C_bulk_a r^z_a + sigma_trapped = 0."""
    def f(r):
        s  = sum(sp['z']*sp['C_bulk']*r**sp['z'] for sp in ION_SPECIES)
        s += sum(a['z'] *a['C_bulk'] *r**a['z']  for a in AUX_SPECIES)
        return s + sigma_trapped
    return brentq(f, 1e-8, 1e8)

sigma_protein = Z_PROTEIN * C_PROTEIN_CYTO
sigma_trapped = sigma_protein              

r_don        = _reservoir_ratio(sigma_trapped)
dphi_don     = -V_T * np.log(r_don)
C_don_cyto   = [s['C_bulk'] * r_don**s['z'] for s in ION_SPECIES]
C_don_bulk   = [s['C_bulk']                 for s in ION_SPECIES]
C_aux_don_cyto = [a['C_bulk'] * r_don**a['z'] for a in AUX_SPECIES]

# backward-compatible exports
sigma_X         = sum(a['z']*C_aux_don_cyto[j] for j, a in enumerate(AUX_SPECIES))  # residual
sigma_total     = sigma_trapped
r_don_prot      = r_don
dphi_don_prot   = dphi_don
C_don_cyto_prot = C_don_cyto

C_don_cyto_tot = sum(C_don_cyto) + C_PROTEIN_CYTO + sum(C_aux_don_cyto)
C_don_bulk_tot = sum(C_don_bulk) + sum(a['C_bulk'] for a in AUX_SPECIES)
Pi_don_cyto = R_GAS*T_K*C_don_cyto_tot / 1e3
Pi_don_bulk = R_GAS*T_K*C_don_bulk_tot / 1e3