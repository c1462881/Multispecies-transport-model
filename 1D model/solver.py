# coding=utf-8
"""PNP RHS and time integration (single phase, RK45)."""
import numpy as np
from time import time as wtime
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
from configuration import *
from initialization import (
    N, dz, z, mem_mask, cyto_mask, bulk_mask, eps_half,
    N_ions, ion_D_profiles, D_p_profile, mu_p_profile,
    N_aux, AUX_SPECIES, aux_D_profiles, y0,
)

def total_rho(y):
    r = np.zeros(N)
    for k in range(N_ions):
        r = r + ION_SPECIES[k]['z'] * y[k*N:(k+1)*N]
    r = r + Z_PROTEIN * y[N_ions*N:(N_ions+1)*N]      
    base = (N_ions+1)*N
    for j, a in enumerate(AUX_SPECIES):               
        r = r + a['z'] * y[base+j*N: base+(j+1)*N]
    return r

def solve_poisson(rho):
    rhs = np.zeros(N); rhs[1:-1] = -(F_C * rho[1:-1])
    lower = eps_half[:-1]/dz**2; upper = eps_half[1:]/dz**2
    diag  = -(eps_half[:-1] + eps_half[1:])/dz**2
    d = np.empty(N); o_up = np.empty(N-1); o_lo = np.empty(N-1)
    d[1:-1] = diag; o_up[1:] = upper; o_lo[:-1] = lower
    d[0]  = -1.0; o_up[0]  = 1.0; rhs[0]  = 0.0
    d[-1] =  1.0; o_lo[-1] = 0.0; rhs[-1] = 0.0
    ab = np.zeros((3, N)); ab[0,1:] = o_up; ab[1] = d; ab[2,:-1] = o_lo
    return solve_banded((1,1), ab, rhs)

def get_phi(y):
    return solve_poisson(total_rho(y))

def sg_flux(C, phi, D_arr, z_i):
    dphi = np.diff(phi); x = z_i*dphi/V_T
    D_int = 0.5*(D_arr[:-1]+D_arr[1:])
    x_clip = np.clip(x, -500.0, 500.0)
    Bx = np.ones_like(x); Bmx = np.ones_like(x)
    m = np.abs(x_clip) > 1e-6
    Bx[m]  =  x_clip[m]/np.expm1( x_clip[m])
    Bmx[m] = -x_clip[m]/np.expm1(-x_clip[m])
    xt = x_clip[~m]
    Bx[~m]  = 1.0 - 0.5*xt + xt**2/12.0
    Bmx[~m] = 1.0 + 0.5*xt + xt**2/12.0
    return (D_int/dz)*(Bx*C[:-1] - Bmx*C[1:])

def protein_flux(Cp, phi):
    J_diff  = sg_flux(Cp, phi, D_p_profile, 0)
    mu_int  = 0.5*(mu_p_profile[:-1]+mu_p_profile[1:])
    dphi_dz = np.diff(phi)/dz
    C_int   = 0.5*(Cp[:-1]+Cp[1:])
    return J_diff - mu_int * C_int * dphi_dz

def positivity_limit(C, J):
    J_lim = J.copy(); Nf = len(C)
    out_r = np.zeros(Nf); out_r[:-1] = np.maximum(J_lim, 0.0)
    out_l = np.zeros(Nf); out_l[1:]  = np.maximum(-J_lim, 0.0)
    tot   = out_r + out_l
    maxo  = 0.9*np.maximum(C, 0.0)*dz/(dz**2/2e-9)
    sc = np.ones(Nf); m = (tot>maxo) & (tot>0); sc[m] = maxo[m]/tot[m]
    J_lim = np.where(J_lim>0, J_lim*sc[:-1],
             np.where(J_lim<0, J_lim*sc[1:],  J_lim))
    J_lim[(J_lim>0) & (C[:-1]<=0)] = 0.0
    J_lim[(J_lim<0) & (C[1:] <=0)] = 0.0
    return J_lim

def rhs_ode(t, y):
    y = np.maximum(y, 0.0); phi = get_phi(y); dC = np.zeros_like(y)
    for k in range(N_ions):
        Ck = y[k*N:(k+1)*N]
        Jk = sg_flux(Ck, phi, ion_D_profiles[k], ION_SPECIES[k]['z'])
        Jk = positivity_limit(Ck, Jk)
        dCk = np.zeros(N); dCk[1:-1] = -(Jk[1:]-Jk[:-1])/dz
        dCk[0] = dCk[1]; dCk[-1] = 0.0
        dC[k*N:(k+1)*N] = dCk
    Cp = y[N_ions*N:(N_ions+1)*N]
    Jp = protein_flux(Cp, phi); Jp = positivity_limit(Cp, Jp)
    dCp = np.zeros(N); dCp[1:-1] = -(Jp[1:]-Jp[:-1])/dz
    dCp[0] = dCp[1]; dCp[-1] = dCp[-2]
    dC[N_ions*N:(N_ions+1)*N] = dCp

    base = (N_ions+1)*N                                
    for j, a in enumerate(AUX_SPECIES):
        Ca = y[base+j*N: base+(j+1)*N]
        Ja = sg_flux(Ca, phi, aux_D_profiles[j], a['z'])
        Ja = positivity_limit(Ca, Ja)
        dCa = np.zeros(N); dCa[1:-1] = -(Ja[1:]-Ja[:-1])/dz
        dCa[0] = dCa[1]        # no-flux at far-cyto (left)
        dCa[-1] = 0.0          # Dirichlet reservoir at bulk (right)
        dC[base+j*N: base+(j+1)*N] = dCa
    return dC

def run_simulation():
    t_span = (0.0, SIM_T*1e-9)
    t_eval = np.linspace(0.0, SIM_T*1e-9, N_T_EVAL)
    atol   = np.full(y0.size, 1e-6)
    D_max  = max(s['D0'] for s in ION_SPECIES)
    max_step = min(dz**2/(2*D_max)*100, SIM_T*1e-9/50)
    t0 = wtime(); _last = [0.0]
    def rhs_timed(t, y):
        if t*1e9 > _last[0] + 100:
            print(f"  t = {t*1e9:.1f} ns  (wall: {wtime()-t0:.1f} s)")
            _last[0] = t*1e9
        return rhs_ode(t, y)
    print(f"\nIntegrating 0 → {SIM_T} ns ...")
    sol = solve_ivp(rhs_timed, t_span, y0, method='RK45',
                    t_eval=t_eval, rtol=1e-4, atol=atol, max_step=max_step)
    print(f"  {sol.message}  ({sol.nfev} evals)")
    if not sol.success:
        raise RuntimeError(sol.message)
    sol.y = np.maximum(sol.y, 0.0)
    return sol