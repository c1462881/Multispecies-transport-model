#!/usr/bin/env python3
"""
solver.py

GPU kernels, PCG Poisson solver, SG updates, convolution helpers,
physics cache, diagnostics.
"""

import cupy as cp
import numpy as np
from cupyx.scipy.signal import fftconvolve
from cupyx.scipy.ndimage import convolve as ndi_convolve

from configuration import F, R, NA, eps0, kB, Tem, VT


# =====================================================================
# Basic helper
# =====================================================================
def harm(a, b):
    return 2 * a * b / (a + b + 1e-300)


# =====================================================================
# Protein convolution kernels
# =====================================================================
def make_ball_kernel(R, dx, dy, dz, dtype, normalize=False):
    rx, ry, rz = int(np.ceil(R / dx)), int(np.ceil(R / dy)), int(np.ceil(R / dz))
    xs = cp.arange(-rx, rx + 1, dtype=dtype) * dx
    ys = cp.arange(-ry, ry + 1, dtype=dtype) * dy
    zs = cp.arange(-rz, rz + 1, dtype=dtype) * dz
    XX, YY, ZZ = cp.meshgrid(xs, ys, zs, indexing="ij")
    K = (cp.sqrt(XX**2 + YY**2 + ZZ**2) <= R).astype(dtype)
    if normalize:
        K = K / cp.sum(K)
    return K


def make_shell_kernel(R, dx, dy, dz, dtype):
    h = max(dx, dy, dz)
    rx, ry, rz = int(np.ceil((R + h) / dx)), int(np.ceil((R + h) / dy)), int(np.ceil((R + h) / dz))
    xs = cp.arange(-rx, rx + 1, dtype=dtype) * dx
    ys = cp.arange(-ry, ry + 1, dtype=dtype) * dy
    zs = cp.arange(-rz, rz + 1, dtype=dtype) * dz
    XX, YY, ZZ = cp.meshgrid(xs, ys, zs, indexing="ij")
    rr = cp.sqrt(XX**2 + YY**2 + ZZ**2)
    K = (cp.abs(rr - R) <= 0.6 * h).astype(dtype)
    if float(cp.sum(K).get()) <= 0:
        K = (cp.abs(rr - R) <= h).astype(dtype)
    return K / cp.sum(K)


class Convolver3D:
    def __init__(self, method="auto", direct_max=4096):
        self.method = method
        self.dmax = direct_max

    def __call__(self, f, k):
        if self.method == "fft":
            return fftconvolve(f, k, mode="same")
        if self.method == "direct":
            return ndi_convolve(f, k, mode="constant", cval=0.0)
        return ndi_convolve(f, k, mode="constant", cval=0.0) if k.size <= self.dmax else fftconvolve(f, k, mode="same")


# =====================================================================
# Physics cache
# =====================================================================
class PhysicsCache:
    def __init__(self, shape, dtype, ions, prot_species, conv, PHI_MAX):
        # prot_species: list of dicts with keys: name, zprot, K_ball, V_prot
        self.ions = ions
        self.prot_species = prot_species
        self.PHI_MAX = PHI_MAX
        self.conv = conv

        self.ion_rm = cp.zeros(shape, dtype=dtype)
        self.prot_rm = cp.zeros(shape, dtype=dtype)
        self.steric = cp.zeros(shape, dtype=dtype)
        self.rho = cp.zeros(shape, dtype=dtype)
        self.osm = cp.zeros(shape, dtype=dtype)
        self._tmp = cp.zeros(shape, dtype=dtype)

        self.rho_imm = None
        self.osm_imm = None

    def set_immobile(self, rho_imm, osm_imm):
        self.rho_imm = rho_imm
        self.osm_imm = osm_imm

    def update_ions(self, C):
        self.ion_rm.fill(0.0)
        for Ci, s in zip(C, self.ions):
            self.ion_rm += s["z"] * Ci

    def update_protein(self, nprot_list, prot_mob_list=None):
        self.steric.fill(0.0)
        self.prot_rm.fill(0.0)
        for i, spx in enumerate(self.prot_species):
            nbar = self.conv(nprot_list[i], spx["K_ball"])
            cp.multiply(nbar, spx["V_prot"], out=self._tmp)
            if prot_mob_list is not None:
                cp.multiply(self._tmp, prot_mob_list[i], out=self._tmp)
            self.steric += self._tmp
            self.prot_rm += (spx["zprot"] / NA) * nprot_list[i]
        cp.clip(self.steric, 0.0, self.PHI_MAX, out=self.steric)

    def update_all(self, C, nprot_list, prot_mob_list=None):
        self.update_ions(C)
        self.update_protein(nprot_list, prot_mob_list=prot_mob_list)

    def charge_density(self, mobile, rw, use_wall=True):
        self.rho[...] = self.ion_rm
        self.rho += self.prot_rm
        self.rho *= F
        self.rho *= mobile
        if self.rho_imm is not None:
            self.rho += self.rho_imm
        if use_wall:
            self.rho += rw
        return self.rho

    def osmolarity(self, C, nprot_list):
        self.osm.fill(0.0)
        for Ci in C:
            self.osm += Ci
        for npi in nprot_list:
            self.osm += npi / NA
        if self.osm_imm is not None:
            self.osm += self.osm_imm
        return self.osm
        
# =====================================================================
# SG kernel
# =====================================================================
_sg_cache = {}
def get_sg_kernel(dtype):
    dtype = cp.dtype(dtype)
    key = str(dtype)
    if key in _sg_cache: return _sg_cache[key]
    real = "float" if dtype==cp.float32 else "double"
    expf = "expf"  if dtype==cp.float32 else "exp"
    fabsf= "fabsf" if dtype==cp.float32 else "fabs"
    feps = "1.0e-30f" if dtype==cp.float32 else "1.0e-300"
    code = f"""
    typedef {real} real_t;
    __device__ inline real_t bern(real_t x){{
        real_t ax = {fabsf}(x);
        if (ax < (real_t)1e-4){{
            real_t x2=x*x;
            return (real_t)1.0 - (real_t)0.5*x + x2/(real_t)12.0 - x2*x2/(real_t)720.0;
        }}
        if (x > (real_t)80.0) return x*{expf}(-x);
        if (x < (real_t)-80.0) return -x;
        return x/({expf}(x)-(real_t)1.0);
    }}
    __device__ inline real_t harm2(real_t a, real_t b){{
        return ((real_t)2.0*a*b)/(a+b+(real_t){feps});
    }}
    extern "C" __global__
    void sg_nu(const real_t* __restrict__ C, const real_t* __restrict__ Db,
            const real_t* __restrict__ gate, const real_t* __restrict__ phi,
            const real_t* __restrict__ mask,
            const real_t* __restrict__ hxf, const real_t* __restrict__ hyf,
            const real_t* __restrict__ hzf,
            const real_t* __restrict__ dxc, const real_t* __restrict__ dyc,
            const real_t* __restrict__ dzc,
            real_t* __restrict__ Cn,
            int Nx, int Ny, int Nz,
            real_t dt, real_t zval, real_t VTv, real_t Cfl)
    {{
        int N = Nx*Ny*Nz;
        int id = blockDim.x*blockIdx.x + threadIdx.x;
        if (id>=N) return;
        int k = id%Nz; int j=(id/Nz)%Ny; int i=id/(Ny*Nz);
        int sx=Ny*Nz, sy=Nz, sz=1;
        if (mask[id] < (real_t)0.5){{ Cn[id]=Cfl; return; }}
        real_t c0 = C[id]; if (c0<Cfl) c0=Cfl;
        real_t rhs=(real_t)0.0;
        real_t D0 = Db[id]*gate[id]*mask[id];
        // x-
        if (i>0){{
            int nb=id-sx; real_t cL=C[nb]; if(cL<Cfl) cL=Cfl;
            real_t Df=harm2(D0, Db[nb]*gate[nb]*mask[nb]);
            real_t hf=hxf[i-1];
            real_t psi=zval*(phi[id]-phi[nb])/VTv;
            real_t J=-(Df/hf)*( c0*bern(-psi) - cL*bern(psi) );
            rhs += J/dxc[i];
        }}
        // x+
        if (i<Nx-1){{
            int nb=id+sx; real_t cR=C[nb]; if(cR<Cfl) cR=Cfl;
            real_t Df=harm2(D0, Db[nb]*gate[nb]*mask[nb]);
            real_t hf=hxf[i];
            real_t psi=zval*(phi[nb]-phi[id])/VTv;
            real_t J=-(Df/hf)*( cR*bern(-psi) - c0*bern(psi) );
            rhs -= J/dxc[i];
        }}
        // y-
        if (j>0){{
            int nb=id-sy; real_t cL=C[nb]; if(cL<Cfl) cL=Cfl;
            real_t Df=harm2(D0, Db[nb]*gate[nb]*mask[nb]);
            real_t hf=hyf[j-1];
            real_t psi=zval*(phi[id]-phi[nb])/VTv;
            real_t J=-(Df/hf)*( c0*bern(-psi) - cL*bern(psi) );
            rhs += J/dyc[j];
        }}
        if (j<Ny-1){{
            int nb=id+sy; real_t cR=C[nb]; if(cR<Cfl) cR=Cfl;
            real_t Df=harm2(D0, Db[nb]*gate[nb]*mask[nb]);
            real_t hf=hyf[j];
            real_t psi=zval*(phi[nb]-phi[id])/VTv;
            real_t J=-(Df/hf)*( cR*bern(-psi) - c0*bern(psi) );
            rhs -= J/dyc[j];
        }}
        // z-
        if (k>0){{
            int nb=id-sz; real_t cL=C[nb]; if(cL<Cfl) cL=Cfl;
            real_t Df=harm2(D0, Db[nb]*gate[nb]*mask[nb]);
            real_t hf=hzf[k-1];
            real_t psi=zval*(phi[id]-phi[nb])/VTv;
            real_t J=-(Df/hf)*( c0*bern(-psi) - cL*bern(psi) );
            rhs += J/dzc[k];
        }}
        if (k<Nz-1){{
            int nb=id+sz; real_t cR=C[nb]; if(cR<Cfl) cR=Cfl;
            real_t Df=harm2(D0, Db[nb]*gate[nb]*mask[nb]);
            real_t hf=hzf[k];
            real_t psi=zval*(phi[nb]-phi[id])/VTv;
            real_t J=-(Df/hf)*( cR*bern(-psi) - c0*bern(psi) );
            rhs -= J/dzc[k];
        }}
        real_t out = c0 + dt*rhs;
        if (!(out==out) || out<Cfl) out = Cfl;
        Cn[id]=out;
    }}
    """
    ker = cp.RawKernel(code, "sg_nu")
    _sg_cache[key] = ker
    return ker

def sg_update(C, Db, gate, zv, phi, hxf,hyf,hzf, dxc,dyc,dzc, dt, mask, out):
    Nx,Ny,Nz = C.shape
    N = Nx*Ny*Nz
    threads = 256
    blocks = (N+threads-1)//threads
    ker = get_sg_kernel(C.dtype)
    dt_t = C.dtype.type
    ker((blocks,),(threads,),(C,Db,gate,phi,mask,hxf,hyf,hzf,dxc,dyc,dzc,out,
        np.int32(Nx),np.int32(Ny),np.int32(Nz),
        dt_t(dt), dt_t(zv), dt_t(VT), dt_t(1e-30)))
    return out


# =====================================================================
# Batched Thomas (line-Jacobi-z preconditioner)
# =====================================================================
_thomas_cache = {}
def get_thomas_kernel(dtype):
    dtype = cp.dtype(dtype)
    key=str(dtype)
    if key in _thomas_cache: return _thomas_cache[key]
    real = "float" if dtype==cp.float32 else "double"
    code = f"""
    typedef {real} real_t;
    extern "C" __global__
    void thomas_solve(const real_t* __restrict__ a,
                    const real_t* __restrict__ d,
                    const real_t* __restrict__ c,
                    const real_t* __restrict__ rhs,
                    real_t* __restrict__ x,
                    real_t* __restrict__ wcp,
                    real_t* __restrict__ wdp,
                    int Nx, int Ny, int Nz)
    {{
        int j = blockIdx.x*blockDim.x + threadIdx.x;
        int i = blockIdx.y*blockDim.y + threadIdx.y;
        if (i>=Nx || j>=Ny) return;
        int base = (i*Ny + j)*Nz;
        real_t b0 = d[base];
        wcp[base] = c[base]/b0;
        wdp[base] = rhs[base]/b0;
        for (int k=1; k<Nz; ++k){{
            int p = base+k;
            real_t m = d[p] - a[p]*wcp[p-1];
            wcp[p] = (k<Nz-1) ? c[p]/m : (real_t)0.0;
            wdp[p] = (rhs[p] - a[p]*wdp[p-1])/m;
        }}
        x[base+Nz-1] = wdp[base+Nz-1];
        for (int k=Nz-2; k>=0; --k){{
            int p = base+k;
            x[p] = wdp[p] - wcp[p]*x[p+1];
        }}
    }}
    """
    ker = cp.RawKernel(code,"thomas_solve")
    _thomas_cache[key]=ker
    return ker

_thomas_x_cache = {}
def get_thomas_x_kernel(dtype):
    dtype = cp.dtype(dtype); key = str(dtype)
    if key in _thomas_x_cache: return _thomas_x_cache[key]
    real = "float" if dtype == cp.float32 else "double"
    code = f"""
    typedef {real} real_t;
    extern "C" __global__
    void thomas_solve_x(const real_t* __restrict__ a,
                        const real_t* __restrict__ d,
                        const real_t* __restrict__ c,
                        const real_t* __restrict__ rhs,
                        real_t* __restrict__ x,
                        real_t* __restrict__ wcp,
                        real_t* __restrict__ wdp,
                        int Nx, int Ny, int Nz)
    {{
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (j >= Ny || k >= Nz) return;
        int stride = Ny * Nz;
        int base   = j * Nz + k;
        real_t b0 = d[base];
        wcp[base] = c[base] / b0;
        wdp[base] = rhs[base] / b0;
        for (int i = 1; i < Nx; ++i) {{
            int p  = base + i * stride;
            int pm = p - stride;
            real_t m = d[p] - a[p] * wcp[pm];
            wcp[p] = (i < Nx-1) ? c[p] / m : (real_t)0.0;
            wdp[p] = (rhs[p] - a[p] * wdp[pm]) / m;
        }}
        int last = base + (Nx-1) * stride;
        x[last] = wdp[last];
        for (int i = Nx-2; i >= 0; --i) {{
            int p = base + i * stride;
            x[p] = wdp[p] - wcp[p] * x[p + stride];
        }}
    }}
    """
    ker = cp.RawKernel(code, "thomas_solve_x")
    _thomas_x_cache[key] = ker
    return ker


_thomas_y_cache = {}
def get_thomas_y_kernel(dtype):
    dtype = cp.dtype(dtype); key = str(dtype)
    if key in _thomas_y_cache: return _thomas_y_cache[key]
    real = "float" if dtype == cp.float32 else "double"
    code = f"""
    typedef {real} real_t;
    extern "C" __global__
    void thomas_solve_y(const real_t* __restrict__ a,
                        const real_t* __restrict__ d,
                        const real_t* __restrict__ c,
                        const real_t* __restrict__ rhs,
                        real_t* __restrict__ x,
                        real_t* __restrict__ wcp,
                        real_t* __restrict__ wdp,
                        int Nx, int Ny, int Nz)
    {{
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= Nx || k >= Nz) return;
        int stride = Nz;
        int base   = i * Ny * Nz + k;
        real_t b0 = d[base];
        wcp[base] = c[base] / b0;
        wdp[base] = rhs[base] / b0;
        for (int j = 1; j < Ny; ++j) {{
            int p  = base + j * stride;
            int pm = p - stride;
            real_t m = d[p] - a[p] * wcp[pm];
            wcp[p] = (j < Ny-1) ? c[p] / m : (real_t)0.0;
            wdp[p] = (rhs[p] - a[p] * wdp[pm]) / m;
        }}
        int last = base + (Ny-1) * stride;
        x[last] = wdp[last];
        for (int j = Ny-2; j >= 0; --j) {{
            int p = base + j * stride;
            x[p] = wdp[p] - wcp[p] * x[p + stride];
        }}
    }}
    """
    ker = cp.RawKernel(code, "thomas_solve_y")
    _thomas_y_cache[key] = ker
    return ker
# =====================================================================
# PCG with line-Jacobi-z preconditioner, nonuniform FV Poisson
# =====================================================================
class PoissonPCG_LJZ:
    def __init__(self, eps, hxf, hyf, hzf, dxc, dyc, dzc):
        self.dtype = eps.dtype
        Nx, Ny, Nz = eps.shape
        self.Nx = Nx; self.Ny = Ny; self.Nz = Nz
        self.kbot = 1
        self.ktop = Nz - 1           
        self.Nzi  = Nz - 2

        DX = dxc[:, None, None]; DY = dyc[None, :, None]; DZ = dzc[None, None, :]
        vol = DX * DY * DZ
        self.vol_i = cp.ascontiguousarray(
            vol[:, :, self.kbot:self.ktop].astype(self.dtype)
        )

        ep = cp.empty((Nx + 2, Ny + 2, Nz + 2), dtype=self.dtype)
        ep[1:-1, 1:-1, 1:-1] = eps
        ep[0, :, :] = ep[1, :, :]; ep[-1, :, :] = ep[-2, :, :]
        ep[:, 0, :] = ep[:, 1, :]; ep[:, -1, :] = ep[:, -2, :]
        ep[:, :, 0] = ep[:, :, 1]; ep[:, :, -1] = ep[:, :, -2]
        ec = ep[1:-1, 1:-1, 1:-1]

        def pad_face(hf, N):
            out = cp.empty(N + 1, dtype=self.dtype)
            out[1:-1] = hf; out[0] = hf[0]; out[-1] = hf[-1]
            return out

        hxp = pad_face(hxf, Nx); hyp = pad_face(hyf, Ny); hzp = pad_face(hzf, Nz)

        eps_xm = harm(ec, ep[:-2, 1:-1, 1:-1]); eps_xp = harm(ec, ep[2:, 1:-1, 1:-1])
        eps_ym = harm(ec, ep[1:-1, :-2, 1:-1]); eps_yp = harm(ec, ep[1:-1, 2:, 1:-1])
        eps_zm = harm(ec, ep[1:-1, 1:-1, :-2]); eps_zp = harm(ec, ep[1:-1, 1:-1, 2:])
        del ep, ec

        area_x = DY * DZ; area_y = DX * DZ; area_z = DX * DY
        cxm_full = eps_xm * area_x / hxp[:Nx, None, None]
        cxp_full = eps_xp * area_x / hxp[1:Nx + 1, None, None]
        cym_full = eps_ym * area_y / hyp[None, :Ny, None]
        cyp_full = eps_yp * area_y / hyp[None, 1:Ny + 1, None]
        czm_full = eps_zm * area_z / hzp[None, None, :Nz]
        czp_full = eps_zp * area_z / hzp[None, None, 1:Nz + 1]

        # Lateral Neumann: zero coupling to non-existent outside cells
        cxm_full[0, :, :] = 0.0; cxp_full[-1, :, :] = 0.0
        cym_full[:, 0, :] = 0.0; cyp_full[:, -1, :] = 0.0

        # Slice to interior z-range
        self.cxm = cp.ascontiguousarray(cxm_full[:, :, self.kbot:self.ktop])
        self.cxp = cp.ascontiguousarray(cxp_full[:, :, self.kbot:self.ktop])
        self.cym = cp.ascontiguousarray(cym_full[:, :, self.kbot:self.ktop])
        self.cyp = cp.ascontiguousarray(cyp_full[:, :, self.kbot:self.ktop])
        self.czm = cp.ascontiguousarray(czm_full[:, :, self.kbot:self.ktop])
        self.czp = cp.ascontiguousarray(czp_full[:, :, self.kbot:self.ktop])

        # Diagonal: includes the Dirichlet-face couplings 
        self.diag = (self.cxm + self.cxp + self.cym + self.cyp + self.czm + self.czp)

        self.czm_off = self.czm.copy()
        self.czp_off = self.czp.copy()
        self.czm_off[:, :, 0]  = 0.0     # interior k=kbot has no lower interior neighbor
        self.czp_off[:, :, -1] = 0.0     # interior k=ktop-1 has no upper interior neighbor

        # Tridiagonal preconditioner: use the off-diagonal versions
        self.tri_a = cp.ascontiguousarray(-self.czm_off)
        self.tri_d = cp.ascontiguousarray(self.diag)
        self.tri_c = cp.ascontiguousarray(-self.czp_off)

        # x-line and y-line tridiagonals share the same self.diag
        self.tri_a_x = cp.ascontiguousarray(-self.cxm)
        self.tri_c_x = cp.ascontiguousarray(-self.cxp)
        self.tri_a_y = cp.ascontiguousarray(-self.cym)
        self.tri_c_y = cp.ascontiguousarray(-self.cyp)

        shape_i = (Nx, Ny, self.Nzi)
        self.x = cp.zeros(shape_i, dtype=self.dtype)
        self.r = cp.zeros(shape_i, dtype=self.dtype)
        self.z = cp.zeros(shape_i, dtype=self.dtype)
        self.p = cp.zeros(shape_i, dtype=self.dtype)
        self.Ap = cp.zeros(shape_i, dtype=self.dtype)
        self.wcp = cp.zeros(shape_i, dtype=self.dtype)
        self.wdp = cp.zeros(shape_i, dtype=self.dtype)
        self.phi_out = cp.zeros((Nx, Ny, Nz), dtype=self.dtype)

        self.wcp_x = cp.zeros_like(self.x)
        self.wdp_x = cp.zeros_like(self.x)
        self.wcp_y = cp.zeros_like(self.x)
        self.wdp_y = cp.zeros_like(self.x)
        self.z_tmp = cp.zeros_like(self.x)

    def apply_A(self, p, out):
        cp.multiply(self.diag, p, out=out)
        out[1:, :, :] -= self.cxm[1:, :, :] * p[:-1, :, :]
        out[:-1, :, :] -= self.cxp[:-1, :, :] * p[1:, :, :]
        out[:, 1:, :] -= self.cym[:, 1:, :] * p[:, :-1, :]
        out[:, :-1, :] -= self.cyp[:, :-1, :] * p[:, 1:, :]
        out[:, :, 1:] -= self.czm[:, :, 1:] * p[:, :, :-1]
        out[:, :, :-1] -= self.czp[:, :, :-1] * p[:, :, 1:]

    def precond(self, r, z_out):
        Nx, Ny, Nzi = self.x.shape

        # --- z line ---
        ker_z = get_thomas_kernel(self.dtype)
        bx, by = 16, 16
        gx = (Ny + bx - 1) // bx
        gy = (Nx + by - 1) // by
        ker_z((gx, gy), (bx, by),
              (self.tri_a, self.tri_d, self.tri_c, r, z_out,
               self.wcp, self.wdp,
               np.int32(Nx), np.int32(Ny), np.int32(Nzi)))

        # --- x line ---
        ker_x = get_thomas_x_kernel(self.dtype)
        bx, by = 16, 8
        gx = (Nzi + bx - 1) // bx
        gy = (Ny + by - 1) // by
        ker_x((gx, gy), (bx, by),
              (self.tri_a_x, self.tri_d, self.tri_c_x, r, self.z_tmp,
               self.wcp_x, self.wdp_x,
               np.int32(Nx), np.int32(Ny), np.int32(Nzi)))
        z_out += self.z_tmp

        # --- y line ---
        ker_y = get_thomas_y_kernel(self.dtype)
        bx, by = 16, 8
        gx = (Nzi + bx - 1) // bx
        gy = (Nx + by - 1) // by
        ker_y((gx, gy), (bx, by),
              (self.tri_a_y, self.tri_d, self.tri_c_y, r, self.z_tmp,
               self.wcp_y, self.wdp_y,
               np.int32(Nx), np.int32(Ny), np.int32(Nzi)))
        z_out += self.z_tmp

        # Average → symmetric, SPD
        z_out *= self.dtype.type(1.0 / 3.0)
        
    def solve(self, rho, phi_init, rtol=1e-7, maxit=2000, cold=False, check_every=10):
        b = rho[:, :, self.kbot:self.ktop] * self.vol_i
        if cold:
            self.x.fill(0.0)
        else:
            self.x[...] = phi_init[:, :, :self.Nzi]
        self.apply_A(self.x, self.Ap)
        self.r[...] = b - self.Ap

        def fused_two(u1, v1, u2, v2):
            s1 = cp.sum(u1.astype(cp.float64) * v1.astype(cp.float64))
            s2 = cp.sum(u2.astype(cp.float64) * v2.astype(cp.float64))
            return cp.asnumpy(cp.stack([s1, s2]))

        init = cp.asnumpy(cp.stack([
            cp.sum(b.astype(cp.float64) ** 2),
            cp.sum(self.r.astype(cp.float64) ** 2),
        ]))
        normb  = float(np.sqrt(init[0]))
        rnorm0 = float(np.sqrt(init[1]))
        if normb < 1e-300:
            normb = 1.0

        if not hasattr(self, "_diag_printed"):
            print(f"PCG diag: |r0|/|b|={rnorm0/normb:.3e}  "
                  f"diag range=[{float(cp.min(self.diag).get()):.3e}, "
                  f"{float(cp.max(self.diag).get()):.3e}]  "
                  f"shape={self.x.shape}", flush=True)
            self._diag_printed = True
            
        if (rnorm0 / normb) < rtol or rnorm0 < 1e-30:
            self.phi_out[:, :, :self.Nzi] = self.x
            self.phi_out[:, :, 0]  = 0.0
            self.phi_out[:, :, -1] = 0.0
            return self.phi_out, 0

        self.precond(self.r, self.z)
        self.p[...] = self.z

        rz = float(cp.asnumpy(cp.sum(self.r.astype(cp.float64) * self.z.astype(cp.float64))))

        converged = False
        it_final = maxit
        for it in range(1, maxit + 1):
            self.apply_A(self.p, self.Ap)

            pAp = float(cp.asnumpy(cp.sum(self.p.astype(cp.float64) * self.Ap.astype(cp.float64))))
            if not np.isfinite(pAp) or abs(pAp) < 1e-300:
                it_final = maxit; break

            alpha = rz / pAp
            self.x += self.dtype.type(alpha) * self.p
            self.r -= self.dtype.type(alpha) * self.Ap

            do_check = (it % check_every == 0 or it == maxit)

            self.precond(self.r, self.z)

            if do_check:
                vals = fused_two(self.r, self.z, self.r, self.r)
                rz_new = float(vals[0])
                rel = float(np.sqrt(vals[1])) / normb
                if not np.isfinite(rel) or not np.isfinite(rz_new):
                    it_final = maxit; break
                if rel < rtol:
                    converged = True; it_final = it; break
            else:
                rz_new = float(cp.asnumpy(cp.sum(self.r.astype(cp.float64) * self.z.astype(cp.float64))))
                if not np.isfinite(rz_new):
                    it_final = maxit; break

            beta = rz_new / (rz + 1e-300)
            self.p *= self.dtype.type(beta)
            self.p += self.z
            rz = rz_new

        self.phi_out[:, :, self.kbot:self.ktop] = self.x
        self.phi_out[:, :, 0]  = 0.0   # Dirichlet bottom
        self.phi_out[:, :, -1] = 0.0   # Dirichlet top
        return self.phi_out, (it_final if converged else maxit)
      

# diagnostic Bernoulli
def bern_cp(x):
        x = cp.asarray(x)
        ax = cp.abs(x)
        small = ax < 1.0e-4
        large_pos = x > 80.0
        large_neg = x < -80.0
        mid = ~(small | large_pos | large_neg)
        out = cp.empty_like(x)
        if bool(cp.any(small).get()):
            xs = x[small]; x2 = xs * xs
            out[small] = 1.0 - 0.5 * xs + x2 / 12.0 - x2 * x2 / 720.0
        if bool(cp.any(large_pos).get()):
            xp = x[large_pos]; out[large_pos] = xp * cp.exp(-xp)
        if bool(cp.any(large_neg).get()):
            xn = x[large_neg]; out[large_neg] = -xn
        if bool(cp.any(mid).get()):
            xm = x[mid]; out[mid] = xm / cp.expm1(xm)
        return out

# =====================================================================
# Diagnostics reusable in time loop
# =====================================================================
def grad_phi(phi, x, y, z, hxf, hyf, hzf):
    Ex = cp.zeros_like(phi)
    Ey = cp.zeros_like(phi)
    Ez = cp.zeros_like(phi)

    Ex[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (x[2:, None, None] - x[:-2, None, None])
    Ey[:, 1:-1, :] = -(phi[:, 2:, :] - phi[:, :-2, :]) / (y[None, 2:, None] - y[None, :-2, None])
    Ez[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (z[None, None, 2:] - z[None, None, :-2])

    Ex[0, :, :] = -(phi[1, :, :] - phi[0, :, :]) / hxf[0]
    Ex[-1, :, :] = -(phi[-1, :, :] - phi[-2, :, :]) / hxf[-1]

    Ey[:, 0, :] = -(phi[:, 1, :] - phi[:, 0, :]) / hyf[0]
    Ey[:, -1, :] = -(phi[:, -1, :] - phi[:, -2, :]) / hyf[-1]

    Ez[:, :, 0] = -(phi[:, :, 1] - phi[:, :, 0]) / hzf[0]
    Ez[:, :, -1] = -(phi[:, :, -1] - phi[:, :, -2]) / hzf[-1]

    return Ex, Ey, Ez
