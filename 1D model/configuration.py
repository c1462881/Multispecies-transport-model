from pathlib import Path
from datetime import datetime

# ---------- Output ----------
OUTPUT_DIR = Path(fr"1D_{datetime.now().strftime('%H%M')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FORMATS = ['svg']

SNAP_FRACS        = [0.0, 0.001, 0.01, 0.1, 1.0]
PLOT_PI_SNAPSHOTS = True     

# ---------- Simulation ----------
SIM_T     = 2000000
SIM_L     = 50
EXT_L     = 500
FRAC_NEAR = 0.20
FRAC_FAR  = 0.80
N_GRID    = 2000
N_T_EVAL  = 300

# ---------- Physical constants ----------
EPS_0 = 8.854e-12
K_B   = 1.381e-23
N_A   = 6.022e23
E_C   = 1.602e-19
F_C   = 96485.0
R_GAS = 8.314
T_K   = 310.0
V_T   = R_GAS * T_K / F_C

# ---------- Material ----------
EPS_R_W = 80.0
EPS_R_M = 2.0
ETA_W   = 6.91e-4
ETA_C   = 1.2e-3
ETA_M   = 0.01

# ---------- Membrane ----------
D_MEM  = 4e-9
D_HALF = D_MEM / 2.0

# ---------- Ions (mM = mol/m³) ----------
ION_SPECIES = [
    {'name': 'Na+',  'z':  1, 'C_bulk': 145.0, 'C_cyto':  10.0,    'D0': 1.33e-9},
    {'name': 'K+',   'z':  1, 'C_bulk':   5.0, 'C_cyto': 140.0,    'D0': 1.96e-9},
    {'name': 'Cl-',  'z': -1, 'C_bulk': 110.0, 'C_cyto':  10.0,    'D0': 2.03e-9},
    {'name': 'Ca2+', 'z':  2, 'C_bulk':   1.8, 'C_cyto':   0.0001, 'D0': 0.79e-9},
]

# ---------- Protein (in mM = mol/m³) ----------
# C_PROTEIN_CYTO = 1e24 / N_A
Z_PROTEIN      = -10
A_P            = 2.5e-9
ALPHA_BLOCK    = 1e-20
R_H            = 1.3 * A_P

# ---------- Background species ----------
Z_X = -1    
# Y is neutral (z = 0)

# ---------- Resting potential setup ----------
V_REST_MV       = -75.0    # φ_cyto − φ_bulk target, mV
N_DEBYE_LAYERS  = 6        # ion-perturbation depth in units of λ_D
V_REST_TOL_MV   = 0.5      # acceptance tolerance
V_REST_MAX_ITER = 12

# ---------- Colors: spatial ----------
SPATIAL_COLORS = {
    'Na+':     'crimson',
    'K+':      'forestgreen',
    'Cl-':     'royalblue',
    'Ca2+':    'darkorange',
    'protein': 'saddlebrown',
    'phi':     'purple',
    'Pi':      'darkgreen',
}

# ---------- Colors: temporal (cyto don = gray solid, bulk don = gray dotted) ----------
TEMPORAL_STYLES = {
    'far_in':      dict(color='red',   ls=':',  lw=1.5),
    'near_in':     dict(color='red',   ls='-',  lw=1.8),
    'far_out':     dict(color='blue',  ls=':',  lw=1.5),
    'near_out':    dict(color='blue',  ls='-',  lw=1.8),
    'don_cyto':    dict(color='gray',  ls='-',  lw=1.0),
    'don_bulk':    dict(color='gray',  ls=':',  lw=1.0),
}

# ---------- Solver ----------
SOLVER_METHOD = 'RK45'   # 'BDF' | 'Radau' | 'RK45'
JAC_BW        = 12      # Jacobian local half-bandwidth [grid cells]; >~ few*lambda_D/dz