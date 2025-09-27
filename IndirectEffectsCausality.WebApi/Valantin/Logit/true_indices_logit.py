
import numpy as np
from scipy.special import expit as plogis
from scipy.optimize import root

# -----------------------
# SETTINGS
# -----------------------
np.random.seed(123)
N = 10**6   # שים לב: אם N=1e7 זה עדיין כבד. התחל ב-1e6 או 1e5 לבדיקה.
# N = 10**5

# -----------------------
# SIMULATE DATA
# -----------------------
L = np.random.normal(0.5, 0.1, N)
A = np.random.binomial(1, plogis(2 - 3 * L))
M = np.random.binomial(1, plogis(-1 + 3 * A - 2 * L))
Y = np.random.binomial(1, plogis(-1 + 1.5 * A + 1.5 * M - 2 * L))

# assign to short names
y = Y
a = A
m = M
l = L

# -----------------------
# PRECOMPUTE DESIGN VECTORS
# -----------------------
# outcome linear predictor uses [1, a, m, l]
X_out_obs = np.column_stack([np.ones(N), a, m, l])

# several fixed combinations used in qvec_sum:
# (a=0,m=0), (a=0,m=1), (a=1,m=0), (a=1,m=1)
X_out_a0m0 = np.column_stack([np.ones(N), np.zeros(N), np.zeros(N), l])
X_out_a0m1 = np.column_stack([np.ones(N), np.zeros(N), np.ones(N),  l])
X_out_a1m0 = np.column_stack([np.ones(N), np.ones(N),  np.zeros(N), l])
X_out_a1m1 = np.column_stack([np.ones(N), np.ones(N),  np.ones(N),  l])

# mediator linear predictor uses [1, a, l]  (gam0 + gama*a + gaml*l)
X_med_obs = np.column_stack([np.ones(N), a, l])
X_med_a1 = np.column_stack([np.ones(N), np.ones(N), l])  # a=1
X_med_a0 = np.column_stack([np.ones(N), np.zeros(N), l]) # a=0

# For speed: precompute some indicator arrays used in sums
mask_a1 = (a == 1)
mask_a0 = (a == 0)

# -----------------------
# TRUE INDICES (monte carlo analytic parts)
# -----------------------
def pimL_fn(Lvals):
    return plogis(-1 + 3*1 - 2*Lvals) - plogis(-1 + 3*0 - 2*Lvals)

def pioML_fn(Lvals):
    return plogis(-1 + 1.5*0 + 1.5*1 - 2*Lvals) - plogis(-1 + 1.5*0 + 1.5*0 - 2*Lvals)

p_i1 = np.mean(pimL_fn(L[a==1])) * np.mean(pioML_fn(L[a==1]))
p_i0 = np.mean(pimL_fn(L[a==0])) * np.mean(pioML_fn(L[a==0]))
p_i  = p_i0 * np.mean(a==0) + p_i1 * np.mean(a==1)

IEIN = 1 / p_i1
INNE = 1 / p_i0
INNT = 1 / p_i

def pioAM0L_fn(Lvals):
    return plogis(-1 + 1.5*1 + 1.5*0 - 2*Lvals) - plogis(-1 + 1.5*0 + 1.5*0 - 2*Lvals)

def pioAM1L_fn(Lvals):
    return plogis(-1 + 1.5*1 + 1.5*1 - 2*Lvals) - plogis(-1 + 1.5*0 + 1.5*1 - 2*Lvals)

p_d1 = (np.mean(pioAM0L_fn(L[a==1])) * (1 - np.mean(plogis(-1 + 3*1 - 2*L[a==1]))) +
        np.mean(pioAM1L_fn(L[a==1])) * np.mean(plogis(-1 + 3*1 - 2*L[a==1])))
p_d0 = (np.mean(pioAM0L_fn(L[a==0])) * (1 - np.mean(plogis(-1 + 3*1 - 2*L[a==0]))) +
        np.mean(pioAM1L_fn(L[a==0])) * np.mean(plogis(-1 + 3*1 - 2*L[a==0])))
p_d = p_d0 * np.mean(a==0) + p_d1 * np.mean(a==1)

DEIN = 1 / p_d1
DNNE = 1 / p_d0
DNNT = 1 / p_d

EIN = 1 / (p_i1 + p_d1)
NNE = 1 / (p_i0 + p_d0)
NNT = 1 / (p_i + p_d)

true_indices = {
    'INNE': INNE, 'IEIN': IEIN, 'INNT': INNT,
    'DNNE': DNNE, 'DEIN': DEIN, 'DNNT': DNNT,
    'NNE': NNE, 'EIN': EIN, 'NNT': NNT
}
print("true indices (MC):", true_indices)

# -----------------------
# FAST vectorized qvec_sum
# -----------------------
def qvec_sum(x):
    # unpack parameters
    b0, ba, bm, bl = x[0:4]
    gam0, gama, gaml = x[4:7]
    E_M1_1, E_M1_0 = x[7:9]
    E_M0_1, E_M0_0 = x[9:11]
    E_I01I00_0, E_I01I00_1 = x[11:13]
    E_I10I00_0, E_I10I00_1 = x[13:15]
    E_I11I01_0, E_I11I01_1 = x[15:17]
    pi0, pi1, pi = x[17:20]
    pd0, pd1, pd = x[20:23]
    INNE_p, IEIN_p, INNT_p = x[23:26]
    DNNE_p, DEIN_p, DNNT_p = x[26:29]
    NNE_p, EIN_p, NNT_p = x[29:32]

    # outcome linear predictors (vectorized)
    beta_out = np.array([b0, ba, bm, bl])
    eta_out_obs = X_out_obs.dot(beta_out)       # b0 + ba*a + bm*m + bl*l
    eta_out_a0m1 = X_out_a0m1.dot(beta_out)     # b0 + ba*0 + bm*1 + bl*l
    eta_out_a0m0 = X_out_a0m0.dot(beta_out)
    eta_out_a1m0 = X_out_a1m0.dot(beta_out)
    eta_out_a1m1 = X_out_a1m1.dot(beta_out)

    # mediator linear predictors
    gamma_med = np.array([gam0, gama, gaml])
    eta_med_obs = X_med_obs.dot(gamma_med)    # gam0 + gama*a + gaml*l
    eta_med_a1 = X_med_a1.dot(gamma_med)      # gam0 + gama*1 + gaml*l
    eta_med_a0 = X_med_a0.dot(gamma_med)      # gam0 + gama*0 + gaml*l

    # apply plogis
    p_out_obs = plogis(eta_out_obs)
    p_out_a0m1 = plogis(eta_out_a0m1)
    p_out_a0m0 = plogis(eta_out_a0m0)
    p_out_a1m0 = plogis(eta_out_a1m0)
    p_out_a1m1 = plogis(eta_out_a1m1)

    p_med_obs = plogis(eta_med_obs)
    p_med_a1 = plogis(eta_med_a1)
    p_med_a0 = plogis(eta_med_a0)

    # Now compute the 32 equations (vectorized sums)
    out = np.empty(32, dtype=float)
    # 1-4 score outcome
    resid_out = y - p_out_obs
    out[0]  = resid_out.sum()
    out[1]  = (resid_out * a).sum()
    out[2]  = (resid_out * m).sum()
    out[3]  = (resid_out * l).sum()
    # 5-7 score mediator
    resid_med = m - p_med_obs
    out[4]  = resid_med.sum()
    out[5]  = (resid_med * a).sum()
    out[6]  = (resid_med * l).sum()
    # 8-11 Mediator potential outcomes
    out[7]  = ((p_med_a1 - E_M1_1) * a).sum()
    out[8]  = ((p_med_a1 - E_M1_0) * (~a.astype(bool))).sum()  # (1-a)
    out[9]  = ((p_med_a0 - E_M0_1) * a).sum()
    out[10] = ((p_med_a0 - E_M0_0) * (~a.astype(bool))).sum()
    # 12-17 mediator/exposure effects on outcome
    out[11] = ((p_out_a0m1 - p_out_a0m0 - E_I01I00_1) * a).sum()
    out[12] = ((p_out_a0m1 - p_out_a0m0 - E_I01I00_0) * (~a.astype(bool))).sum()
    out[13] = ((p_out_a1m0 - p_out_a0m0 - E_I10I00_1) * a).sum()
    out[14] = ((p_out_a1m0 - p_out_a0m0 - E_I10I00_0) * (~a.astype(bool))).sum()
    out[15] = ((p_out_a1m1 - p_out_a0m1 - E_I11I01_1) * a).sum()
    out[16] = ((p_out_a1m1 - p_out_a0m1 - E_I11I01_0) * (~a.astype(bool))).sum()
    # 18-20 Indirect effects
    out[17] = (((E_M1_1 - E_M0_1) * E_I01I00_1 - pi1) * a).sum()
    out[18] = (((E_M1_0 - E_M0_0) * E_I01I00_0 - pi0) * (~a.astype(bool))).sum()
    out[19] = (pi0 * (~a.astype(bool)) + pi1 * a - pi).sum()
    # 21-23 Direct effects
    out[20] = (((E_I10I00_1*(1-E_M1_1) + E_I11I01_1*E_M1_1 - pd1) * a)).sum()
    out[21] = (((E_I10I00_0*(1-E_M1_0) + E_I11I01_0*E_M1_0 - pd0) * (~a.astype(bool)))).sum()
    out[22] = (pd0 * (~a.astype(bool)) + pd1 * a - pd).sum()
    # 24-26 Indirect indices
    out[23] = (1/pi0 - INNE_p).sum() if np.ndim(1/pi0) > 0 else (1/pi0 - INNE_p)  # keep scalar
    out[24] = (1/pi1 - IEIN_p).sum() if np.ndim(1/pi1) > 0 else (1/pi1 - IEIN_p)
    out[25] = (1/pi - INNT_p).sum()   if np.ndim(1/pi) > 0  else (1/pi - INNT_p)
    # 27-29 Direct indices
    out[26] = (1/pd0 - DNNE_p).sum()  if np.ndim(1/pd0) > 0 else (1/pd0 - DNNE_p)
    out[27] = (1/pd1 - DEIN_p).sum()  if np.ndim(1/pd1) > 0 else (1/pd1 - DEIN_p)
    out[28] = (1/pd - DNNT_p).sum()   if np.ndim(1/pd) > 0  else (1/pd - DNNT_p)
    # 30-32 marginal indices
    out[29] = (1/(pi0+pd0) - NNE_p).sum() if np.ndim(pi0+pd0) > 0 else (1/(pi0+pd0) - NNE_p)
    out[30] = (1/(pi1+pd1) - EIN_p).sum() if np.ndim(pi1+pd1) > 0 else (1/(pi1+pd1) - EIN_p)
    out[31] = (1/(pi+pd) - NNT_p).sum()   if np.ndim(pi+pd) > 0  else (1/(pi+pd) - NNT_p)

    # Note: the .sum() wrapping for scalars is harmless; these entries are scalars
    return out

# -----------------------
# SOLVE SYSTEM
# -----------------------
x0 = np.ones(32)
# use method 'hybr' (default) or 'lm' if you prefer Levenberg-Marquardt
sol = root(qvec_sum, x0, method='hybr', options={'maxfev': 2000})
print("root success:", sol.success, "message:", sol.message)
true_ind = sol.x[23:32]  # INNE -> NNT (note python 0-based indexing)
print("estimated true indices from solution (INNE..NNT):", true_ind)