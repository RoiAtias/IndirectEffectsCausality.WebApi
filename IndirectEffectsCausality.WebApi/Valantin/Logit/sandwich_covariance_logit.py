import numpy as np
import pandas as pd
from scipy.special import expit as plogis
from scipy.optimize import root
from numpy.linalg import inv

# -----------------------
# qvec2 + qvec2_sum (Python version)
# -----------------------
def qvec2(x, y, a, m, l):
    b0, ba, bm, bl = x[0:4]
    gam0, gama, gaml = x[4:7]
    E_M1_1, E_M1_0 = x[7:9]
    E_M0_1, E_M0_0 = x[9:11]
    E_I01I00_0, E_I01I00_1 = x[11:13]
    E_I10I00_0, E_I10I00_1 = x[13:15]
    E_I11I01_0, E_I11I01_1 = x[15:17]
    pi0, pi1, pi = x[17:20]
    pd0, pd1, pd = x[20:23]
    INNE, IEIN, INNT = x[23:26]
    DNNE, DEIN, DNNT = x[26:29]
    NNE, EIN, NNT = x[29:32]

    out = np.array([
        (y - plogis(b0 + ba*a + bm*m + bl*l)),
        (y - plogis(b0 + ba*a + bm*m + bl*l)) * a,
        (y - plogis(b0 + ba*a + bm*m + bl*l)) * m,
        (y - plogis(b0 + ba*a + bm*m + bl*l)) * l,
        (m - plogis(gam0 + gama*a + gaml*l)),
        (m - plogis(gam0 + gama*a + gaml*l)) * a,
        (m - plogis(gam0 + gama*a + gaml*l)) * l,
        (plogis(gam0 + gama*1 + gaml*l) - E_M1_1) * a,
        (plogis(gam0 + gama*1 + gaml*l) - E_M1_0) * (1-a),
        (plogis(gam0 + gama*0 + gaml*l) - E_M0_1) * a,
        (plogis(gam0 + gama*0 + gaml*l) - E_M0_0) * (1-a),
        (plogis(b0 + ba*0 + bm*1 + bl*l) - plogis(b0 + ba*0 + bm*0 + bl*l) - E_I01I00_1) * a,
        (plogis(b0 + ba*0 + bm*1 + bl*l) - plogis(b0 + ba*0 + bm*0 + bl*l) - E_I01I00_0) * (1-a),
        (plogis(b0 + ba*1 + bm*0 + bl*l) - plogis(b0 + ba*0 + bm*0 + bl*l) - E_I10I00_1) * a,
        (plogis(b0 + ba*1 + bm*0 + bl*l) - plogis(b0 + ba*0 + bm*0 + bl*l) - E_I10I00_0) * (1-a),
        (plogis(b0 + ba*1 + bm*1 + bl*l) - plogis(b0 + ba*0 + bm*1 + bl*l) - E_I11I01_1) * a,
        (plogis(b0 + ba*1 + bm*1 + bl*l) - plogis(b0 + ba*0 + bm*1 + bl*l) - E_I11I01_0) * (1-a),
        ((E_M1_1 - E_M0_1) * E_I01I00_1 - pi1) * a,
        ((E_M1_0 - E_M0_0) * E_I01I00_0 - pi0) * (1-a),
        (pi0*(1-a) + pi1*a - pi),
        (E_I10I00_1*(1-E_M1_1) + E_I11I01_1*E_M1_1 - pd1) * a,
        (E_I10I00_0*(1-E_M1_0) + E_I11I01_0*E_M1_0 - pd0) * (1-a),
        (pd0*(1-a) + pd1*a - pd),
        (1/pi0 - INNE),
        (1/pi1 - IEIN),
        (1/pi - INNT),
        (1/pd0 - DNNE),
        (1/pd1 - DEIN),
        (1/pd - DNNT),
        (1/(pi0+pd0) - NNE),
        (1/(pi1+pd1) - EIN),
        (1/(pi+pd) - NNT)
    ])
    return out


def qvec2_sum(x, Y, A, M, L):
    return np.sum([qvec2(x, Y[i], A[i], M[i], L[i]) for i in range(len(Y))], axis=0)


# -----------------------
# Simulation parameters
# -----------------------
# N_vals = [200, 400, 800, 1600]
N_vals = [400]
K = 10
np.random.seed(123)

all_ci = {}
all_est = {}

for n in N_vals:
    mat_est = np.full((K, 9), np.nan)
    mat_ci = np.full((K, 18), np.nan)

    colnames_ci = ["INNE_L","INNE_U","IEIN_L","IEIN_U","INNT_L","INNT_U",
                   "DNNE_L","DNNE_U","DEIN_L","DEIN_U","DNNT_L","DNNT_U",
                   "NNE_L","NNE_U","EIN_L","EIN_U","NNT_L","NNT_U"]

    for k in range(K):
        print(f"n={n}, k={k+1}")

        L = np.random.normal(0.5, 0.1, n)
        A = np.random.binomial(1, plogis(2 - 3*L))
        M = np.random.binomial(1, plogis(-1 + 3*A - 2*L))
        Y = np.random.binomial(1, plogis(-1 + 1.5*A + 1.5*M - 2*L))

        # Solve system
        x0 = np.ones(32)
        sol = root(lambda x: qvec2_sum(x, Y, A, M, L), x0).x
        mat_est[k, :] = sol[23:32]  # save indices

        # BREAD
        bread = np.zeros((32, 32))
        for i in range(n):
            eps = 1e-6
            jac = np.zeros((32, 32))
            f0 = qvec2(sol, Y[i], A[i], M[i], L[i])
            for j in range(32):
                x_eps = sol.copy()
                x_eps[j] += eps
                f1 = qvec2(x_eps, Y[i], A[i], M[i], L[i])
                jac[:, j] = (f1 - f0) / eps
            bread += -jac
        bread /= n

        try:
            inv_a = inv(bread)
        except np.linalg.LinAlgError:
            continue

        # MEAT
        meat = np.zeros((32, 32))
        for i in range(n):
            f = qvec2(sol, Y[i], A[i], M[i], L[i])
            meat += np.outer(f, f)
        meat /= n

        # SANDWICH
        sand = (1/n) * inv_a @ meat @ inv_a.T

        ci = []
        for idx in range(23, 32):
            est = sol[idx]
            se = np.sqrt(sand[idx, idx])
            if est >= 1:
                ci.extend([max(est - 1.96*se, 1), est + 1.96*se])
            else:
                ci.extend([np.inf, np.inf])
        mat_ci[k, :] = ci

    all_ci[str(n)] = pd.DataFrame(mat_ci, columns=colnames_ci)
    all_est[str(n)] = pd.DataFrame(mat_est, columns=["INNE","IEIN","INNT","DNNE","DEIN","DNNT","NNE","EIN","NNT"])

# -----------------------
# Coverage rates
# -----------------------
true_values = {
    "INNE": 6.525974, "IEIN": 6.282517, "INNT": 6.372880,
    "DNNE": 3.081796, "DEIN": 3.066874, "DNNT": 3.072529,
    "NNE": 2.093277, "EIN": 2.060850, "NNT": 2.073056
}

coverage_df = pd.DataFrame({"Index": list(true_values.keys())})

for n in all_ci.keys():
    mat = all_ci[n]
    coverage = []
    for name in true_values:
        coverage.append(np.mean((true_values[name] >= mat[f"{name}_L"]) & (true_values[name] <= mat[f"{name}_U"])))
    coverage_df[f"N{n}"] = coverage

print(coverage_df)

# -----------------------
# Save results
# -----------------------
ci_long = pd.concat([df.assign(n=int(n), iteration=df.index+1) for n, df in all_ci.items()])
est_long = pd.concat([df.assign(n=int(n), iteration=df.index+1) for n, df in all_est.items()])

ci_long.to_csv("all_ci_long_logit.csv", index=False)
est_long.to_csv("all_est_long_logit.csv", index=False)