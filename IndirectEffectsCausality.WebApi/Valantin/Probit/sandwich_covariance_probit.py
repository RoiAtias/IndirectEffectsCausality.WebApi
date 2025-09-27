
import numpy as np
import pandas as pd
from scipy.stats import norm, logistic
from scipy.optimize import root, approx_fprime

# פונקציות שאתה צריך להגדיר (כמו ב־R):
# qvec2(x) -> np.array([...])
# qvec2_sum(x) -> np.array([...])

# N_list = [200, 400, 800, 1600]
N_list = [400]
# K = 100
K = 10
np.random.seed(123)

all_ci = {}
all_est = {}

for N in N_list:
    mat_est = pd.DataFrame(np.nan, index=range(K), columns=[
        "INNE","IEIN","INNT","DNNE","DEIN","DNNT","NNE","EIN","NNT"
    ])

    ci_cols = []
    for name in ["INNE","IEIN","INNT","DNNE","DEIN","DNNT","NNE","EIN","NNT"]:
        ci_cols += [f"{name}_L", f"{name}_U"]
    mat_ci = pd.DataFrame(np.nan, index=range(K), columns=ci_cols)

    for k in range(K):
        print(N, k+1)

        L = np.random.normal(0.5, 0.1, N)
        A = np.random.binomial(1, logistic.cdf(2 - 3 * L))
        M = np.random.binomial(1, norm.cdf(-1 + 3 * A - 2 * L))
        Y = np.random.binomial(1, norm.cdf(-1 + 1.5 * A + 1.5 * M - 2 * L))

        # פתרון המערכת
        sol = root(lambda x: qvec2_sum(x, Y, A, M, L), np.ones(32)).x
        mat_est.iloc[k,:] = sol[23:32]  # ב־R היו 24:32 כי אינדקסים שם מ־1

        # Bread
        bread_mat = np.zeros((32,32))
        for y,a,m,l in zip(Y,A,M,L):
            jac = -approx_fprime(sol, lambda x: qvec2(x, y,a,m,l), epsilon=1e-8)
            bread_mat += jac
        bread_mat /= len(Y)

        if np.linalg.cond(bread_mat) > 1/np.finfo(bread_mat.dtype).eps:
            print("Bread matrix singular, skipping")
            continue

        inv_a = np.linalg.inv(bread_mat)

        # Meat
        meat_mat = np.zeros((32,32))
        for y,a,m,l in zip(Y,A,M,L):
            qv = qvec2(sol, y,a,m,l)
            meat_mat += np.outer(qv, qv)
        meat_mat /= len(Y)

        sand_mat = inv_a @ meat_mat @ inv_a.T / len(Y)

        # CI
        cis = []
        for idx in range(23,32):
            if sol[idx] >= 1:
                low = max(sol[idx] - 1.96*np.sqrt(sand_mat[idx,idx]), 1)
                high = sol[idx] + 1.96*np.sqrt(sand_mat[idx,idx])
                cis += [low, high]
            else:
                cis += [np.inf, np.inf]
        mat_ci.iloc[k,:] = cis

    all_ci[str(N)] = mat_ci
    all_est[str(N)] = mat_est

# true values
true_values = {
  "INNE":4.493874,
  "IEIN":4.176306,
  "INNT":4.291575,
  "DNNE":2.062010,
  "DEIN":2.056743,
  "DNNT":2.058742,
  "NNE":1.413450,
  "EIN":1.378072,
  "NNT":1.391308
}

coverage_df = pd.DataFrame({"Index": list(true_values.keys())})
for n in all_ci.keys():
    mat = all_ci[n]
    coverage = []
    for name,val in true_values.items():
        lower = mat[f"{name}_L"]
        upper = mat[f"{name}_U"]
        cov = np.mean((val >= lower) & (val <= upper))
        coverage.append(cov)
    coverage_df[f"N{n}"] = coverage

print(coverage_df)

# שמירת קבצים
ci_long = pd.concat([
    df.assign(n=int(n), iteration=np.arange(1, len(df)+1))
    for n,df in all_ci.items()
])
ci_long.to_csv("all_ci_long_probit.csv", index=False)

est_long = pd.concat([
    df.assign(n=int(n), iteration=np.arange(1, len(df)+1))
    for n,df in all_est.items()
])
est_long.to_csv("all_est_long_probit.csv", index=False)