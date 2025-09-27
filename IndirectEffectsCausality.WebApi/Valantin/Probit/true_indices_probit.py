
import numpy as np
from scipy.stats import norm, logistic
from scipy.optimize import root

# ---------------------------
# חלק 1: True indices Monte Carlo
# ---------------------------
np.random.seed(123)
N = 10**6   # שים לב - ב-R השתמשת 10^7, כאן הקטנתי כדי שירוץ מהר יותר

L = np.random.normal(0.5, 0.1, N)
A = np.random.binomial(1, logistic.cdf(2 - 3 * L))
M = np.random.binomial(1, norm.cdf(-1 + 3 * A - 2 * L))
Y = np.random.binomial(1, norm.cdf(-1 + 1.5*A + 1.5*M - 2*L))

def pimL(L):
    return norm.cdf(-1 + 3*1 - 2*L) - norm.cdf(-1 + 3*0 - 2*L)

def pioML(L):
    return norm.cdf(-1 + 1.5*0 + 1.5*1 - 2*L) - norm.cdf(-1 + 1.5*0 + 1.5*0 - 2*L)

# Indirect effects
p_i1 = np.mean(pimL(L[A==1])) * np.mean(pioML(L[A==1]))
IEIN = 1 / p_i1

p_i0 = np.mean(pimL(L[A==0])) * np.mean(pioML(L[A==0]))
INNE = 1 / p_i0

p_i = p_i0 * np.mean(A==0) + p_i1 * np.mean(A==1)
INNT = 1 / p_i

# Direct effects
def pioAM0L(L):  # M=0
    return norm.cdf(-1 + 1.5*1 + 1.5*0 - 2*L) - norm.cdf(-1 + 1.5*0 + 1.5*0 - 2*L)

def pioAM1L(L):  # M=1
    return norm.cdf(-1 + 1.5*1 + 1.5*1 - 2*L) - norm.cdf(-1 + 1.5*0 + 1.5*1 - 2*L)

p_d1 = (np.mean(pioAM0L(L[A==1])) * (1 - np.mean(norm.cdf(-1 + 3*1 - 2*L[A==1]))) +
        np.mean(pioAM1L(L[A==1])) * np.mean(norm.cdf(-1 + 3*1 - 2*L[A==1])))
DEIN = 1 / p_d1

p_d0 = (np.mean(pioAM0L(L[A==0])) * (1 - np.mean(norm.cdf(-1 + 3*1 - 2*L[A==0]))) +
        np.mean(pioAM1L(L[A==0])) * np.mean(norm.cdf(-1 + 3*1 - 2*L[A==0])))
DNNE = 1 / p_d0

p_d = p_d0 * np.mean(A==0) + p_d1 * np.mean(A==1)
DNNT = 1 / p_d

# Marginal
EIN = 1 / (p_i1 + p_d1)
NNE = 1 / (p_i0 + p_d0)
NNT = 1 / (p_i + p_d)

true_indices = {
    "INNE": INNE, "IEIN": IEIN, "INNT": INNT,
    "DNNE": DNNE, "DEIN": DEIN, "DNNT": DNNT,
    "NNE": NNE, "EIN": EIN, "NNT": NNT
}
print("Monte Carlo true indices:")
print(true_indices)


# ---------------------------
# חלק 2: פתרון מערכת משוואות (root-finding)
# ---------------------------
y, a, m, l = Y, A, M, L

def qvec_sum(x):
    (b0, ba, bm, bl,
     gam0, gama, gaml,
     E_M1_1, E_M1_0,
     E_M0_1, E_M0_0,
     E_I01I00_0, E_I01I00_1,
     E_I10I00_0, E_I10I00_1,
     E_I11I01_0, E_I11I01_1,
     pi0, pi1, pi,
     pd0, pd1, pd,
     INNE, IEIN, INNT,
     DNNE, DEIN, DNNT,
     NNE, EIN, NNT) = x

    out = []
    # Outcome probit score
    xb = b0 + ba*a + bm*m + bl*l
    out.append(np.sum((y - norm.cdf(xb)) / norm.pdf(xb)))
    out.append(np.sum((y - norm.cdf(xb)) / norm.pdf(xb) * a))
    out.append(np.sum((y - norm.cdf(xb)) / norm.pdf(xb) * m))
    out.append(np.sum((y - norm.cdf(xb)) / norm.pdf(xb) * l))

    # Mediator probit score
    xm = gam0 + gama*a + gaml*l
    out.append(np.sum((m - norm.cdf(xm)) / norm.pdf(xm)))
    out.append(np.sum((m - norm.cdf(xm)) / norm.pdf(xm) * a))
    out.append(np.sum((m - norm.cdf(xm)) / norm.pdf(xm) * l))

    # Mediator potential outcomes
    out.append(np.sum((norm.cdf(gam0 + gama*1 + gaml*l) - E_M1_1) * a))
    out.append(np.sum((norm.cdf(gam0 + gama*1 + gaml*l) - E_M1_0) * (1-a)))
    out.append(np.sum((norm.cdf(gam0 + gama*0 + gaml*l) - E_M0_1) * a))
    out.append(np.sum((norm.cdf(gam0 + gama*0 + gaml*l) - E_M0_0) * (1-a)))

    # mediator effect on the outcome
    out.append(np.sum((norm.cdf(b0 + ba*0 + bm*1 + bl*l) - norm.cdf(b0 + ba*0 + bm*0 + bl*l) - E_I01I00_1) * a))
    out.append(np.sum((norm.cdf(b0 + ba*0 + bm*1 + bl*l) - norm.cdf(b0 + ba*0 + bm*0 + bl*l) - E_I01I00_0) * (1-a)))

    # exposure effect on the outcome
    out.append(np.sum((norm.cdf(b0 + ba*1 + bm*0 + bl*l) - norm.cdf(b0 + ba*0 + bm*0 + bl*l) - E_I10I00_1) * a))
    out.append(np.sum((norm.cdf(b0 + ba*1 + bm*0 + bl*l) - norm.cdf(b0 + ba*0 + bm*0 + bl*l) - E_I10I00_0) * (1-a)))
    out.append(np.sum((norm.cdf(b0 + ba*1 + bm*1 + bl*l) - norm.cdf(b0 + ba*0 + bm*1 + bl*l) - E_I11I01_1) * a))
    out.append(np.sum((norm.cdf(b0 + ba*1 + bm*1 + bl*l) - norm.cdf(b0 + ba*0 + bm*1 + bl*l) - E_I11I01_0) * (1-a)))

    # Indirect effects
    out.append(np.sum(((E_M1_1 - E_M0_1) * E_I01I00_1 - pi1) * a))
    out.append(np.sum(((E_M1_0 - E_M0_0) * E_I01I00_0 - pi0) * (1-a)))
    out.append(np.sum(pi0*(1-a) + pi1*a - pi))

    # Direct effects
    out.append(np.sum((E_I10I00_1*(1 - E_M1_1) + E_I11I01_1*E_M1_1 - pd1) * a))
    out.append(np.sum((E_I10I00_0*(1 - E_M1_0) + E_I11I01_0*E_M1_0 - pd0) * (1-a)))
    out.append(np.sum(pd0*(1-a) + pd1*a - pd))

    # Indirect indices
    out.append(1/pi0 - INNE)
    out.append(1/pi1 - IEIN)
    out.append(1/pi - INNT)

    # Direct indices
    out.append(1/pd0 - DNNE)
    out.append(1/pd1 - DEIN)
    out.append(1/pd - DNNT)

    # marginal indices
    out.append(1/(pi0 + pd0) - NNE)
    out.append(1/(pi1 + pd1) - EIN)
    out.append(1/(pi + pd) - NNT)

    return np.array(out)

# פתרון המערכת
x0 = np.ones(32)
sol = root(qvec_sum, x0, method="hybr")
print("Solution success:", sol.success)
print("Estimated indices:", sol.x[23:32])  # באינדקסים 24-32 כמו ב-R