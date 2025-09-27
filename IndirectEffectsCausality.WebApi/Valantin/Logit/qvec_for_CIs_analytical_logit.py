import numpy as np
from scipy.special import expit  # logistic function
from scipy.optimize import root

# Fixed values for demonstration
y = 1
a = 1
m = 1
l = 1

def qvec(
    b0, ba, bm, bl,
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
    NNE, EIN, NNT
):
    out = np.array([
        # score outcome model
        (y - expit(b0 + ba*a + bm*m + bl*l)),
        (y - expit(b0 + ba*a + bm*m + bl*l)) * a,
        (y - expit(b0 + ba*a + bm*m + bl*l)) * m,
        (y - expit(b0 + ba*a + bm*m + bl*l)) * l,
        # score mediator model
        (m - expit(gam0 + gama*a + gaml*l)),
        (m - expit(gam0 + gama*a + gaml*l)) * a,
        (m - expit(gam0 + gama*a + gaml*l)) * l,
        # Mediator potential outcomes
        (expit(gam0 + gama*1 + gaml*l) - E_M1_1) * a,
        (expit(gam0 + gama*1 + gaml*l) - E_M1_0) * (1 - a),
        (expit(gam0 + gama*0 + gaml*l) - E_M0_1) * a,
        (expit(gam0 + gama*0 + gaml*l) - E_M0_0) * (1 - a),
        # mediator effect on the outcome
        (expit(b0 + ba*0 + bm*1 + bl*l) - expit(b0 + ba*0 + bm*0 + bl*l) - E_I01I00_1) * a,
        (expit(b0 + ba*0 + bm*1 + bl*l) - expit(b0 + ba*0 + bm*0 + bl*l) - E_I01I00_0) * (1 - a),
        # exposure effect on the outcome
        (expit(b0 + ba*1 + bm*0 + bl*l) - expit(b0 + ba*0 + bm*0 + bl*l) - E_I10I00_1) * a,
        (expit(b0 + ba*1 + bm*0 + bl*l) - expit(b0 + ba*0 + bm*0 + bl*l) - E_I10I00_0) * (1 - a),
        (expit(b0 + ba*1 + bm*1 + bl*l) - expit(b0 + ba*0 + bm*1 + bl*l) - E_I11I01_1) * a,
        (expit(b0 + ba*1 + bm*1 + bl*l) - expit(b0 + ba*0 + bm*1 + bl*l) - E_I11I01_0) * (1 - a),
        # Indirect effects
        ((E_M1_1 - E_M0_1) * E_I01I00_1  - pi1) * a,
        ((E_M1_0 - E_M0_0) * E_I01I00_0  - pi0) * (1 - a),
        (pi0 * (1 - a) + pi1 * a - pi),
        # Direct effects
        ((E_I10I00_1 * (1 - E_M1_1) + E_I11I01_1 *  E_M1_1 - pd1) * a),
        ((E_I10I00_0 * (1 - E_M1_0) + E_I11I01_0 *  E_M1_0 - pd0) * (1 - a)),
        (pd0 * (1 - a) + pd1 * a - pd),
        # Indirect indices
        (1/pi0 - INNE),
        (1/pi1 - IEIN),
        (1/pi - INNT),
        # Direct indices
        (1/pd0 - DNNE),
        (1/pd1 - DEIN),
        (1/pd - DNNT),
        # marginal indices
        (1/(pi0 + pd0) - NNE),
        (1/(pi1 + pd1) - EIN),
        (1/(pi + pd) - NNT)
    ])
    return out

def qvec2(x):
    return qvec(
        b0 = x[0], ba = x[1], bm = x[2], bl = x[3],
        gam0 = x[4], gama = x[5], gaml = x[6],
        E_M1_1 = x[7], E_M1_0 = x[8],
        E_M0_1 = x[9], E_M0_0 = x[10],
        E_I01I00_0 = x[11], E_I01I00_1 = x[12],
        E_I10I00_0 = x[13], E_I10I00_1 = x[14],
        E_I11I01_0 = x[15], E_I11I01_1 = x[16],
        pi0 = x[17], pi1 = x[18], pi = x[19],
        pd0 = x[20], pd1 = x[21], pd = x[22],
        INNE = x[23], IEIN = x[24], INNT = x[25],
        DNNE = x[26], DEIN = x[27], DNNT = x[28],
        NNE = x[29], EIN = x[30], NNT = x[31]
    )

# Example: solving the system of nonlinear equations
x0 = np.ones(32)  # initial guess
solution = root(qvec2, x0)
print("Solution vector:")
print(solution.x)

# Compute Jacobian if needed (approximate)
# from numdifftools import Jacobian
# jac = Jacobian(qvec2)(x0)
# inv_jac = np.linalg.inv(-jac)