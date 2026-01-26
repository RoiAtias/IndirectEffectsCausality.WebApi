
import numpy as np
from scipy.stats import norm
from scipy.optimize import root

# n = 32 sized system as in R
# norm.cdf == pnorm, norm.pdf == dnorm

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
    NNE, EIN, NNT,
    y=1, a=1, m=1, l=1
):
    # convenience: linear predictors for outcome and mediator
    eta_out = b0 + ba * a + bm * m + bl * l
    eta_med = gam0 + gama * a + gaml * l

    # for probit score terms we use (y - Phi(eta)) / phi(eta)
    # careful with phi(eta)==0 (extremely unlikely for normal), but users should be aware.
    phi_out = norm.pdf(eta_out)
    Phi_out = norm.cdf(eta_out)
    phi_med = norm.pdf(eta_med)
    Phi_med = norm.cdf(eta_med)

    # Build output vector of length 32
    out = np.array([
        # 1-4 score outcome model (probit)
        (y - Phi_out) / phi_out,
        ((y - Phi_out) / phi_out) * a,
        ((y - Phi_out) / phi_out) * m,
        ((y - Phi_out) / phi_out) * l,

        # 5-7 score mediator model (probit)
        (m - Phi_med) / phi_med,
        ((m - Phi_med) / phi_med) * a,
        ((m - Phi_med) / phi_med) * l,

        # 8-11 Mediator potential outcomes
        (norm.cdf(gam0 + gama*1 + gaml*l) - E_M1_1) * a,
        (norm.cdf(gam0 + gama*1 + gaml*l) - E_M1_0) * (1 - a),
        (norm.cdf(gam0 + gama*0 + gaml*l) - E_M0_1) * a,
        (norm.cdf(gam0 + gama*0 + gaml*l) - E_M0_0) * (1 - a),

        # 12-17 mediator effect on the outcome
        (norm.cdf(b0 + ba*0 + bm*1 + bl*l) - norm.cdf(b0 + ba*0 + bm*0 + bl*l) - E_I01I00_1) * a,
        (norm.cdf(b0 + ba*0 + bm*1 + bl*l) - norm.cdf(b0 + ba*0 + bm*0 + bl*l) - E_I01I00_0) * (1 - a),

        # 14-17 exposure effect on the outcome
        (norm.cdf(b0 + ba*1 + bm*0 + bl*l) - norm.cdf(b0 + ba*0 + bm*0 + bl*l) - E_I10I00_1) * a,
        (norm.cdf(b0 + ba*1 + bm*0 + bl*l) - norm.cdf(b0 + ba*0 + bm*0 + bl*l) - E_I10I00_0) * (1 - a),

        (norm.cdf(b0 + ba*1 + bm*1 + bl*l) - norm.cdf(b0 + ba*0 + bm*1 + bl*l) - E_I11I01_1) * a,
        (norm.cdf(b0 + ba*1 + bm*1 + bl*l) - norm.cdf(b0 + ba*0 + bm*1 + bl*l) - E_I11I01_0) * (1 - a),

        # 18-20 Indirect effects
        ((E_M1_1 - E_M0_1) * E_I01I00_1 - pi1) * a,
        ((E_M1_0 - E_M0_0) * E_I01I00_0 - pi0) * (1 - a),
        (pi0 * (1 - a) + pi1 * a - pi),

        # 21-23 Direct effects
        ((E_I10I00_1 * (1 - E_M1_1) + E_I11I01_1 * E_M1_1 - pd1) * a),
        ((E_I10I00_0 * (1 - E_M1_0) + E_I11I01_0 * E_M1_0 - pd0) * (1 - a)),
        (pd0 * (1 - a) + pd1 * a - pd),

        # 24-26 Indirect indices
        (1.0/pi0 - INNE),
        (1.0/pi1 - IEIN),
        (1.0/pi - INNT),

        # 27-29 Direct indices
        (1.0/pd0 - DNNE),
        (1.0/pd1 - DEIN),
        (1.0/pd - DNNT),

        # 30-32 marginal indices
        (1.0/(pi0 + pd0) - NNE),
        (1.0/(pi1 + pd1) - EIN),
        (1.0/(pi + pd) - NNT)
    ], dtype=float)

    return out


def qvec2(x, y=1, a=1, m=1, l=1):
    """
    Wrapper: x is length-32 parameter vector (1-based in R, 0-based here).
    Returns vector length 32.
    """
    # Unpack exactly in the same order as in your R code
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

    return qvec(
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
        NNE, EIN, NNT,
        y=y, a=a, m=m, l=l
    )


# ---------------------------
# Example: solve the system
# ---------------------------
if __name__ == "__main__":
    # initial guess (match R rep(1,32))
    x0 = np.ones(32)

    # Solve for default single observation (y=1,a=1,m=1,l=1) — analogous to calling nleqslv(x0, qvec2) in R
    sol = root(lambda xx: qvec2(xx, y=1, a=1, m=1, l=1), x0, method="hybr")
    print("Success:", sol.success, "message:", sol.message)
    print("Solution (first 10 elements):", sol.x[:10])

    # Optional: Jacobian numeric approximation (requires numdifftools or fallback)
    try:
        import numdifftools as nd
        J = nd.Jacobian(lambda xx: qvec2(xx, y=1, a=1, m=1, l=1))(sol.x)
        print("Jacobian shape:", J.shape)
    except Exception:
        # fallback finite differences (slow)
        eps = 1e-6
        p = len(x0)
        J = np.zeros((p, p))
        f0 = qvec2(sol.x, y=1, a=1, m=1, l=1)
        for j in range(p):
            xp = sol.x.copy()
            xp[j] += eps
            f1 = qvec2(xp, y=1, a=1, m=1, l=1)
            J[:, j] = (f1 - f0) / eps
        print("Jacobian (approx) computed with FD.")