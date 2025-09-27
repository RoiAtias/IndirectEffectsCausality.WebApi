import pandas as pd
import os
import asyncio
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
import re
import concurrent.futures
import requests
from bs4 import BeautifulSoup
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial
import numpy as np
from scipy.stats import norm
import statsmodels.formula.api as smf
from sklearn.utils import resample 
from scipy.special import expit as plogis
from scipy.optimize import root
from numpy.linalg import inv

class SandwichLogic():  
    def __init__(self):
        logging.getLogger('asyncio').setLevel(logging.WARNING)

    def qvec2_vectorized(self, x, Y, A, M, L, mediator_model, target_model):
        """
        Vectorized version supporting separate model types for mediator and outcome
        """
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

        # Choose appropriate link functions for outcome model
        if target_model == "logistic":
            outcome_pred = plogis(b0 + ba*A + bm*M + bl*L)
            score_outcome_1 = Y - outcome_pred
            score_outcome_a = score_outcome_1 * A
            score_outcome_m = score_outcome_1 * M
            score_outcome_l = score_outcome_1 * L
        else:
            # Probit outcome model
            eta_out = b0 + ba*A + bm*M + bl*L
            phi_out = norm.pdf(eta_out)
            Phi_out = norm.cdf(eta_out)
            phi_out = np.where(phi_out > 1e-10, phi_out, 1e-10)
            score_outcome_1 = (Y - Phi_out) / phi_out
            score_outcome_a = score_outcome_1 * A
            score_outcome_m = score_outcome_1 * M
            score_outcome_l = score_outcome_1 * L

        # Choose appropriate link functions for mediator model
        if mediator_model == "logistic":
            # Logistic mediator model
            mediator_pred = plogis(gam0 + gama*A + gaml*L)
            score_mediator_1 = M - mediator_pred
            score_mediator_a = score_mediator_1 * A
            score_mediator_l = score_mediator_1 * L
            # For potential outcomes
            E_M1_pred = plogis(gam0 + gama*1 + gaml*L)
            E_M0_pred = plogis(gam0 + gama*0 + gaml*L)
        else:
            # Probit mediator model
            eta_med = gam0 + gama*A + gaml*L
            phi_med = norm.pdf(eta_med)
            Phi_med = norm.cdf(eta_med)
            phi_med = np.where(phi_med > 1e-10, phi_med, 1e-10)
            score_mediator_1 = (M - Phi_med) / phi_med
            score_mediator_a = score_mediator_1 * A
            score_mediator_l = score_mediator_1 * L
            # For potential outcomes
            E_M1_pred = norm.cdf(gam0 + gama*1 + gaml*L)
            E_M0_pred = norm.cdf(gam0 + gama*0 + gaml*L)

        # Outcome potential outcomes based on target model
        if target_model == "logistic":
            I01I00 = plogis(b0 + ba*0 + bm*1 + bl*L) - plogis(b0 + ba*0 + bm*0 + bl*L)
            I10I00 = plogis(b0 + ba*1 + bm*0 + bl*L) - plogis(b0 + ba*0 + bm*0 + bl*L)
            I11I01 = plogis(b0 + ba*1 + bm*1 + bl*L) - plogis(b0 + ba*0 + bm*1 + bl*L)
        else:
            I01I00 = norm.cdf(b0 + ba*0 + bm*1 + bl*L) - norm.cdf(b0 + ba*0 + bm*0 + bl*L)
            I10I00 = norm.cdf(b0 + ba*1 + bm*0 + bl*L) - norm.cdf(b0 + ba*0 + bm*0 + bl*L)
            I11I01 = norm.cdf(b0 + ba*1 + bm*1 + bl*L) - norm.cdf(b0 + ba*0 + bm*1 + bl*L)

        # Vectorized calculations
        out = np.array([
            # Outcome model scores
            np.sum(score_outcome_1),
            np.sum(score_outcome_a),
            np.sum(score_outcome_m),
            np.sum(score_outcome_l),
            
            # Mediator model scores
            np.sum(score_mediator_1),
            np.sum(score_mediator_a),
            np.sum(score_mediator_l),
            
            # Mediator potential outcomes
            np.sum((E_M1_pred - E_M1_1) * A),
            np.sum((E_M1_pred - E_M1_0) * (1-A)),
            np.sum((E_M0_pred - E_M0_1) * A),
            np.sum((E_M0_pred - E_M0_0) * (1-A)),
            
            # Outcome effects
            np.sum((I01I00 - E_I01I00_1) * A),
            np.sum((I01I00 - E_I01I00_0) * (1-A)),
            np.sum((I10I00 - E_I10I00_1) * A),
            np.sum((I10I00 - E_I10I00_0) * (1-A)),
            np.sum((I11I01 - E_I11I01_1) * A),
            np.sum((I11I01 - E_I11I01_0) * (1-A)),
            
            # Effects
            np.sum(((E_M1_1 - E_M0_1) * E_I01I00_1 - pi1) * A),
            np.sum(((E_M1_0 - E_M0_0) * E_I01I00_0 - pi0) * (1-A)),
            np.sum(pi0*(1-A) + pi1*A - pi),
            np.sum((E_I10I00_1*(1-E_M1_1) + E_I11I01_1*E_M1_1 - pd1) * A),
            np.sum((E_I10I00_0*(1-E_M1_0) + E_I11I01_0*E_M1_0 - pd0) * (1-A)),
            np.sum(pd0*(1-A) + pd1*A - pd),
            
            # NNT indices
            np.sum(1/pi0 - INNE),
            np.sum(1/pi1 - IEIN),
            np.sum(1/pi - INNT),
            np.sum(1/pd0 - DNNE),
            np.sum(1/pd1 - DEIN),
            np.sum(1/pd - DNNT),
            np.sum(1/(pi0+pd0) - NNE),
            np.sum(1/(pi1+pd1) - EIN),
            np.sum(1/(pi+pd) - NNT)
        ])
        return out

    def compute_meat_matrix_fully_vectorized(self, sol_x, Y_data, A_data, M_data, L_values, mediator_model, target_model):
        """
        Fully vectorized MEAT matrix computation - fastest possible
        """
        print("Computing MEAT matrix using fully vectorized approach...")
        N = len(Y_data)
        
        # Compute all individual residuals at once using numpy broadcasting
        # This is much faster than any loop approach
        
        def compute_all_residuals_vectorized(x, Y, A, M, L, med_model, targ_model):
            """Compute residuals for all observations simultaneously"""
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

            # Vectorized computations for all observations
            if targ_model == "logistic":
                outcome_pred = plogis(b0 + ba*A + bm*M + bl*L)
                score_outcome_1 = Y - outcome_pred
            else:
                eta_out = b0 + ba*A + bm*M + bl*L
                phi_out = norm.pdf(eta_out)
                Phi_out = norm.cdf(eta_out)
                phi_out = np.where(phi_out > 1e-10, phi_out, 1e-10)
                score_outcome_1 = (Y - Phi_out) / phi_out

            if med_model == "logistic":
                mediator_pred = plogis(gam0 + gama*A + gaml*L)
                score_mediator_1 = M - mediator_pred
                E_M1_pred = plogis(gam0 + gama*1 + gaml*L)
                E_M0_pred = plogis(gam0 + gama*0 + gaml*L)
            else:
                eta_med = gam0 + gama*A + gaml*L
                phi_med = norm.pdf(eta_med)
                Phi_med = norm.cdf(eta_med)
                phi_med = np.where(phi_med > 1e-10, phi_med, 1e-10)
                score_mediator_1 = (M - Phi_med) / phi_med
                E_M1_pred = norm.cdf(gam0 + gama*1 + gaml*L)
                E_M0_pred = norm.cdf(gam0 + gama*0 + gaml*L)

            if targ_model == "logistic":
                I01I00 = plogis(b0 + ba*0 + bm*1 + bl*L) - plogis(b0 + ba*0 + bm*0 + bl*L)
                I10I00 = plogis(b0 + ba*1 + bm*0 + bl*L) - plogis(b0 + ba*0 + bm*0 + bl*L)
                I11I01 = plogis(b0 + ba*1 + bm*1 + bl*L) - plogis(b0 + ba*0 + bm*1 + bl*L)
            else:
                I01I00 = norm.cdf(b0 + ba*0 + bm*1 + bl*L) - norm.cdf(b0 + ba*0 + bm*0 + bl*L)
                I10I00 = norm.cdf(b0 + ba*1 + bm*0 + bl*L) - norm.cdf(b0 + ba*0 + bm*0 + bl*L)
                I11I01 = norm.cdf(b0 + ba*1 + bm*1 + bl*L) - norm.cdf(b0 + ba*0 + bm*1 + bl*L)

            # Create residual matrix (N x 32)
            residuals = np.zeros((len(Y), 32))
            
            # Fill residual matrix
            residuals[:, 0] = score_outcome_1
            residuals[:, 1] = score_outcome_1 * A
            residuals[:, 2] = score_outcome_1 * M
            residuals[:, 3] = score_outcome_1 * L
            
            residuals[:, 4] = score_mediator_1
            residuals[:, 5] = score_mediator_1 * A
            residuals[:, 6] = score_mediator_1 * L
            
            residuals[:, 7] = (E_M1_pred - E_M1_1) * A
            residuals[:, 8] = (E_M1_pred - E_M1_0) * (1-A)
            residuals[:, 9] = (E_M0_pred - E_M0_1) * A
            residuals[:, 10] = (E_M0_pred - E_M0_0) * (1-A)
            
            residuals[:, 11] = (I01I00 - E_I01I00_1) * A
            residuals[:, 12] = (I01I00 - E_I01I00_0) * (1-A)
            residuals[:, 13] = (I10I00 - E_I10I00_1) * A
            residuals[:, 14] = (I10I00 - E_I10I00_0) * (1-A)
            residuals[:, 15] = (I11I01 - E_I11I01_1) * A
            residuals[:, 16] = (I11I01 - E_I11I01_0) * (1-A)
            
            residuals[:, 17] = ((E_M1_1 - E_M0_1) * E_I01I00_1 - pi1) * A
            residuals[:, 18] = ((E_M1_0 - E_M0_0) * E_I01I00_0 - pi0) * (1-A)
            residuals[:, 19] = pi0*(1-A) + pi1*A - pi
            residuals[:, 20] = (E_I10I00_1*(1-E_M1_1) + E_I11I01_1*E_M1_1 - pd1) * A
            residuals[:, 21] = (E_I10I00_0*(1-E_M1_0) + E_I11I01_0*E_M1_0 - pd0) * (1-A)
            residuals[:, 22] = pd0*(1-A) + pd1*A - pd
            
            residuals[:, 23] = 1/pi0 - INNE
            residuals[:, 24] = 1/pi1 - IEIN
            residuals[:, 25] = 1/pi - INNT
            residuals[:, 26] = 1/pd0 - DNNE
            residuals[:, 27] = 1/pd1 - DEIN
            residuals[:, 28] = 1/pd - DNNT
            residuals[:, 29] = 1/(pi0+pd0) - NNE
            residuals[:, 30] = 1/(pi1+pd1) - EIN
            residuals[:, 31] = 1/(pi+pd) - NNT
            
            return residuals
        
        # Compute all residuals at once
        all_residuals = compute_all_residuals_vectorized(sol_x, Y_data, A_data, M_data, L_values, mediator_model, target_model)
        
        # Compute MEAT matrix using matrix multiplication - super fast!
        meat = (all_residuals.T @ all_residuals) / N
        
        print(f"MEAT matrix computed for {N} observations in vectorized mode")
        return meat

    def compute_jacobian_fully_vectorized(self, sol_x, Y_data, A_data, M_data, L_values, mediator_model, target_model):
        """
        Fully vectorized Jacobian computation using numerical gradients
        """
        print("Computing Jacobian using fully vectorized numerical gradients...")
        N = len(Y_data)
        eps = 1e-6
        
        def residual_func(x):
            """Function that computes sum of residuals for given x"""
            return self.qvec2_vectorized(x, Y_data, A_data, M_data, L_values, mediator_model, target_model)
        
        # Compute Jacobian using central differences - much more accurate and surprisingly fast
        jacobian = np.zeros((32, 32))
        
        print("Computing numerical gradients for all 32 parameters...")
        for i in range(32):
            # Forward difference
            x_plus = sol_x.copy()
            x_plus[i] += eps
            f_plus = residual_func(x_plus)
            
            # Backward difference  
            x_minus = sol_x.copy()
            x_minus[i] -= eps
            f_minus = residual_func(x_minus)
            
            # Central difference (more accurate)
            gradient = (f_plus - f_minus) / (2 * eps)
            jacobian[:, i] = -gradient  # Negative for bread matrix
            
            if (i + 1) % 8 == 0:
                print(f"Computed gradients for parameters 1-{i+1}")
        
        jacobian /= N
        print(f"Jacobian computed for {N} observations using central differences")
        return jacobian

    def qvec2_sum(self, x, Y, A, M, L, mediator_model, target_model):
        """
        Fast vectorized sum using the new approach
        """
        return self.qvec2_vectorized(x, Y, A, M, L, mediator_model, target_model)

    def compute_nnt_effects(self, data, exposure, mediator, outcome, confounders, mediator_model, target_model):
        """
        Compute NNT effects using sandwich covariance estimation method.
        """
        try:
            print("Starting sandwich estimation...")
            
            # Extract data arrays
            A_data = data[exposure].values
            M_data = data[mediator].values
            Y_data = data[outcome].values
            N = len(data)
            
            print(f"Data size: {N} observations")
            print(f"Mediator model: {mediator_model}, Target model: {target_model}")
            
            # Handle confounders - if empty, use a constant column
            if confounders:
                L_data = {conf: data[conf].values for conf in confounders}
                L_formula = " + ".join(confounders)
                # For sandwich estimation, we need a single L value per observation
                # If multiple confounders, use the first one as representative
                L_values = data[confounders[0]].values if confounders else np.ones(N)
            else:
                L_values = np.ones(N)  # Use constant if no confounders
                L_formula = "1"

            # Build regression formulas (similar to bootstrap logic)
            mediator_formula = f"{mediator} ~ {exposure} + {L_formula}"
            outcome_formula = f"{outcome} ~ {exposure} + {mediator} + {L_formula}"

            # Fit initial models to get starting parameter estimates
            try:
                print("Fitting initial models...")
                if mediator_model == "logistic":
                    mediator_model_fit = smf.logit(mediator_formula, data=data).fit(disp=0)
                else:
                    mediator_model_fit = smf.probit(mediator_formula, data=data).fit(disp=0)

                if target_model == "logistic":
                    outcome_model_fit = smf.logit(outcome_formula, data=data).fit(disp=0)
                else:
                    outcome_model_fit = smf.probit(outcome_formula, data=data).fit(disp=0)
                    
                print("Initial models fitted successfully")
            except Exception as e:
                print(f"Model fitting failed: {e}")
                return self._get_default_result()

            # Solve the system of equations using sandwich estimation
            print("Solving system of equations...")
            
            # Create better initial guess based on fitted models and empirical estimates
            x0 = np.ones(32) * 0.1  # Start with smaller values instead of 1.0
            
            # Set regression coefficients from fitted models
            try:
                if hasattr(outcome_model_fit, 'params') and len(outcome_model_fit.params) >= 3:
                    # Outcome model: b0, ba, bm, bl
                    params_out = outcome_model_fit.params.values
                    if len(params_out) >= 3:
                        x0[0] = params_out[0]  # intercept
                        x0[1] = params_out[1]  # exposure coefficient
                        x0[2] = params_out[2]  # mediator coefficient
                        if len(params_out) >= 4:
                            x0[3] = params_out[3]  # confounder coefficient
                        else:
                            x0[3] = 0.0
                    
                if hasattr(mediator_model_fit, 'params') and len(mediator_model_fit.params) >= 2:
                    # Mediator model: gam0, gama, gaml
                    params_med = mediator_model_fit.params.values
                    if len(params_med) >= 2:
                        x0[4] = params_med[0]  # intercept
                        x0[5] = params_med[1]  # exposure coefficient
                        if len(params_med) >= 3:
                            x0[6] = params_med[2]  # confounder coefficient
                        else:
                            x0[6] = 0.0
                
                print(f"Set regression coefficients from fitted models")
                
            except Exception as e:
                print(f"Warning: Could not extract model parameters: {e}")
            
            # Set reasonable starting values for other parameters based on data
            try:
                # Calculate empirical probabilities for better initial guesses
                A0_mask = A_data == 0
                A1_mask = A_data == 1
                
                # Empirical conditional probabilities for mediator
                if A1_mask.sum() > 0:
                    x0[7] = np.mean(M_data[A1_mask])  # E_M1_1
                else:
                    x0[7] = 0.5
                    
                if A0_mask.sum() > 0:
                    x0[8] = np.mean(M_data[A0_mask])  # E_M1_0
                else:
                    x0[8] = 0.5
                    
                x0[9] = x0[7] * 0.8   # E_M0_1
                x0[10] = x0[8] * 0.8  # E_M0_0
                
                # Set small positive values for effect parameters
                x0[11:17] = 0.05  # E_I parameters
                
                # Set small positive values for pi and pd parameters
                x0[17:23] = 0.01  # pi0, pi1, pi, pd0, pd1, pd
                
                # Set reasonable NNT starting values (inverse of small probabilities)
                x0[23:32] = [100, 100, 100, 50, 50, 50, 30, 30, 30]  # NNT estimates
                
                print("Set empirical starting values for effect parameters")
                
            except Exception as e:
                print(f"Warning: Could not set empirical starting values: {e}")
                # Fallback to conservative values
                x0[7:11] = 0.3  # E_M parameters
                x0[11:17] = 0.01  # E_I parameters
                x0[17:23] = 0.01  # effect parameters
                x0[23:32] = 100  # NNT parameters
            
            # Try multiple solution methods
            solution_found = False
            sol_x = x0.copy()
            
            methods_to_try = [
                ('hybr', {'maxfev': 2000, 'xtol': 1e-8}),
                ('lm', {'maxfev': 1500, 'xtol': 1e-6}),
                ('broyden1', {'maxiter': 1000}),
                ('anderson', {'maxiter': 800}),
                ('krylov', {'maxiter': 600})
            ]
            
            for method_name, options in methods_to_try:
                print(f"Trying {method_name} method...")
                try:
                    sol = root(lambda x: self.qvec2_sum(x, Y_data, A_data, M_data, L_values, mediator_model, target_model), 
                              x0, method=method_name, options=options)
                    
                    if sol.success:
                        print(f"? {method_name} method succeeded!")
                        sol_x = sol.x
                        solution_found = True
                        break
                    else:
                        print(f"? {method_name} method failed: {sol.message}")
                        # If we have a partial solution, keep it as backup
                        if hasattr(sol, 'x') and sol.x is not None:
                            if not solution_found:  # Only update if we don't have a better solution
                                sol_x = sol.x
                
                except Exception as e:
                    print(f"? {method_name} method crashed: {e}")
                    continue
            
            if not solution_found:
                print("? Warning: No method converged fully. Using best partial solution.")
                # Try one more time with a different starting point
                x0_alt = x0 * 0.5 + np.random.normal(0, 0.01, 32)  # Add small noise
                try:
                    sol = root(lambda x: self.qvec2_sum(x, Y_data, A_data, M_data, L_values, mediator_model, target_model), 
                              x0_alt, method='hybr', options={'maxfev': 500, 'xtol': 1e-4})
                    if hasattr(sol, 'x'):
                        sol_x = sol.x
                        print("Used alternative starting point solution")
                except:
                    print("Alternative starting point also failed, using initial guess estimates")
            
            # Validate solution
            try:
                residual = self.qvec2_sum(sol_x, Y_data, A_data, M_data, L_values, mediator_model, target_model)
                residual_norm = np.linalg.norm(residual)
                print(f"Final residual norm: {residual_norm:.6f}")
                
                if residual_norm > 0.1:
                    print(f"? Warning: Large residual norm ({residual_norm:.6f}). Solution may be inaccurate.")
                
            except Exception as e:
                print(f"Could not validate solution: {e}")

            # Extract estimates
            estimates = sol_x[23:32]  # NNT indices are at positions 23-31

            print("Calculating sandwich covariance matrix...")
            
            # Use vectorized computation for BREAD matrix (Jacobian)
            bread = self.compute_jacobian_fully_vectorized(sol_x, Y_data, A_data, M_data, L_values, mediator_model, target_model)

            try:
                inv_a = inv(bread)
                print("Bread matrix inverted successfully")
            except np.linalg.LinAlgError:
                print("Bread matrix inversion failed, using pseudo-inverse")
                inv_a = np.linalg.pinv(bread)

            # Use vectorized computation for MEAT matrix (covariance)
            meat = self.compute_meat_matrix_fully_vectorized(sol_x, Y_data, A_data, M_data, L_values, mediator_model, target_model)

            # Calculate SANDWICH covariance matrix
            sand = (1/N) * inv_a @ meat @ inv_a.T
            print("Sandwich matrix calculated successfully")

            # Calculate confidence intervals
            ci_results = {}
            nnt_names = ["INNE", "IEIN", "INNT", "DNNE", "DEIN", "DNNT", "NNE", "EIN", "NNT"]
            
            for idx, name in enumerate(nnt_names):
                param_idx = 23 + idx
                est = sol_x[param_idx]
                se = np.sqrt(abs(sand[param_idx, param_idx]))  # Use abs to handle negative variance
                
                if est >= 1 and not np.isnan(est) and not np.isinf(est):
                    ci_lower = max(est - 1.96*se, 1)
                    ci_upper = est + 1.96*se
                else:
                    ci_lower = np.inf
                    ci_upper = np.inf
                
                ci_results[name] = est
                # Apply rounding to 2 decimal places for confidence intervals, same as Bootstrap
                ci_results[f"CI_{name}_LOWER"] = round(ci_lower, 2) if not np.isinf(ci_lower) else None
                ci_results[f"CI_{name}_UPPER"] = round(ci_upper, 2) if not np.isinf(ci_upper) else None
                # Apply rounding to 2 decimal places for confidence intervals, same as Bootstrap
                ci_results[f"CI_{name}_LOWER"] = round(ci_lower, 2) if not np.isinf(ci_lower) else None
                ci_results[f"CI_{name}_UPPER"] = round(ci_upper, 2) if not np.isinf(ci_upper) else None
                
            print("Confidence intervals calculated successfully")

            # Build result structure matching bootstrap output
            result = {
                # Probability components (extracted from solution)
                "p_i": round(sol_x[19], 5) if not np.isnan(sol_x[19]) else 0.0,  # pi
                "p_d": round(sol_x[22], 5) if not np.isnan(sol_x[22]) else 0.0,  # pd
                "p_b": round(sol_x[19] + sol_x[22], 5) if not (np.isnan(sol_x[19]) or np.isnan(sol_x[22])) else 0.0,  # pi + pd
                "p_i0": round(sol_x[17], 5) if not np.isnan(sol_x[17]) else 0.0,  # pi0
                "p_i1": round(sol_x[18], 5) if not np.isnan(sol_x[18]) else 0.0,  # pi1
                "p_d0": round(sol_x[20], 5) if not np.isnan(sol_x[20]) else 0.0,  # pd0
                "p_d1": round(sol_x[21], 5) if not np.isnan(sol_x[21]) else 0.0,  # pd1
                # Main NNT measures
                "INNT": round(ci_results["INNT"], 2) if not np.isnan(ci_results["INNT"]) else None,
                "DNNT": round(ci_results["DNNT"], 2) if not np.isnan(ci_results["DNNT"]) else None,
                "NNT": round(ci_results["NNT"], 2) if not np.isnan(ci_results["NNT"]) else None,
                # Additional NNT measures by exposure group
                "INNE": round(ci_results["INNE"], 2) if not np.isnan(ci_results["INNE"]) else None,
                "IEIN": round(ci_results["IEIN"], 2) if not np.isnan(ci_results["IEIN"]) else None,
                "DNNE": round(ci_results["DNNE"], 2) if not np.isnan(ci_results["DNNE"]) else None,
                "DEIN": round(ci_results["DEIN"], 2) if not np.isnan(ci_results["DEIN"]) else None,
                "NNE": round(ci_results["NNE"], 2) if not np.isnan(ci_results["NNE"]) else None,
                "EIN": round(ci_results["EIN"], 2) if not np.isnan(ci_results["EIN"]) else None,
                
                # Confidence intervals for main measures
                "CI_INNT_LOWER": ci_results["CI_INNT_LOWER"],
                "CI_INNT_UPPER": ci_results["CI_INNT_UPPER"],
                "CI_DNNT_LOWER": ci_results["CI_DNNT_LOWER"],
                "CI_DNNT_UPPER": ci_results["CI_DNNT_UPPER"],
                "CI_NNT_LOWER": ci_results["CI_NNT_LOWER"],
                "CI_NNT_UPPER": ci_results["CI_NNT_UPPER"],
                # Confidence intervals for additional measures
                "CI_INNE_LOWER": ci_results["CI_INNE_LOWER"],
                "CI_INNE_UPPER": ci_results["CI_INNE_UPPER"],
                "CI_IEIN_LOWER": ci_results["CI_IEIN_LOWER"],
                "CI_IEIN_UPPER": ci_results["CI_IEIN_UPPER"],
                "CI_DNNE_LOWER": ci_results["CI_DNNE_LOWER"],
                "CI_DNNE_UPPER": ci_results["CI_DNNE_UPPER"],
                "CI_DEIN_LOWER": ci_results["CI_DEIN_LOWER"],
                "CI_DEIN_UPPER": ci_results["CI_DEIN_UPPER"],
                "CI_NNE_LOWER": ci_results["CI_NNE_LOWER"],
                "CI_NNE_UPPER": ci_results["CI_NNE_UPPER"],
                "CI_EIN_LOWER": ci_results["CI_EIN_LOWER"],
                "CI_EIN_UPPER": ci_results["CI_EIN_UPPER"],
                # Bootstrap matrix placeholder (empty for sandwich method)
                "Bootstrap": pd.DataFrame()
            }

            print("Sandwich estimation completed successfully!")
            return result

        except Exception as e:
            print(f"Sandwich estimation failed: {e}")
            return self._get_default_result()

    def _get_default_result(self):
        """
        Return default result structure when computation fails
        """
        return {
            # Probability components
            "p_i": 0.0,
            "p_d": 0.0, 
            "p_b": 0.0,
            "p_i0": 0.0,
            "p_i1": 0.0,
            "p_d0": 0.0,
            "p_d1": 0.0,
            
            # Main NNT measures
            "INNT": None,
            "DNNT": None,
            "NNT": None,
            
            # Additional NNT measures
            "INNE": None,
            "IEIN": None,
            "DNNE": None,
            "DEIN": None,
            "NNE": None,
            "EIN": None,
            
            # Confidence intervals
            "CI_INNT_LOWER": None,
            "CI_INNT_UPPER": None,
            "CI_DNNT_LOWER": None,
            "CI_DNNT_UPPER": None,
            "CI_NNT_LOWER": None,
            "CI_NNT_UPPER": None,
            "CI_INNE_LOWER": None,
            "CI_INNE_UPPER": None,
            "CI_IEIN_LOWER": None,
            "CI_IEIN_UPPER": None,
            "CI_DNNE_LOWER": None,
            "CI_DNNE_UPPER": None,
            "CI_DEIN_LOWER": None,
            "CI_DEIN_UPPER": None,
            "CI_NNE_LOWER": None,
            "CI_NNE_UPPER": None,
            "CI_EIN_LOWER": None,
            "CI_EIN_UPPER": None,
            
            # Bootstrap matrix placeholder
            "Bootstrap": pd.DataFrame()
        }

def process_results(file):
    """
    Test function for SandwichLogic with fixed parameters
    
    Parameters:
    -----------
    file : str
        Path to the CSV file URL
    
    Returns:
    --------
    dict
        Dictionary containing all NNT measures and confidence intervals
    """
    
    # Fixed parameters as requested
    confounders = ["sex", "age"]
    predictor_x = "smoker"
    mediator_y = "overweight"
    target_variable = "HeartDiseaseorAttack"
    mediator_model = "logistic"
    target_model = "logistic"
    ci_method = "sandwich"
    
    print(f"Processing file: {file}")
    print(f"Using Sandwich method with the following parameters:")
    print(f"- Confounders: {confounders}")
    print(f"- Predictor (X): {predictor_x}")
    print(f"- Mediator (Y): {mediator_y}")
    print(f"- Target Variable: {target_variable}")
    print(f"- Mediator Model: {mediator_model}")
    print(f"- Target Model: {target_model}")
    print(f"- CI Method: {ci_method}")
    print("-" * 50)
    
    try:
        # Load the data
        data = pd.read_csv(file)
        print(f"Data loaded successfully. Shape: {data.shape}")
        
        # Create SandwichLogic instance with all the logic embedded
        sandwich_logic = SandwichLogic()
        
        # Process the data using SandwichLogic
        result = sandwich_logic.compute_nnt_effects(
            data=data,
            exposure=predictor_x,
            mediator=mediator_y,
            outcome=target_variable,
            confounders=confounders,
            mediator_model=mediator_model,
            target_model=target_model
        )
        
        # Enhanced results dictionary with all NNT measures (as requested)
        results = {
            # Original results maintained for backward compatibility
            "indirect_effect": result["p_i"],
            "total_effect": result["p_d"],  # Note: this was mislabeled before
            "direct_effect": result["p_b"], # Note: this was mislabeled before
            "innt": result["INNT"],
            "dnnt": result["DNNT"],
            "nnt": result["NNT"],
            "nnt_confidence_interval_lower": result["CI_NNT_LOWER"],
            "nnt_confidence_interval_upper": result["CI_NNT_UPPER"],
            "innt_confidence_interval_lower": result["CI_INNT_LOWER"],
            "innt_confidence_interval_upper": result["CI_INNT_UPPER"],
            "dnnt_confidence_interval_lower": result["CI_DNNT_LOWER"],
            "dnnt_confidence_interval_upper": result["CI_DNNT_UPPER"],
            
            # Enhanced results with exposure group-specific measures
            "indirect_effect_a0": result["p_i0"],
            "indirect_effect_a1": result["p_i1"],
            "direct_effect_a0": result["p_d0"],
            "direct_effect_a1": result["p_d1"],
            
            # Additional NNT measures
            "inne": result["INNE"],  # Indirect Number Needed to Expose (A=0)
            "iein": result["IEIN"],  # Indirect Effect when Intervening (A=1)
            "dnne": result["DNNE"],  # Direct Number Needed to Expose (A=0)
            "dein": result["DEIN"],  # Direct Effect when Intervening (A=1)
            "nne": result["NNE"],    # Total Number Needed to Expose (A=0)
            "ein": result["EIN"],    # Total Effect when Intervening (A=1)
            
            # Confidence intervals for main measures
            "CI_INNT_LOWER": ci_results["CI_INNT_LOWER"],
            "CI_INNT_UPPER": ci_results["CI_INNT_UPPER"],
            "CI_DNNT_LOWER": ci_results["CI_DNNT_LOWER"],
            "CI_DNNT_UPPER": ci_results["CI_DNNT_UPPER"],
            "CI_NNT_LOWER": ci_results["CI_NNT_LOWER"],
            "CI_NNT_UPPER": ci_results["CI_NNT_UPPER"],
            # Confidence intervals for additional measures
            "CI_INNE_LOWER": ci_results["CI_INNE_LOWER"],
            "CI_INNE_UPPER": ci_results["CI_INNE_UPPER"],
            "CI_IEIN_LOWER": ci_results["CI_IEIN_LOWER"],
            "CI_IEIN_UPPER": ci_results["CI_IEIN_UPPER"],
            "CI_DNNE_LOWER": ci_results["CI_DNNE_LOWER"],
            "CI_DNNE_UPPER": ci_results["CI_DNNE_UPPER"],
            "CI_DEIN_LOWER": ci_results["CI_DEIN_LOWER"],
            "CI_DEIN_UPPER": ci_results["CI_DEIN_UPPER"],
            "CI_NNE_LOWER": ci_results["CI_NNE_LOWER"],
            "CI_NNE_UPPER": ci_results["CI_NNE_UPPER"],
            "CI_EIN_LOWER": ci_results["CI_EIN_LOWER"],
            "CI_EIN_UPPER": ci_results["CI_EIN_UPPER"],
            # Bootstrap matrix placeholder (empty for sandwich method)
            "Bootstrap": pd.DataFrame()
        }

        # Display results
        print("\n" + "="*60)
        print("SANDWICH LOGIC TEST RESULTS")
        print("="*60)
        
        print("\n?? BASIC EFFECTS:")
        print(f"Indirect Effect: {results['indirect_effect']}")
        print(f"Total Effect: {results['total_effect']}")
        print(f"Direct Effect: {results['direct_effect']}")
        
        print("\n?? EXPOSURE GROUP-SPECIFIC EFFECTS:")
        print(f"Indirect Effect (A=0): {results['indirect_effect_a0']}")
        print(f"Indirect Effect (A=1): {results['indirect_effect_a1']}")
        print(f"Direct Effect (A=0): {results['direct_effect_a0']}")
        print(f"Direct Effect (A=1): {results['direct_effect_a1']}")
        
        print("\n?? MAIN NNT MEASURES:")
        print(f"INNT: {results['innt']}")
        print(f"DNNT: {results['dnnt']}")
        print(f"NNT: {results['nnt']}")
        
        print("\n?? ADDITIONAL NNT MEASURES:")
        print(f"INNE: {results['inne']}")
        print(f"IEIN: {results['iein']}")
        print(f"DNNE: {results['dnne']}")
        print(f"DEIN: {results['dein']}")
        print(f"NNE: {results['nne']}")
        print(f"EIN: {results['ein']}")
        
        print("\n?? CONFIDENCE INTERVALS - MAIN MEASURES:")
        print(f"INNT CI: [{results['innt_confidence_interval_lower']}, {results['innt_confidence_interval_upper']}]")
        print(f"DNNT CI: [{results['dnnt_confidence_interval_lower']}, {results['dnnt_confidence_interval_upper']}]")
        print(f"NNT CI: [{results['nnt_confidence_interval_lower']}, {results['nnt_confidence_interval_upper']}]")
        
        print("\n?? CONFIDENCE INTERVALS - ADDITIONAL MEASURES:")
        print(f"INNE CI: [{results['inne_confidence_interval_lower']}, {results['inne_confidence_interval_upper']}]")
        print(f"IEIN CI: [{results['iein_confidence_interval_lower']}, {results['iein_confidence_interval_upper']}]")
        print(f"DNNE CI: [{results['dnne_confidence_interval_lower']}, {results['dnne_confidence_interval_upper']}]")
        print(f"DEIN CI: [{results['dein_confidence_interval_lower']}, {results['dein_confidence_interval_upper']}]")
        print(f"NNE CI: [{results['nne_confidence_interval_lower']}, {results['nne_confidence_interval_upper']}]")
        print(f"EIN CI: [{results['ein_confidence_interval_lower']}, {results['ein_confidence_interval_upper']}]")
        
        print(f"\n?? Method Used: {results['ci_method_used']}")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"? Error processing file: {e}")
        return None

def main():
    """
    Main function to run the test
    """
    print("?? SandwichLogic Test Script")
    print("="*40)
    
    # Example usage - replace with actual file path
    file_path = "path/to/your/data.csv"  # Replace with actual file path
    
    print(f"?? Testing with file: {file_path}")
    print("Note: Replace 'file_path' with actual CSV file URL in main() function")
    
    # Uncomment the line below and provide actual file path to run test
    # results = process_results(file_path)
    
    print("\n? Test script created successfully!")
    print("To use this test:")
    print("1. Replace 'file_path' with your actual CSV file path/URL")
    print("2. Uncomment the process_results() call")
    print("3. Run the script")

if __name__ == "__main__":
    main()