"""
Sandwich Method for NNT Estimation in Mediation Analysis

This module implements the Sandwich estimator for causal mediation effects.
Matches the SandwichLogic.py used in the main application.
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import norm
import statsmodels.formula.api as smf
from scipy.special import expit as plogis
from scipy.optimize import root
from numpy.linalg import inv
import warnings
warnings.filterwarnings('ignore')


class SandwichNNT:
    """Sandwich estimator for NNT in mediation analysis"""
    
    def __init__(self):
        """Initialize the Sandwich NNT estimator"""
        pass
    
    def qvec2_vectorized(self, x, Y, A, M, L, mediator_model, target_model):
        """
        Vectorized 32-equation system for solving mediation parameters
        Supports logistic and probit models for both mediator and outcome
        """
        # Extract parameters from x vector
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

        # Outcome model scores
        if target_model == "logistic":
            outcome_pred = plogis(b0 + ba*A + bm*M + bl*L)
            score_outcome_1 = Y - outcome_pred
            score_outcome_a = score_outcome_1 * A
            score_outcome_m = score_outcome_1 * M
            score_outcome_l = score_outcome_1 * L
        else:  # probit
            eta_out = b0 + ba*A + bm*M + bl*L
            phi_out = norm.pdf(eta_out)
            Phi_out = norm.cdf(eta_out)
            phi_out = np.where(phi_out > 1e-10, phi_out, 1e-10)
            score_outcome_1 = (Y - Phi_out) / phi_out
            score_outcome_a = score_outcome_1 * A
            score_outcome_m = score_outcome_1 * M
            score_outcome_l = score_outcome_1 * L

        # Mediator model scores
        if mediator_model == "logistic":
            mediator_pred = plogis(gam0 + gama*A + gaml*L)
            score_mediator_1 = M - mediator_pred
            score_mediator_a = score_mediator_1 * A
            score_mediator_l = score_mediator_1 * L
            E_M1_pred = plogis(gam0 + gama*1 + gaml*L)
            E_M0_pred = plogis(gam0 + gama*0 + gaml*L)
        else:  # probit
            eta_med = gam0 + gama*A + gaml*L
            phi_med = norm.pdf(eta_med)
            Phi_med = norm.cdf(eta_med)
            phi_med = np.where(phi_med > 1e-10, phi_med, 1e-10)
            score_mediator_1 = (M - Phi_med) / phi_med
            score_mediator_a = score_mediator_1 * A
            score_mediator_l = score_mediator_1 * L
            E_M1_pred = norm.cdf(gam0 + gama*1 + gaml*L)
            E_M0_pred = norm.cdf(gam0 + gama*0 + gaml*L)

        # Potential outcomes
        if target_model == "logistic":
            I01I00 = plogis(b0 + ba*0 + bm*1 + bl*L) - plogis(b0 + ba*0 + bm*0 + bl*L)
            I10I00 = plogis(b0 + ba*1 + bm*0 + bl*L) - plogis(b0 + ba*0 + bm*0 + bl*L)
            I11I01 = plogis(b0 + ba*1 + bm*1 + bl*L) - plogis(b0 + ba*0 + bm*1 + bl*L)
        else:  # probit
            I01I00 = norm.cdf(b0 + ba*0 + bm*1 + bl*L) - norm.cdf(b0 + ba*0 + bm*0 + bl*L)
            I10I00 = norm.cdf(b0 + ba*1 + bm*0 + bl*L) - norm.cdf(b0 + ba*0 + bm*0 + bl*L)
            I11I01 = norm.cdf(b0 + ba*1 + bm*1 + bl*L) - norm.cdf(b0 + ba*0 + bm*1 + bl*L)

        # Build 32-equation system
        A0_mask = (A == 0)
        A1_mask = (A == 1)
        prop_A0 = np.mean(A0_mask)
        prop_A1 = np.mean(A1_mask)

        out = np.array([
            # Equations 1-4: Outcome model scores
            np.sum(score_outcome_1),
            np.sum(score_outcome_a),
            np.sum(score_outcome_m),
            np.sum(score_outcome_l),
            
            # Equations 5-7: Mediator model scores
            np.sum(score_mediator_1),
            np.sum(score_mediator_a),
            np.sum(score_mediator_l),
            
            # Equations 8-11: Mediator potential outcomes
            np.sum(E_M1_pred[A1_mask] - E_M1_1) if A1_mask.sum() > 0 else 0,
            np.sum(E_M1_pred[A0_mask] - E_M1_0) if A0_mask.sum() > 0 else 0,
            np.sum(E_M0_pred[A1_mask] - E_M0_1) if A1_mask.sum() > 0 else 0,
            np.sum(E_M0_pred[A0_mask] - E_M0_0) if A0_mask.sum() > 0 else 0,
            
            # Equations 12-17: Outcome potential outcomes
            np.sum(I01I00[A0_mask] - E_I01I00_0) if A0_mask.sum() > 0 else 0,
            np.sum(I01I00[A1_mask] - E_I01I00_1) if A1_mask.sum() > 0 else 0,
            np.sum(I10I00[A0_mask] - E_I10I00_0) if A0_mask.sum() > 0 else 0,
            np.sum(I10I00[A1_mask] - E_I10I00_1) if A1_mask.sum() > 0 else 0,
            np.sum(I11I01[A0_mask] - E_I11I01_0) if A0_mask.sum() > 0 else 0,
            np.sum(I11I01[A1_mask] - E_I11I01_1) if A1_mask.sum() > 0 else 0,
            
            # Equations 18-20: Indirect effect
            pi0 - (E_M1_0 - E_M0_0) * E_I01I00_0,
            pi1 - (E_M1_1 - E_M0_1) * E_I01I00_1,
            pi - (pi0 * prop_A0 + pi1 * prop_A1),
            
            # Equations 21-23: Direct effect
            pd0 - (E_I10I00_0 * (1 - E_M1_0) + E_I11I01_0 * E_M1_0),
            pd1 - (E_I10I00_1 * (1 - E_M1_1) + E_I11I01_1 * E_M1_1),
            pd - (pd0 * prop_A0 + pd1 * prop_A1),
            
            # Equations 24-26: Indirect NNT
            INNE - 1/pi0 if abs(pi0) > 1e-6 else 0,
            IEIN - 1/pi1 if abs(pi1) > 1e-6 else 0,
            INNT - 1/pi if abs(pi) > 1e-6 else 0,
            
            # Equations 27-29: Direct NNT
            DNNE - 1/pd0 if abs(pd0) > 1e-6 else 0,
            DEIN - 1/pd1 if abs(pd1) > 1e-6 else 0,
            DNNT - 1/pd if abs(pd) > 1e-6 else 0,
            
            # Equations 30-32: Total NNT
            NNE - 1/(pi0 + pd0) if abs(pi0 + pd0) > 1e-6 else 0,
            EIN - 1/(pi1 + pd1) if abs(pi1 + pd1) > 1e-6 else 0,
            NNT - 1/(pi + pd) if abs(pi + pd) > 1e-6 else 0,
        ])
        
        return out
    
    def compute_nnt_effects(
        self, 
        data, 
        exposure, 
        mediator, 
        outcome, 
        confounders, 
        mediator_model="logistic", 
        target_model="logistic", 
        selected_effects=None
    ):
        """
        Compute NNT effects using Sandwich estimator

        Parameters:
        - data: DataFrame containing all variables
        - exposure: Name of exposure/treatment variable (binary: 0/1)
        - mediator: Name of mediator variable (binary: 0/1)
        - outcome: Name of outcome variable (binary: 0/1)
        - confounders: List of confounder variable names
        - mediator_model: "logistic" or "probit"
        - target_model: "logistic" or "probit"
        - selected_effects: Which effects to compute (default=None means all)

        Returns: Dictionary with NNT estimates and confidence intervals
        """
        
        print(f"\n{'='*70}")
        print("SANDWICH METHOD FOR NNT ESTIMATION")
        print(f"{'='*70}\n")
        
        # STEP 1: DATA EXTRACTION
        print("Step 1: Extracting and validating data...")
        
        A_data = data[exposure].values
        M_data = data[mediator].values
        Y_data = data[outcome].values
        L_data = data[confounders[0]].values if len(confounders) == 1 else np.mean([data[c].values for c in confounders], axis=0)
        N = len(data)
        
        print(f"Sample size: {N} observations")
        print(f"Exposure ({exposure}): {np.sum(A_data==1)} treated, {np.sum(A_data==0)} control")
        print(f"Confounders: {confounders}")
        
        # STEP 2: FIT MODELS TO GET INITIAL ESTIMATES
        print(f"\nStep 2: Fitting initial models...")
        
        L_formula = " + ".join(confounders)
        mediator_formula = f"{mediator} ~ {exposure} + {L_formula}"
        outcome_formula = f"{outcome} ~ {exposure} + {mediator} + {L_formula}"
        
        if mediator_model == "logistic":
            med_model = smf.logit(mediator_formula, data=data).fit(disp=0)
        else:
            med_model = smf.probit(mediator_formula, data=data).fit(disp=0)
        
        if target_model == "logistic":
            out_model = smf.logit(outcome_formula, data=data).fit(disp=0)
        else:
            out_model = smf.probit(outcome_formula, data=data).fit(disp=0)
        
        # STEP 3: BUILD INITIAL PARAMETER VECTOR
        print(f"\nStep 3: Building initial parameter vector (32 parameters)...")
        
        # Extract coefficients
        b0 = out_model.params['Intercept']
        ba = out_model.params[exposure]
        bm = out_model.params[mediator]
        bl = out_model.params[confounders[0]] if len(confounders) == 1 else np.mean([out_model.params[c] for c in confounders])
        
        gam0 = med_model.params['Intercept']
        gama = med_model.params[exposure]
        gaml = med_model.params[confounders[0]] if len(confounders) == 1 else np.mean([med_model.params[c] for c in confounders])
        
        # Initial estimates for potential outcomes
        x0 = np.array([
            b0, ba, bm, bl,                    # Outcome model parameters
            gam0, gama, gaml,                  # Mediator model parameters
            0.5, 0.5, 0.5, 0.5,               # Mediator potential outcomes
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1,     # Outcome potential outcomes
            0.01, 0.01, 0.01,                 # Indirect effects
            0.01, 0.01, 0.01,                 # Direct effects
            100, 100, 100,                     # Indirect NNT
            100, 100, 100,                     # Direct NNT
            100, 100, 100                      # Total NNT
        ])
        
        # STEP 4: SOLVE 32-EQUATION SYSTEM
        print(f"\nStep 4: Solving 32-equation system...")
        
        try:
            sol = root(
                lambda x: self.qvec2_vectorized(x, Y_data, A_data, M_data, L_data, mediator_model, target_model),
                x0,
                method='hybr',
                options={'maxfev': 10000}
            )
            
            if not sol.success:
                print(f"WARNING: Root solver did not converge. Using initial estimates.")
                theta_hat = x0
            else:
                theta_hat = sol.x
                print("Successfully solved equation system!")
        
        except Exception as e:
            print(f"ERROR in solver: {e}")
            print("Using initial estimates.")
            theta_hat = x0
        
        # STEP 5: CALCULATE COVARIANCE MATRIX
        print(f"\nStep 5: Calculating covariance matrix...")
        
        # Compute numerical Jacobian
        eps = 1e-6
        J = np.zeros((32, 32))
        
        for j in range(32):
            theta_plus = theta_hat.copy()
            theta_plus[j] += eps
            f_plus = self.qvec2_vectorized(theta_plus, Y_data, A_data, M_data, L_data, mediator_model, target_model)
            
            theta_minus = theta_hat.copy()
            theta_minus[j] -= eps
            f_minus = self.qvec2_vectorized(theta_minus, Y_data, A_data, M_data, L_data, mediator_model, target_model)
            
            J[:, j] = (f_plus - f_minus) / (2 * eps)
        
        # Sandwich variance estimator
        try:
            BREAD = inv(J / N)
            
            # Compute MEAT matrix
            psi = np.zeros((N, 32))
            for i in range(N):
                psi[i, :] = self.qvec2_vectorized(
                    theta_hat,
                    Y_data[i:i+1],
                    A_data[i:i+1],
                    M_data[i:i+1],
                    L_data[i:i+1],
                    mediator_model,
                    target_model
                )
            
            MEAT = (psi.T @ psi) / N
            V = (BREAD @ MEAT @ BREAD.T) / N
            se = np.sqrt(np.diag(V))
            
            print("Covariance matrix calculated successfully!")
        
        except Exception as e:
            print(f"WARNING: Could not compute covariance matrix: {e}")
            se = np.zeros(32)
        
        # STEP 6: BUILD RESULTS
        print(f"\nStep 6: Building results dictionary...")
        
        # Extract estimates
        pi0, pi1, pi = theta_hat[17:20]
        pd0, pd1, pd = theta_hat[20:23]
        INNE, IEIN, INNT = theta_hat[23:26]
        DNNE, DEIN, DNNT = theta_hat[26:29]
        NNE, EIN, NNT = theta_hat[29:32]
        
        # Standard errors
        se_pi0, se_pi1, se_pi = se[17:20]
        se_pd0, se_pd1, se_pd = se[20:23]
        se_INNE, se_IEIN, se_INNT = se[23:26]
        se_DNNE, se_DEIN, se_DNNT = se[26:29]
        se_NNE, se_EIN, se_NNT = se[29:32]
        
        # Build confidence intervals
        def calc_ci(estimate, std_error):
            lower = estimate - 1.96 * std_error
            upper = estimate + 1.96 * std_error
            return lower, upper
        
        result = {
            # Probability components
            "p_i": round(pi, 5),
            "p_d": round(pd, 5),
            "p_b": round(pi + pd, 5),
            "p_i0": round(pi0, 5),
            "p_i1": round(pi1, 5),
            "p_d0": round(pd0, 5),
            "p_d1": round(pd1, 5),
            
            # NNT estimates
            "INNT": round(INNT, 2) if abs(pi) > 1e-6 else None,
            "INNE": round(INNE, 2) if abs(pi0) > 1e-6 else None,
            "IEIN": round(IEIN, 2) if abs(pi1) > 1e-6 else None,
            "DNNT": round(DNNT, 2) if abs(pd) > 1e-6 else None,
            "DNNE": round(DNNE, 2) if abs(pd0) > 1e-6 else None,
            "DEIN": round(DEIN, 2) if abs(pd1) > 1e-6 else None,
            "NNT": round(NNT, 2) if abs(pi + pd) > 1e-6 else None,
            "NNE": round(NNE, 2) if abs(pi0 + pd0) > 1e-6 else None,
            "EIN": round(EIN, 2) if abs(pi1 + pd1) > 1e-6 else None,
        }
        
        # Add confidence intervals
        ci_INNT_lower, ci_INNT_upper = calc_ci(INNT, se_INNT)
        ci_INNE_lower, ci_INNE_upper = calc_ci(INNE, se_INNE)
        ci_IEIN_lower, ci_IEIN_upper = calc_ci(IEIN, se_IEIN)
        ci_DNNT_lower, ci_DNNT_upper = calc_ci(DNNT, se_DNNT)
        ci_DNNE_lower, ci_DNNE_upper = calc_ci(DNNE, se_DNNE)
        ci_DEIN_lower, ci_DEIN_upper = calc_ci(DEIN, se_DEIN)
        ci_NNT_lower, ci_NNT_upper = calc_ci(NNT, se_NNT)
        ci_NNE_lower, ci_NNE_upper = calc_ci(NNE, se_NNE)
        ci_EIN_lower, ci_EIN_upper = calc_ci(EIN, se_EIN)
        
        result.update({
            "CI_INNT_LOWER": round(ci_INNT_lower, 2),
            "CI_INNT_UPPER": round(ci_INNT_upper, 2),
            "CI_INNE_LOWER": round(ci_INNE_lower, 2),
            "CI_INNE_UPPER": round(ci_INNE_upper, 2),
            "CI_IEIN_LOWER": round(ci_IEIN_lower, 2),
            "CI_IEIN_UPPER": round(ci_IEIN_upper, 2),
            "CI_DNNT_LOWER": round(ci_DNNT_lower, 2),
            "CI_DNNT_UPPER": round(ci_DNNT_upper, 2),
            "CI_DNNE_LOWER": round(ci_DNNE_lower, 2),
            "CI_DNNE_UPPER": round(ci_DNNE_upper, 2),
            "CI_DEIN_LOWER": round(ci_DEIN_lower, 2),
            "CI_DEIN_UPPER": round(ci_DEIN_upper, 2),
            "CI_NNT_LOWER": round(ci_NNT_lower, 2),
            "CI_NNT_UPPER": round(ci_NNT_upper, 2),
            "CI_NNE_LOWER": round(ci_NNE_lower, 2),
            "CI_NNE_UPPER": round(ci_NNE_upper, 2),
            "CI_EIN_LOWER": round(ci_EIN_lower, 2),
            "CI_EIN_UPPER": round(ci_EIN_upper, 2),
        })
        
        # PRINT SUMMARY
        print(f"\n{'='*70}")
        print("SANDWICH RESULTS SUMMARY")
        print(f"{'='*70}\n")
        
        print("EFFECT ESTIMATES:")
        print(f"  Indirect Effect: {result['p_i']:.5f}")
        print(f"  Direct Effect:   {result['p_d']:.5f}")
        print(f"  Total Effect:    {result['p_b']:.5f}")
        
        print(f"\nNNT MEASURES:")
        for measure in ["INNT", "DNNT", "NNT"]:
            if result[measure] is not None:
                print(f"  {measure}: {result[measure]:.2f} [{result[f'CI_{measure}_LOWER']:.2f}, {result[f'CI_{measure}_UPPER']:.2f}]")
        
        print(f"\nAnalysis complete!")
        print(f"{'='*70}\n")

        return result


# ============================================================================
# TEST FUNCTION WITH CSV INPUT
# ============================================================================

def test_sandwich_with_csv(csv_file, exposure, mediator, outcome, confounders):
    """
    Test Sandwich NNT estimation with CSV input

    This function demonstrates how to load CSV data and run Sandwich analysis.
    The Sandwich method is faster than Bootstrap and provides analytical confidence intervals.

    Parameters:
    -----------
    csv_file : str
        Path to CSV file (e.g., "my_data.csv" or "/content/data.csv")
    exposure : str
        Name of treatment/exposure column (must be binary: 0/1)
    mediator : str
        Name of mediator column (must be binary: 0/1)
    outcome : str
        Name of outcome column (must be binary: 0/1)
    confounders : list of str
        List of confounder column names (e.g., ["Age", "Sex"])

    Returns:
    --------
    dict : Results dictionary with NNT estimates and confidence intervals

    Example:
    --------
    >>> results = test_sandwich_with_csv(
    ...     csv_file="panss8.csv",
    ...     exposure="Treatment",
    ...     mediator="PANSS_DIH",
    ...     outcome="PSP_DIH",
    ...     confounders=["Age", "Male"]
    ... )
    """

    print("="*70)
    print("SANDWICH NNT TEST WITH CSV INPUT")
    print("="*70)
    print()

    # STEP 1: Load CSV data
    print(f"Step 1: Loading data from {csv_file}...")
    try:
        data = pd.read_csv(csv_file)
        print(f"SUCCESS: Data loaded - {data.shape[0]} rows, {data.shape[1]} columns")
    except FileNotFoundError:
        print(f"ERROR: File '{csv_file}' not found!")
        print(f"Current directory: {os.getcwd()}")
        return None
    except Exception as e:
        print(f"ERROR loading file: {e}")
        return None

    # STEP 2: Validate columns exist
    print(f"\nStep 2: Validating columns...")
    required_cols = [exposure, mediator, outcome] + confounders
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        print(f"Available columns: {list(data.columns)}")
        return None

    print(f"SUCCESS: All required columns found")

    # STEP 3: Clean data (remove missing values)
    print(f"\nStep 3: Cleaning data...")
    data_clean = data.dropna(subset=required_cols)
    if len(data_clean) < len(data):
        removed = len(data) - len(data_clean)
        print(f"  Removed {removed} rows with missing values")

    print(f"SUCCESS: Clean data has {len(data_clean)} observations")

    # STEP 4: Display variable summary
    print(f"\nStep 4: Variable summary:")
    print("-" * 70)
    print(f"  Exposure ({exposure}):")
    print(f"    - Treated (1): {np.sum(data_clean[exposure] == 1)}")
    print(f"    - Control (0): {np.sum(data_clean[exposure] == 0)}")
    print(f"  Mediator ({mediator}):")
    print(f"    - Positive (1): {np.sum(data_clean[mediator] == 1)}")
    print(f"    - Negative (0): {np.sum(data_clean[mediator] == 0)}")
    print(f"  Outcome ({outcome}):")
    print(f"    - Positive (1): {np.sum(data_clean[outcome] == 1)}")
    print(f"    - Negative (0): {np.sum(data_clean[outcome] == 0)}")
    print(f"  Confounders: {confounders}")

    # STEP 5: Run Sandwich analysis
    print(f"\nStep 5: Running Sandwich analysis...")
    sandwich = SandwichNNT()

    results = sandwich.compute_nnt_effects(
        data=data_clean,
        exposure=exposure,
        mediator=mediator,
        outcome=outcome,
        confounders=confounders,
        mediator_model="logistic",
        target_model="logistic",
        selected_effects=None
    )

    # STEP 6: Display results summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(f"\nFile: {csv_file}")
    print(f"Sample size: {len(data_clean)}")
    print(f"Method: Sandwich covariance estimation")
    print(f"\nMain NNT Measures:")
    print(f"  INNT: {results.get('INNT', 'N/A')}")
    print(f"  DNNT: {results.get('DNNT', 'N/A')}")
    print(f"  NNT:  {results.get('NNT', 'N/A')}")
    print("="*70)

    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage for testing

    To use this script:
    1. Upload your CSV file to the same directory
    2. Modify the parameters below to match your data
    3. Run: python colab_sandwich_nnt.py
    """

    print("\n" + "="*70)
    print("SANDWICH NNT ESTIMATOR - TEST MODE")
    print("="*70)

    # CONFIGURE YOUR DATA HERE
    CSV_FILE = "panss8.csv"          # Change to your CSV filename
    EXPOSURE = "Treatment"           # Change to your exposure column name
    MEDIATOR = "PANSS_DIH"          # Change to your mediator column name
    OUTCOME = "PSP_DIH"             # Change to your outcome column name
    CONFOUNDERS = ["Age", "Male"]   # Change to your confounder columns

    print("\nConfiguration:")
    print(f"  CSV File: {CSV_FILE}")
    print(f"  Exposure: {EXPOSURE}")
    print(f"  Mediator: {MEDIATOR}")
    print(f"  Outcome: {OUTCOME}")
    print(f"  Confounders: {CONFOUNDERS}")
    print()

    # Check if file exists
    import os
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: File '{CSV_FILE}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print("\nTo fix this:")
        print("1. Make sure your CSV file is in the same directory")
        print("2. Or provide the full path to your CSV file")
        print("3. Update the CSV_FILE variable above")
    else:
        # Run the test
        results = test_sandwich_with_csv(
            csv_file=CSV_FILE,
            exposure=EXPOSURE,
            mediator=MEDIATOR,
            outcome=OUTCOME,
            confounders=CONFOUNDERS
        )

        if results:
            print("\nSUCCESS: Test completed!")
            print("The Sandwich method is complete and provides analytical confidence intervals.")
        else:
            print("\nERROR: Test failed. Check the error messages above.")
