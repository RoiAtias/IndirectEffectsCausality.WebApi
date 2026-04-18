"""
Bootstrap Method for NNT Estimation in Mediation Analysis

This module implements the Bootstrap method for estimating causal mediation effects.
Matches the BootstrapLogic.py used in the main application.
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import norm
import statsmodels.formula.api as smf
from scipy.special import expit as plogis
import warnings
warnings.filterwarnings('ignore')


class BootstrapNNT:
    """Bootstrap estimator for NNT in mediation analysis"""
    
    def __init__(self):
        """Initialize the Bootstrap NNT estimator"""
        pass
    
    def compute_nnt_effects(
        self, 
        data, 
        exposure, 
        mediator, 
        outcome, 
        confounders, 
        mediator_model="logistic", 
        target_model="logistic", 
        B=5000,
        selected_effects=None
    ):
        """
        Compute NNT effects using Bootstrap method

        Parameters:
        - data: DataFrame containing all variables
        - exposure: Name of exposure/treatment variable (binary: 0/1)
        - mediator: Name of mediator variable (binary: 0/1)
        - outcome: Name of outcome variable (binary: 0/1)
        - confounders: List of confounder variable names
        - mediator_model: "logistic" or "probit"
        - target_model: "logistic" or "probit"
        - B: Number of bootstrap iterations (default=5000)
        - selected_effects: Which effects to compute (default=None means all)

        Returns: Dictionary with NNT estimates and confidence intervals
        """
        
        print(f"\n{'='*70}")
        print("BOOTSTRAP METHOD FOR NNT ESTIMATION")
        print(f"{'='*70}\n")
        
        # STEP 1: DATA EXTRACTION
        print("Step 1: Extracting and validating data...")
        
        A_data = data[exposure].values
        M_data = data[mediator].values
        Y_data = data[outcome].values
        N = len(data)
        
        # Validate binary variables
        for var_name, var_data in [(exposure, A_data), (mediator, M_data), (outcome, Y_data)]:
            unique_vals = np.unique(var_data)
            if not np.array_equal(unique_vals, [0, 1]) and not np.array_equal(unique_vals, [0]) and not np.array_equal(unique_vals, [1]):
                print(f"WARNING: {var_name} should be binary (0/1). Found: {unique_vals}")
        
        print(f"Sample size: {N} observations")
        print(f"Exposure ({exposure}): {np.sum(A_data==1)} treated, {np.sum(A_data==0)} control")
        
        # Handle confounders
        L_data = {conf: data[conf].values for conf in confounders}
        L_formula = " + ".join(confounders)
        print(f"Confounders: {confounders}")
        
        # STEP 2: DETERMINE EFFECTS TO COMPUTE
        if selected_effects and len(selected_effects) > 0:
            effects_to_compute = set(selected_effects)
            print(f"\nComputing selected effects: {effects_to_compute}")
        else:
            effects_to_compute = {"DNNT", "INNT", "NNT", "DNNE", "INNE", "NNE", "DEIN", "IEIN", "EIN"}
            print(f"\nComputing all 9 NNT measures")
        
        need_indirect = any(effect in effects_to_compute for effect in ["INNT", "INNE", "IEIN"])
        need_direct = any(effect in effects_to_compute for effect in ["DNNT", "DNNE", "DEIN"])
        need_total = any(effect in effects_to_compute for effect in ["NNT", "NNE", "EIN"])
        
        # STEP 3: PREPARE BOOTSTRAP STORAGE
        print(f"\nStep 2: Preparing bootstrap storage for {B} iterations...")
        
        probability_columns = ["p_i", "p_d", "p_b", "p_i0", "p_i1", "p_d0", "p_d1"]
        bootstrap_columns = probability_columns + list(effects_to_compute)
        BS_mat = pd.DataFrame(np.nan, index=range(B), columns=bootstrap_columns)
        
        # STEP 4: BOOTSTRAP ITERATIONS
        print(f"\nStep 3: Running {B} bootstrap iterations...")
        print("This may take a few minutes...")
        
        mediator_formula = f"{mediator} ~ {exposure} + {L_formula}"
        outcome_formula = f"{outcome} ~ {exposure} + {mediator} + {L_formula}"
        
        progress_points = [int(B * p / 10) for p in range(1, 11)]
        
        for i in range(B):
            if i in progress_points:
                pct = int((i / B) * 100)
                print(f"  Progress: {pct}% ({i}/{B} iterations)")
            
            # Resample data with replacement
            ind = np.random.choice(N, size=N, replace=True)
            data_b = data.iloc[ind]
            
            # Fit models
            if mediator_model == "logistic":
                mediator_model_b = smf.logit(mediator_formula, data=data_b).fit(disp=0)
            else:
                mediator_model_b = smf.probit(mediator_formula, data=data_b).fit(disp=0)
            
            if target_model == "logistic":
                outcome_model_b = smf.logit(outcome_formula, data=data_b).fit(disp=0)
            else:
                outcome_model_b = smf.probit(outcome_formula, data=data_b).fit(disp=0)
            
            mdtr_b = mediator_model_b.params
            drct_b = outcome_model_b.params
            
            L_data_b = {conf: data_b[conf].values for conf in confounders}
            A_data_b = data_b[exposure].values
            
            # Calculate indirect effect
            p_i, p_i0, p_i1 = 0, 0, 0
            
            if need_indirect or need_total:
                def pred_mediator(a_val):
                    eta = (mdtr_b['Intercept'] + 
                           mdtr_b[exposure] * a_val + 
                           sum(mdtr_b[conf] * L_data_b[conf] for conf in confounders))
                    if mediator_model == "logistic":
                        return plogis(eta)
                    else:
                        return norm.cdf(eta)
                
                pimL = pred_mediator(1) - pred_mediator(0)
                
                def pred_outcome_indirect(m_val):
                    eta = (drct_b['Intercept'] + 
                           drct_b[exposure] * 0 + 
                           drct_b[mediator] * m_val + 
                           sum(drct_b[conf] * L_data_b[conf] for conf in confounders))
                    if target_model == "logistic":
                        return plogis(eta)
                    else:
                        return norm.cdf(eta)
                
                pioML = pred_outcome_indirect(1) - pred_outcome_indirect(0)
                
                A0_mask = A_data_b == 0
                A1_mask = A_data_b == 1
                prop_A0 = np.mean(A0_mask)
                prop_A1 = np.mean(A1_mask)
                
                p_i0 = np.mean(pimL[A0_mask]) * np.mean(pioML[A0_mask]) if A0_mask.sum() > 0 else 0
                p_i1 = np.mean(pimL[A1_mask]) * np.mean(pioML[A1_mask]) if A1_mask.sum() > 0 else 0
                p_i = p_i0 * prop_A0 + p_i1 * prop_A1
            
            # Calculate direct effect
            p_d, p_d0, p_d1 = 0, 0, 0
            
            if need_direct or need_total:
                def pred_outcome_direct(a_val, m_val):
                    eta = (drct_b['Intercept'] + 
                           drct_b[exposure] * a_val + 
                           drct_b[mediator] * m_val + 
                           sum(drct_b[conf] * L_data_b[conf] for conf in confounders))
                    if target_model == "logistic":
                        return plogis(eta)
                    else:
                        return norm.cdf(eta)
                
                pioAM0L = pred_outcome_direct(1, 0) - pred_outcome_direct(0, 0)
                pioAM1L = pred_outcome_direct(1, 1) - pred_outcome_direct(0, 1)
                P_M1_A1 = pred_mediator(1)
                
                def mean_direct_parts(mask):
                    return np.mean(
                        pioAM0L[mask] * (1 - P_M1_A1[mask]) +
                        pioAM1L[mask] * P_M1_A1[mask]
                    ) if mask.sum() > 0 else 0
                
                if 'A0_mask' not in locals():
                    A0_mask = A_data_b == 0
                    A1_mask = A_data_b == 1
                    prop_A0 = np.mean(A0_mask)
                    prop_A1 = np.mean(A1_mask)
                
                p_d0 = mean_direct_parts(A0_mask)
                p_d1 = mean_direct_parts(A1_mask)
                p_d = p_d0 * prop_A0 + p_d1 * prop_A1
            
            # Calculate total effect
            p_b = 0
            if need_total:
                p_b = p_i + p_d
            
            # Store probability components
            BS_mat.loc[i, "p_i"] = p_i
            BS_mat.loc[i, "p_d"] = p_d
            BS_mat.loc[i, "p_b"] = p_b
            BS_mat.loc[i, "p_i0"] = p_i0
            BS_mat.loc[i, "p_i1"] = p_i1
            BS_mat.loc[i, "p_d0"] = p_d0
            BS_mat.loc[i, "p_d1"] = p_d1
            
            # Calculate NNT measures
            def calc_nnt(p_val):
                if abs(p_val) < 0.0001:
                    return np.nan
                return 1 / p_val
            
            if "INNT" in effects_to_compute:
                BS_mat.loc[i, "INNT"] = calc_nnt(p_i)
            if "INNE" in effects_to_compute:
                BS_mat.loc[i, "INNE"] = calc_nnt(p_i0)
            if "IEIN" in effects_to_compute:
                BS_mat.loc[i, "IEIN"] = calc_nnt(p_i1)
            if "DNNT" in effects_to_compute:
                BS_mat.loc[i, "DNNT"] = calc_nnt(p_d)
            if "DNNE" in effects_to_compute:
                BS_mat.loc[i, "DNNE"] = calc_nnt(p_d0)
            if "DEIN" in effects_to_compute:
                BS_mat.loc[i, "DEIN"] = calc_nnt(p_d1)
            if "NNT" in effects_to_compute:
                BS_mat.loc[i, "NNT"] = calc_nnt(p_b)
            if "NNE" in effects_to_compute:
                p_b0 = p_i0 + p_d0
                BS_mat.loc[i, "NNE"] = calc_nnt(p_b0)
            if "EIN" in effects_to_compute:
                p_b1 = p_i1 + p_d1
                BS_mat.loc[i, "EIN"] = calc_nnt(p_b1)
        
        print(f"\nCompleted {B} bootstrap iterations!")
        
        # STEP 5: CALCULATE CONFIDENCE INTERVALS
        print(f"\nStep 4: Calculating confidence intervals...")
        
        ci_lower = BS_mat.quantile(0.025)
        ci_upper = BS_mat.quantile(0.975)
        means = BS_mat.mean()
        
        # STEP 6: BUILD RESULTS
        result = {
            "p_i": round(means.get("p_i", 0), 5),
            "p_d": round(means.get("p_d", 0), 5),
            "p_b": round(means.get("p_b", 0), 5),
            "p_i0": round(means.get("p_i0", 0), 5),
            "p_i1": round(means.get("p_i1", 0), 5),
            "p_d0": round(means.get("p_d0", 0), 5),
            "p_d1": round(means.get("p_d1", 0), 5),
        }
        
        all_measures = ["INNT", "DNNT", "NNT", "INNE", "IEIN", "DNNE", "DEIN", "NNE", "EIN"]
        
        for measure in all_measures:
            if measure in effects_to_compute:
                result[measure] = round(means[measure], 2) if not np.isnan(means[measure]) else None
                result[f"CI_{measure}_LOWER"] = round(ci_lower[measure], 2) if not np.isnan(ci_lower[measure]) else None
                result[f"CI_{measure}_UPPER"] = round(ci_upper[measure], 2) if not np.isnan(ci_upper[measure]) else None
            else:
                result[measure] = None
                result[f"CI_{measure}_LOWER"] = None
                result[f"CI_{measure}_UPPER"] = None
        
        result["Bootstrap"] = BS_mat
        
        # PRINT SUMMARY
        print(f"\n{'='*70}")
        print("BOOTSTRAP RESULTS SUMMARY")
        print(f"{'='*70}\n")
        
        print("EFFECT ESTIMATES:")
        print(f"  Indirect Effect: {result['p_i']:.5f}")
        print(f"  Direct Effect:   {result['p_d']:.5f}")
        print(f"  Total Effect:    {result['p_b']:.5f}")
        
        print(f"\nNNT MEASURES:")
        for measure in ["INNT", "DNNT", "NNT"]:
            if measure in effects_to_compute and result[measure] is not None:
                print(f"  {measure}: {result[measure]:.2f} [{result[f'CI_{measure}_LOWER']:.2f}, {result[f'CI_{measure}_UPPER']:.2f}]")
        
        print(f"\nAnalysis complete!")
        print(f"{'='*70}\n")

        return result


# ============================================================================
# TEST FUNCTION WITH CSV INPUT
# ============================================================================

def test_bootstrap_with_csv(csv_file, exposure, mediator, outcome, confounders, B=1000):
    """
    Test Bootstrap NNT estimation with CSV input

    This function demonstrates how to load CSV data and run Bootstrap analysis.
    Perfect for testing with your own data before running full B=5000 analysis.

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
    B : int
        Number of bootstrap iterations (default: 1000 for quick testing)
        For publication, use B=5000 or higher

    Returns:
    --------
    dict : Results dictionary with NNT estimates and confidence intervals

    Example:
    --------
    >>> results = test_bootstrap_with_csv(
    ...     csv_file="panss8.csv",
    ...     exposure="Treatment",
    ...     mediator="PANSS_DIH",
    ...     outcome="PSP_DIH",
    ...     confounders=["Age", "Male"],
    ...     B=1000
    ... )
    """

    print("="*70)
    print("BOOTSTRAP NNT TEST WITH CSV INPUT")
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

    # STEP 5: Run Bootstrap analysis
    print(f"\nStep 5: Running Bootstrap analysis with B={B} iterations...")
    bootstrap = BootstrapNNT()

    results = bootstrap.compute_nnt_effects(
        data=data_clean,
        exposure=exposure,
        mediator=mediator,
        outcome=outcome,
        confounders=confounders,
        mediator_model="logistic",
        target_model="logistic",
        B=B,
        selected_effects=None
    )

    # STEP 6: Display results summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(f"\nFile: {csv_file}")
    print(f"Sample size: {len(data_clean)}")
    print(f"Bootstrap iterations: {B}")
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
    3. Run: python colab_bootstrap_nnt.py
    """

    print("\n" + "="*70)
    print("BOOTSTRAP NNT ESTIMATOR - TEST MODE")
    print("="*70)

    # CONFIGURE YOUR DATA HERE
    CSV_FILE = "panss8.csv"          # Change to your CSV filename
    EXPOSURE = "Treatment"           # Change to your exposure column name
    MEDIATOR = "PANSS_DIH"          # Change to your mediator column name
    OUTCOME = "PSP_DIH"             # Change to your outcome column name
    CONFOUNDERS = ["Age", "Male"]   # Change to your confounder columns
    B_ITERATIONS = 1000             # Use 1000 for testing, 5000+ for publication

    print("\nConfiguration:")
    print(f"  CSV File: {CSV_FILE}")
    print(f"  Exposure: {EXPOSURE}")
    print(f"  Mediator: {MEDIATOR}")
    print(f"  Outcome: {OUTCOME}")
    print(f"  Confounders: {CONFOUNDERS}")
    print(f"  Bootstrap iterations: {B_ITERATIONS}")
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
        results = test_bootstrap_with_csv(
            csv_file=CSV_FILE,
            exposure=EXPOSURE,
            mediator=MEDIATOR,
            outcome=OUTCOME,
            confounders=CONFOUNDERS,
            B=B_ITERATIONS
        )

        if results:
            print("\nSUCCESS: Test completed!")
            print("You can now use these results or run with higher B for publication.")
        else:
            print("\nERROR: Test failed. Check the error messages above.")
