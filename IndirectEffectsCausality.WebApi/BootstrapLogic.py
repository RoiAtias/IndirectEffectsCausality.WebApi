import os
import asyncio
import logging
import json
import pandas as pd
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

class BootstrapLogic():  
   def __init__(self):
      logging.getLogger('asyncio').setLevel(logging.WARNING)

   def compute_nnt_effects(self, data, exposure, mediator, outcome, confounders, mediator_model, target_model, B=5000, selected_effects=None):
        A_data = data[exposure].values
        M_data = data[mediator].values
        Y_data = data[outcome].values
        N = len(data)

        # Confounder values
        L_data = {conf: data[conf].values for conf in confounders}
        L_formula = " + ".join(confounders)

        # Regression formulas
        mediator_formula = f"{mediator} ~ {exposure} + {L_formula}"
        outcome_formula = f"{outcome} ~ {exposure} + {mediator} + {L_formula}"

        # Define which effects to compute based on selection
        if selected_effects and len(selected_effects) > 0:
            effects_to_compute = set(selected_effects)
            print(f"Bootstrap - Computing only selected effects: {effects_to_compute}")
        else:
            # Default: compute all effects
            effects_to_compute = {"DNNT", "INNT", "NNT", "DNNE", "INNE", "NNE", "DEIN", "IEIN", "EIN"}
            print("Bootstrap - Computing all effects")

        # Determine which probability components we actually need
        need_indirect = any(effect in effects_to_compute for effect in ["INNT", "INNE", "IEIN"])
        need_direct = any(effect in effects_to_compute for effect in ["DNNT", "DNNE", "DEIN"])
        need_total = any(effect in effects_to_compute for effect in ["NNT", "NNE", "EIN"])

        # Always need basic probabilities for result structure
        need_p_i0 = "INNE" in effects_to_compute or "NNE" in effects_to_compute
        need_p_i1 = "IEIN" in effects_to_compute or "EIN" in effects_to_compute
        need_p_d0 = "DNNE" in effects_to_compute or "NNE" in effects_to_compute
        need_p_d1 = "DEIN" in effects_to_compute or "EIN" in effects_to_compute

        print(f"Optimization: indirect={need_indirect}, direct={need_direct}, total={need_total}")

        # Bootstrap results storage - only for needed components
        probability_columns = ["p_i", "p_d", "p_b"]
        if need_p_i0: probability_columns.append("p_i0")
        if need_p_i1: probability_columns.append("p_i1")
        if need_p_d0: probability_columns.append("p_d0") 
        if need_p_d1: probability_columns.append("p_d1")

        bootstrap_columns = probability_columns + list(effects_to_compute)
        BS_mat = pd.DataFrame(np.nan, index=range(B), columns=bootstrap_columns)

        print(f"DataFrame columns optimized: {len(bootstrap_columns)} instead of {17}")

        for i in range(B):
            ind = np.random.choice(N, size=N, replace=True)
            data_b = data.iloc[ind]

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

            # Update L_data for bootstrap sample
            L_data_b = {conf: data_b[conf].values for conf in confounders}
            A_data_b = data_b[exposure].values

            # Only compute indirect components if needed
            p_i, p_i0, p_i1 = 0, 0, 0
            if need_indirect or need_total:
                # Calculate pimL(L) - indirect effect mediator component
                def pred_pim(a_val):
                    eta = (mdtr_b['Intercept'] + 
                           mdtr_b[exposure] * a_val + 
                           sum(mdtr_b[conf] * L_data_b[conf] for conf in confounders))
                    # Use appropriate link function for mediator model
                    if mediator_model == "logistic":
                        return plogis(eta)
                    else:  # probit
                        return norm.cdf(eta)

                pimL = pred_pim(1) - pred_pim(0)

                # Calculate pioML(L) - indirect effect outcome component
                def pred_pio_ml(m_val):
                    eta = (drct_b['Intercept'] + 
                           drct_b[exposure] * 0 + 
                           drct_b[mediator] * m_val + 
                           sum(drct_b[conf] * L_data_b[conf] for conf in confounders))
                    # Use appropriate link function for target model
                    if target_model == "logistic":
                        return plogis(eta)
                    else:  # probit
                        return norm.cdf(eta)

                pioML = pred_pio_ml(1) - pred_pio_ml(0)

                # Group masks and proportions
                A0_mask = A_data_b == 0
                A1_mask = A_data_b == 1
                prop_A0 = np.mean(A0_mask)
                prop_A1 = np.mean(A1_mask)

                # Only compute needed indirect effects
                if need_p_i0 or need_indirect:
                    p_i0 = np.mean(pimL[A0_mask]) * np.mean(pioML[A0_mask]) if A0_mask.sum() > 0 else 0
                if need_p_i1 or need_indirect:
                    p_i1 = np.mean(pimL[A1_mask]) * np.mean(pioML[A1_mask]) if A1_mask.sum() > 0 else 0
                if need_indirect or need_total:
                    p_i = p_i0 * prop_A0 + p_i1 * prop_A1

            # Only compute direct components if needed  
            p_d, p_d0, p_d1 = 0, 0, 0
            if need_direct or need_total:
                # pioAM(L) - direct effect components
                def pred_pio_am(a_val, m_val):
                    eta = (drct_b['Intercept'] + 
                           drct_b[exposure] * a_val + 
                           drct_b[mediator] * m_val + 
                           sum(drct_b[conf] * L_data_b[conf] for conf in confounders))
                    # Use appropriate link function for target model
                    if target_model == "logistic":
                        return plogis(eta)
                    else:  # probit
                        return norm.cdf(eta)

                pioAM0L = pred_pio_am(1, 0) - pred_pio_am(0, 0)
                pioAM1L = pred_pio_am(1, 1) - pred_pio_am(0, 1)

                # P(M=1 | A=1, L) - reuse from indirect if available, otherwise compute
                if 'pred_pim' not in locals():
                    def pred_pim(a_val):
                        eta = (mdtr_b['Intercept'] + 
                               mdtr_b[exposure] * a_val + 
                               sum(mdtr_b[conf] * L_data_b[conf] for conf in confounders))
                        # Use appropriate link function for mediator model
                        if mediator_model == "logistic":
                            return plogis(eta)
                        else:  # probit
                            return norm.cdf(eta)
                
                P_M1_A1 = pred_pim(1)
            
                def mean_am_parts(mask):
                    return np.mean(
                        pioAM0L[mask] * (1 - P_M1_A1[mask]) +
                        pioAM1L[mask] * P_M1_A1[mask]
                    ) if mask.sum() > 0 else 0
                
                # Only compute needed direct effects
                if 'A0_mask' not in locals():
                    A0_mask = A_data_b == 0
                    A1_mask = A_data_b == 1
                    prop_A0 = np.mean(A0_mask)
                    prop_A1 = np.mean(A1_mask)
                
                if need_p_d0 or need_direct:
                    p_d0 = mean_am_parts(A0_mask)
                if need_p_d1 or need_direct:
                    p_d1 = mean_am_parts(A1_mask)
                if need_direct or need_total:
                    p_d = p_d0 * prop_A0 + p_d1 * prop_A1
            
            # Compute total effects only if needed
            p_b = 0
            if need_total:
                p_b = p_i + p_d
                
            # Store probability components - only those we computed
            BS_mat.loc[i, "p_i"] = p_i
            BS_mat.loc[i, "p_d"] = p_d
            BS_mat.loc[i, "p_b"] = p_b
            if need_p_i0: BS_mat.loc[i, "p_i0"] = p_i0
            if need_p_i1: BS_mat.loc[i, "p_i1"] = p_i1
            if need_p_d0: BS_mat.loc[i, "p_d0"] = p_d0
            if need_p_d1: BS_mat.loc[i, "p_d1"] = p_d1

            # Calculate NNT measures - ONLY for selected effects
            # Handle both positive and negative effects
            def calc_nnt_bootstrap(p_val):
                """Calculate NNT for bootstrap, handling positive and negative values"""
                if abs(p_val) < 0.0001:  # Too close to zero
                    return np.nan
                return 1 / p_val

            if "INNT" in effects_to_compute:
                BS_mat.loc[i, "INNT"] = calc_nnt_bootstrap(p_i)
            if "INNE" in effects_to_compute:
                BS_mat.loc[i, "INNE"] = calc_nnt_bootstrap(p_i0)
            if "IEIN" in effects_to_compute:
                BS_mat.loc[i, "IEIN"] = calc_nnt_bootstrap(p_i1)
            if "DNNT" in effects_to_compute:
                BS_mat.loc[i, "DNNT"] = calc_nnt_bootstrap(p_d)
            if "DNNE" in effects_to_compute:
                BS_mat.loc[i, "DNNE"] = calc_nnt_bootstrap(p_d0)
            if "DEIN" in effects_to_compute:
                BS_mat.loc[i, "DEIN"] = calc_nnt_bootstrap(p_d1)
            if "NNT" in effects_to_compute:
                BS_mat.loc[i, "NNT"] = calc_nnt_bootstrap(p_b)
            if "NNE" in effects_to_compute:
                p_b0 = p_i0 + p_d0
                BS_mat.loc[i, "NNE"] = calc_nnt_bootstrap(p_b0)
            if "EIN" in effects_to_compute:
                p_b1 = p_i1 + p_d1
                BS_mat.loc[i, "EIN"] = calc_nnt_bootstrap(p_b1)

        print(f"Completed {B} bootstrap iterations with optimized computations")

        # Calculate confidence intervals for all measures
        ci_lower = BS_mat.quantile(0.025)
        ci_upper = BS_mat.quantile(0.975)
        means = BS_mat.mean()

        # Prepare results dictionary - include defaults for missing components
        result = {
            # Probability components
            "p_i": round(means.get("p_i", 0), 5),
            "p_d": round(means.get("p_d", 0), 5), 
            "p_b": round(means.get("p_b", 0), 5),
            "p_i0": round(means.get("p_i0", 0), 5),
            "p_i1": round(means.get("p_i1", 0), 5),
            "p_d0": round(means.get("p_d0", 0), 5),
            "p_d1": round(means.get("p_d1", 0), 5),
        }

        # Add NNT measures and confidence intervals - only for computed effects
        all_measures = ["INNT", "DNNT", "NNT", "INNE", "IEIN", "DNNE", "DEIN", "NNE", "EIN"]

        for measure in all_measures:
            if measure in effects_to_compute:
                # Include the computed measure
                result[measure] = round(means[measure], 2) if not np.isnan(means[measure]) else None
                result[f"CI_{measure}_LOWER"] = round(ci_lower[measure], 2) if not np.isnan(ci_lower[measure]) else None
                result[f"CI_{measure}_UPPER"] = round(ci_upper[measure], 2) if not np.isnan(ci_upper[measure]) else None
            else:
                # Set to None for non-selected effects
                result[measure] = None
                result[f"CI_{measure}_LOWER"] = None
                result[f"CI_{measure}_UPPER"] = None

        # Full bootstrap matrix for further analysis (now optimized)
        result["Bootstrap"] = BS_mat

        print(f"Bootstrap optimization complete: computed {len(effects_to_compute)} effects instead of 9")
        return result



