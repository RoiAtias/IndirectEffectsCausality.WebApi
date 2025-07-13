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

class IndirectEffectsLogic():  
   def __init__(self):
      logging.getLogger('asyncio').setLevel(logging.WARNING)

   def process_csv_file(self, file):
        df = pd.read_csv(file)
        result = {
            "Confounders": df.columns.tolist(),  
            "Predictor": df.columns.tolist(),  
            "Mediator": df.columns.tolist(),  
            "Target_Variable": df.columns.tolist()
        }
        return result

   def compute_nnt_effects(self,data, exposure, mediator, outcome, confounders,mediator_model,target_model, B=1):
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

        # Bootstrap results storage
        BS_mat = pd.DataFrame(np.nan, index=range(B), columns=["DNNT", "INNT", "NNT"])
    
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
        
            # Calculate pimL(L)
            def pred_pim(a_val):
                return plogis(
                    mdtr_b['Intercept'] + 
                    mdtr_b[exposure] * a_val + 
                    sum(mdtr_b[conf] * L_data[conf] for conf in confounders)
                )
        
            pimL = pred_pim(1) - pred_pim(0)
        
            # Calculate pioML(L)
            def pred_pio_ml(m_val):
                return plogis(
                    drct_b['Intercept'] + 
                    drct_b[exposure] * 0 + 
                    drct_b[mediator] * m_val + 
                    sum(drct_b[conf] * L_data[conf] for conf in confounders)
                )
        
            pioML = pred_pio_ml(1) - pred_pio_ml(0)
        
            A0_mask = A_data == 0
            A1_mask = A_data == 1
            prop_A0 = np.mean(A0_mask)
            prop_A1 = np.mean(A1_mask)

            p_i0 = np.mean(pimL[A0_mask]) * np.mean(pioML[A0_mask]) if A0_mask.sum() > 0 else 0
            p_i1 = np.mean(pimL[A1_mask]) * np.mean(pioML[A1_mask]) if A1_mask.sum() > 0 else 0
            p_i = p_i0 * prop_A0 + p_i1 * prop_A1

            # pioAM(L)
            def pred_pio_am(a_val, m_val):
                return plogis(
                    drct_b['Intercept'] + 
                    drct_b[exposure] * a_val + 
                    drct_b[mediator] * m_val + 
                    sum(drct_b[conf] * L_data[conf] for conf in confounders)
                )
        
            pioAM0L = pred_pio_am(1, 0) - pred_pio_am(0, 0)
            pioAM1L = pred_pio_am(1, 1) - pred_pio_am(0, 1)
        
            # P(M=1 | A=1, L)
            P_M1_A1 = pred_pim(1)
        
            def mean_am_parts(mask):
                return (
                    np.mean(pioAM0L[mask]) * (1 - np.mean(P_M1_A1[mask])) +
                    np.mean(pioAM1L[mask]) * np.mean(P_M1_A1[mask])
                ) if mask.sum() > 0 else 0
        
            p_d0 = mean_am_parts(A0_mask)
            p_d1 = mean_am_parts(A1_mask)
            p_d = p_d0 * prop_A0 + p_d1 * prop_A1
            p_b = p_i + p_d
        
            BS_mat.loc[i, "INNT"] = 1 / p_i if p_i > 0 else np.nan
            BS_mat.loc[i, "DNNT"] = 1 / p_d if p_d > 0 else np.nan
            BS_mat.loc[i, "NNT"] = 1 / p_b if p_b > 0 else np.nan

        ci_lower = BS_mat.quantile(0.025)
        ci_upper = BS_mat.quantile(0.975)

        return {
            "p_i": round(p_i, 5),
            "p_d": round(p_d, 5),
            "p_b": round(p_b, 5),
            "INNT": round(BS_mat["INNT"].mean(), 2),
            "DNNT": round(BS_mat["DNNT"].mean(), 2),
            "NNT": round(BS_mat["NNT"].mean(), 2),
            "CI_INNT_LOWER": round(ci_lower["INNT"], 2), 
            "CI_INNT_UPPER": round(ci_upper["INNT"], 2),
            "CI_DNNT_LOWER": round(ci_lower["DNNT"], 2),
            "CI_DNNT_UPPER": round(ci_upper["DNNT"], 2),
            "CI_NNT_LOWER": round(ci_lower["NNT"], 2),
            "CI_NNT_UPPER": round(ci_upper["NNT"], 2),
            "Bootstrap": BS_mat
        }

   def process_results(self, file, confounders, predictor_x, mediator_y, target_variable,mediator_model, target_model):
        
        data = pd.read_csv(file)

        confounders_list = eval(confounders)
        
        n_iterations = 5

        result = self.compute_nnt_effects(data=data,
                                      exposure=predictor_x,
                                      mediator= mediator_y,
                                      outcome= target_variable,
                                      confounders= confounders_list,
                                      mediator_model = mediator_model,
                                      target_model = target_model,
                                      B=n_iterations)

        results = {
            "indirect_effect": result["p_i"],
            "total_effect": result["p_d"],
            "direct_effect": result["p_b"],
            "innt": result["INNT"],
            "dnnt": result["DNNT"],
            "nnt": result["NNT"],
            "nnt_confidence_interval_lower": result["CI_NNT_LOWER"],
            "nnt_confidence_interval_upper": result["CI_NNT_UPPER"],
            "innt_confidence_interval_lower": result["CI_INNT_LOWER"],
            "innt_confidence_interval_upper": result["CI_INNT_UPPER"],
            "dnnt_confidence_interval_lower": result["CI_DNNT_LOWER"],
            "dnnt_confidence_interval_upper": result["CI_DNNT_UPPER"],
        }

        return results