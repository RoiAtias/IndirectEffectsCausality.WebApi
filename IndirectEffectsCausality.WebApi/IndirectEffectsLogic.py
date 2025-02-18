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


   def process_results(self, file, confounders, predictor_x, mediator_y, target_variable,mediator_model, target_model):
        
        data = pd.read_csv(file)

        confounders_list = eval(confounders)

        total_effect, a, b, indirect_effect = self.perform_mediation_analysis(data, confounders_list, predictor_x, mediator_y, target_variable,mediator_model,target_model)

        alpha = 0.05
        n_iterations = 10

        indirect_ci, direct_ci, total_ci = self.bootstrap_mediation(
        data, n_iterations=n_iterations, alpha=alpha, confounders=confounders_list, predictor_xm=predictor_x, mediator_ym=mediator_y, target_variable=target_variable,mediator_model = mediator_model,target_model = target_model)

        nnt_median, nnt_ci = self.bootstrap_nnt(data, predictor_xm=predictor_x, target_variable=target_variable, confounders=confounders_list,mediator_model = mediator_model,target_model = target_model, n_iterations=n_iterations, alpha=alpha)

        results = {
            "total_effect": total_effect,
            "effect_of_smoker_on_overweight": a,
            "direct_effect": total_effect,
            "effect_of_overweight_on_heart_disease": b,
            "indirect_effect": indirect_effect,
            "indirect_effect_ci": indirect_ci,
            "direct_effect_ci": direct_ci,
            "total_effect_ci": total_ci,
            "nnt": nnt_median,
            "nnt_confidence_interval": nnt_ci,
        }

        return results

   def fit_model(self,data, formula, model_type="logistic"):
    """Fits a regression model using statsmodels."""
    if model_type == "logistic":
        return smf.logit(formula, data=data).fit(disp=False)
    else:
        return smf.ols(formula, data=data).fit(disp=False)

   def calculate_indirect_effect(self,model_xm, model_ym, predictor_xm, mediator_ym):
        """Calculates the indirect effect (a*b)."""
        a = model_xm.params[predictor_xm]
        b = model_ym.params[mediator_ym]
        return round(a * b, 2)

   def bootstrap_mediation(self,data, n_iterations, alpha, confounders, predictor_xm, mediator_ym, target_variable, mediator_model, target_model):
        """Performs bootstrapping to estimate confidence intervals for mediation effects."""
        indirect_effects, direct_effects, total_effects = [], [], []
        confounders_formula = ' + '.join(confounders)

        for _ in range(n_iterations):
            bootstrap_sample = resample(data, replace=True)
            bootstrap_df = pd.DataFrame(bootstrap_sample, columns=data.columns)

            # Bootstrapped models
            model_xm_boot = self.fit_model(bootstrap_df, f'{mediator_ym} ~ {predictor_xm} + {confounders_formula}', model_type=mediator_model)
            model_ym_boot = self.fit_model(bootstrap_df, f'{target_variable} ~ {predictor_xm} + {confounders_formula} + {mediator_ym}', model_type=target_model)
            model_total_boot = self.fit_model(bootstrap_df, f'{target_variable} ~ {predictor_xm} + {confounders_formula}', model_type=target_model)

            indirect_effect_boot = self.calculate_indirect_effect(model_xm_boot, model_ym_boot, predictor_xm, mediator_ym)
            direct_effect_boot = model_ym_boot.params[predictor_xm]
            total_effect_boot = model_total_boot.params[predictor_xm]

            indirect_effects.append(indirect_effect_boot)
            direct_effects.append(direct_effect_boot)
            total_effects.append(total_effect_boot)

        # Calculate confidence intervals and round to 2 decimal places
        lower_ci_indirect = round(np.percentile(indirect_effects, (alpha/2)*100), 2)
        upper_ci_indirect = round(np.percentile(indirect_effects, (1-alpha/2)*100), 2)

        lower_ci_direct = round(np.percentile(direct_effects, (alpha/2)*100), 2)
        upper_ci_direct = round(np.percentile(direct_effects, (1-alpha/2)*100), 2)

        lower_ci_total = round(np.percentile(total_effects, (alpha/2)*100), 2)
        upper_ci_total = round(np.percentile(total_effects, (1-alpha/2)*100), 2)

        return (lower_ci_indirect, upper_ci_indirect), (lower_ci_direct, upper_ci_direct), (lower_ci_total, upper_ci_total)

   def perform_mediation_analysis(self,data, confounders, predictor_xm, mediator_ym, target_variable, mediator_model, target_model):
        """Performs mediation analysis and returns the effects."""
    
        # Total Effect (c) - השתמש במודל שנשלח כפרמטר
        model_total = self.fit_model(data, f'{target_variable} ~ {predictor_xm} + {" + ".join(confounders)}', model_type=target_model)
        total_effect = round(model_total.params[predictor_xm], 2)

        # Effect of predictor_xm on mediator_ym (a)
        model_xm = self.fit_model(data, f'{mediator_ym} ~ {predictor_xm} + {" + ".join(confounders)}', model_type=mediator_model)
        a = round(model_xm.params[predictor_xm], 2)

        # Direct Effect (c') and Effect of mediator_ym on response (b)
        model_ym = self.fit_model(data, f'{target_variable} ~ {predictor_xm} + {" + ".join(confounders)} + {mediator_ym}', model_type=target_model)
        direct_effect = round(model_ym.params[predictor_xm], 2)
        b = round(model_ym.params[mediator_ym], 2)

        # Indirect Effect (a*b)
        indirect_effect = round(self.calculate_indirect_effect(model_xm, model_ym, predictor_xm, mediator_ym), 2)

        return total_effect, a, b, indirect_effect


   def sigmoid(self,x):
    return 1 / (1 + np.exp(-x))


   def bootstrap_nnt(self,data, predictor_xm, target_variable, confounders,mediator_model, target_model, n_iterations, alpha):
        """Performs bootstrapping to estimate the Number Needed to Treat (NNT) and its confidence intervals."""
        nnt_values = []
        confounders_formula = ' + '.join(confounders)

        for _ in range(n_iterations):
            bootstrap_sample = resample(data, replace=True)
            bootstrap_df = pd.DataFrame(bootstrap_sample, columns=data.columns)

            model = self.fit_model(bootstrap_df, f'{target_variable} ~ {predictor_xm} + {confounders_formula}', model_type=target_model)
        
            p1 = self.sigmoid(model.params['Intercept'] + model.params[predictor_xm])
            p0 = self.sigmoid(model.params['Intercept']) 

            if p1 != p0 and (p1 - p0) != 0:
                nnt = round(1 / abs(p1 - p0), 2)
                nnt_values.append(nnt)

        lower_ci = round(np.percentile(nnt_values, (alpha/2)*100), 2)
        upper_ci = round(np.percentile(nnt_values, (1-alpha/2)*100), 2)
    
        return round(np.median(nnt_values), 2), (lower_ci, upper_ci)

    