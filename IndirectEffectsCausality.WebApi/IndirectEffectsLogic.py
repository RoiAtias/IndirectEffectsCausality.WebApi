import pandas as pd
from BootstrapLogic import BootstrapLogic
from SandwichLogic import SandwichLogic

class IndirectEffectsLogic():  
   def __init__(self):
      self.bootstrap_logic = BootstrapLogic()
      self.sandwich_logic = SandwichLogic()

   def process_csv_file(self, file):
        df = pd.read_csv(file)
        result = {
            "Confounders": df.columns.tolist(),  
            "Predictor": df.columns.tolist(),  
            "Mediator": df.columns.tolist(),  
            "Target_Variable": df.columns.tolist()
        }
        return result

   def process_results(self, file, confounders, predictor_x, mediator_y, target_variable, mediator_model, target_model, ci_method, selected_effects=None):
        
        data = pd.read_csv(file)

        confounders_list = eval(confounders)
        
        # Parse selected effects if provided
        selected_effects_list = []
        if selected_effects:
            try:
                import json
                selected_effects_list = json.loads(selected_effects)
                print(f"Selected effects to calculate: {selected_effects_list}")
            except:
                try:
                    selected_effects_list = eval(selected_effects)
                    print(f"Selected effects to calculate (via eval): {selected_effects_list}")
                except:
                    selected_effects_list = []  # Default to all effects if parsing fails
                    print("Could not parse selected_effects, calculating all effects")
        else:
            print("No selected_effects provided, calculating all effects")
        
        n_iterations = 5

        # Choose computation method based on ci_method parameter
        if ci_method == 'bootstrap':
            result = self.bootstrap_logic.compute_nnt_effects(data=data,
                                          exposure=predictor_x,
                                          mediator= mediator_y,
                                          outcome= target_variable,
                                          confounders= confounders_list,
                                          mediator_model = mediator_model,
                                          target_model = target_model,
                                          B=n_iterations,
                                          selected_effects=selected_effects_list)
        elif ci_method == 'sandwich':
            result = self.sandwich_logic.compute_nnt_effects(data=data,
                                          exposure=predictor_x,
                                          mediator= mediator_y,
                                          outcome= target_variable,
                                          confounders= confounders_list,
                                          mediator_model = mediator_model,
                                          target_model = target_model,
                                          selected_effects=selected_effects_list)
        else:
            # Default to bootstrap if method is not recognized
            result = self.bootstrap_logic.compute_nnt_effects(data=data,
                                          exposure=predictor_x,
                                          mediator= mediator_y,
                                          outcome= target_variable,
                                          confounders= confounders_list,
                                          mediator_model = mediator_model,
                                          target_model = target_model,
                                          B=n_iterations,
                                          selected_effects=selected_effects_list)

        # Enhanced results dictionary with all NNT measures - use get() for optional values
        results = {
            # Original results maintained for backward compatibility
            "indirect_effect": result["p_i"],
            "total_effect": result["p_d"],  # Note: this was mislabeled before
            "direct_effect": result["p_b"], # Note: this was mislabeled before
            "innt": result.get("INNT"),
            "dnnt": result.get("DNNT"),
            "nnt": result.get("NNT"),
            "nnt_confidence_interval_lower": result.get("CI_NNT_LOWER"),
            "nnt_confidence_interval_upper": result.get("CI_NNT_UPPER"),
            "innt_confidence_interval_lower": result.get("CI_INNT_LOWER"),
            "innt_confidence_interval_upper": result.get("CI_INNT_UPPER"),
            "dnnt_confidence_interval_lower": result.get("CI_DNNT_LOWER"),
            "dnnt_confidence_interval_upper": result.get("CI_DNNT_UPPER"),
            
            # Enhanced results with exposure group-specific measures
            "indirect_effect_a0": result["p_i0"],
            "indirect_effect_a1": result["p_i1"],
            "direct_effect_a0": result["p_d0"],
            "direct_effect_a1": result["p_d1"],
            
            # Additional NNT measures
            "inne": result.get("INNE"),  # Indirect Number Needed to Expose (A=0)
            "iein": result.get("IEIN"),  # Indirect Effect when Intervening (A=1)
            "dnne": result.get("DNNE"),  # Direct Number Needed to Expose (A=0)
            "dein": result.get("DEIN"),  # Direct Effect when Intervening (A=1)
            "nne": result.get("NNE"),    # Total Number Needed to Expose (A=0)
            "ein": result.get("EIN"),    # Total Effect when Intervening (A=1)
            
            # Confidence intervals for additional measures
            "inne_confidence_interval_lower": result.get("CI_INNE_LOWER"),
            "inne_confidence_interval_upper": result.get("CI_INNE_UPPER"),
            "iein_confidence_interval_lower": result.get("CI_IEIN_LOWER"),
            "iein_confidence_interval_upper": result.get("CI_IEIN_UPPER"),
            "dnne_confidence_interval_lower": result.get("CI_DNNE_LOWER"),
            "dnne_confidence_interval_upper": result.get("CI_DNNE_UPPER"),
            "dein_confidence_interval_lower": result.get("CI_DEIN_LOWER"),
            "dein_confidence_interval_upper": result.get("CI_DEIN_UPPER"),
            "nne_confidence_interval_lower": result.get("CI_NNE_LOWER"),
            "nne_confidence_interval_upper": result.get("CI_NNE_UPPER"),
            "ein_confidence_interval_lower": result.get("CI_EIN_LOWER"),
            "ein_confidence_interval_upper": result.get("CI_EIN_UPPER"),
            
            # Add method indicator to results
            "ci_method_used": ci_method if ci_method in ['bootstrap', 'sandwich'] else 'bootstrap',
            "selected_effects": selected_effects_list  # Return what was selected for debugging
        }

        return results