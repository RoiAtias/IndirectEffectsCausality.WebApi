import pandas as pd
import numpy as np

############################
### logit coverage rates ###
############################

# 1. Read the CI data
ci_data = pd.read_csv("all_ci_long_logit.csv")

# 2. True values
true_values = {
    "INNE": 6.525974,
    "IEIN": 6.282517,
    "INNT": 6.372880,
    "DNNE": 3.081796,
    "DEIN": 3.066874,
    "DNNT": 3.072529,
    "NNE": 2.093277,
    "EIN": 2.060850,
    "NNT": 2.073056
}

# 3. Reshape to long and compute coverage
ci_long = ci_data.melt(id_vars=["n", "iteration"], var_name="IndexBound", value_name="Value")

# Extract Index and Bound
ci_long["Index"] = ci_long["IndexBound"].str.replace(r'_[LU]$', '', regex=True)
ci_long["Bound"] = np.where(ci_long["IndexBound"].str.endswith("_L"), "Lower", "Upper")

# Pivot to wide (Lower, Upper)
ci_wide = ci_long.pivot_table(index=["n", "iteration", "Index"], columns="Bound", values="Value").reset_index()

# Exclude full-Inf intervals
ci_wide = ci_wide[~(np.isinf(ci_wide["Lower"]) & np.isinf(ci_wide["Upper"]))]

# Add True value and coverage flag
ci_wide["True"] = ci_wide["Index"].map(true_values)
ci_wide["Covered"] = (ci_wide["Lower"] <= ci_wide["True"]) & (ci_wide["True"] <= ci_wide["Upper"])

# 4. Compute coverage by Index and n
coverage_by_n = ci_wide.groupby(["Index", "n"])["Covered"].mean().reset_index(name="Coverage")

# 5. Pivot to wide format and reorder
coverage_wide = (
    coverage_by_n
    .pivot(index="Index", columns="n", values="Coverage")
    .reindex(list(true_values.keys()))   # סדר לפי true_values
    .reset_index()
)

print("Coverage rates table:")
print(coverage_wide)

#######################
#### reality check ####
#######################

# 6. Count number of full-Inf CI rows per sample size and index
ci_long_inf = ci_data.melt(id_vars=["n", "iteration"], var_name="IndexBound", value_name="Value")
ci_long_inf["Index"] = ci_long_inf["IndexBound"].str.replace(r'_[LU]$', '', regex=True)
ci_long_inf["Bound"] = np.where(ci_long_inf["IndexBound"].str.endswith("_L"), "Lower", "Upper")

ci_wide_inf = ci_long_inf.pivot_table(index=["n", "iteration", "Index"], columns="Bound", values="Value").reset_index()

# Keep only rows where both Lower and Upper are Inf
ci_full_inf = ci_wide_inf[np.isinf(ci_wide_inf["Lower"]) & np.isinf(ci_wide_inf["Upper"])]

# Count by Index and n
inf_count_by_index = ci_full_inf.groupby(["Index", "n"]).size().unstack(fill_value=0)

# Reindex rows and columns to match R order
all_indices = list(true_values.keys())
all_ns = sorted(ci_data["n"].unique())
inf_count_by_index = inf_count_by_index.reindex(index=all_indices, columns=all_ns, fill_value=0)

print("\nFull-Inf CI counts:")
print(inf_count_by_index)

print("\nNumber of observations with both CI limits equal to Inf, by index and sample size:")
print(inf_count_by_index)

print(inf_count_by_index)

##### POINT ESTIMATORS #####
# Read point estimates
est_data = pd.read_csv("all_est_long_logit.csv")

desired_order = ["DEIN", "DNNE", "DNNT", "EIN", "IEIN", "INNT", "NNE", "NNT"]

est_long = est_data.melt(id_vars=["n", "iteration"], var_name="Index", value_name="Estimate")
bad_estimates = est_long[(est_long["Estimate"] < 1) | np.isinf(est_long["Estimate"])]
bad_estimates = bad_estimates[bad_estimates["Index"].isin(desired_order)]

# Count by Index and n
bad_counts = bad_estimates.groupby(["Index", "n"]).size().unstack(fill_value=0)
bad_counts = bad_counts.reindex(desired_order)

print("\nNumber of negative or infinite point estimates by sample size and index:")
print(bad_counts)

# Optional: column sums normalized
normalized_sums = bad_counts.sum(axis=0) / (100 * 9)
print("\nNormalized sums per sample size:")
print(normalized_sums)