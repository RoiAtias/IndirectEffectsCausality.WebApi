import pandas as pd
import numpy as np

# ============================
# Read the CI data
# ============================
ci_data = pd.read_csv("all_ci_long_probit.csv")

# True values
true_values = {
    "INNE": 4.493874,
    "IEIN": 4.176306,
    "INNT": 4.291575,
    "DNNE": 2.062010,
    "DEIN": 2.056743,
    "DNNT": 2.058742,
    "NNE": 1.413450,
    "EIN": 1.378072,
    "NNT": 1.391308
}

# ============================
# Step 1: Reshape and calculate coverage
# ============================
ci_long = ci_data.melt(
    id_vars=["n", "iteration"],
    var_name="IndexBound",
    value_name="Value"
)

ci_long["Index"] = ci_long["IndexBound"].str.replace(r"_[LU]$", "", regex=True)
ci_long["Bound"] = np.where(ci_long["IndexBound"].str.endswith("_L"), "Lower", "Upper")

ci_long = ci_long.drop(columns="IndexBound").pivot_table(
    index=["n", "iteration", "Index"],
    columns="Bound",
    values="Value",
    aggfunc="first"
).reset_index()

# Exclude rows where both bounds are Inf
ci_long = ci_long[~(np.isinf(ci_long["Lower"]) & np.isinf(ci_long["Upper"]))]

# Add true values and coverage flag
ci_long["True"] = ci_long["Index"].map(true_values)
ci_long["Covered"] = (ci_long["True"] >= ci_long["Lower"]) & (ci_long["True"] <= ci_long["Upper"])

# ============================
# Step 2: Compute coverage by index and n
# ============================
coverage_by_n = (
    ci_long.groupby(["Index", "n"])["Covered"]
    .mean()
    .reset_index(name="Coverage")
)

# ============================
# Step 3: Wide format
# ============================
coverage_wide = coverage_by_n.pivot(index="Index", columns="n", values="Coverage").reset_index()

# Step 4: Reorder by true_values vector
coverage_wide["Index"] = pd.Categorical(coverage_wide["Index"], categories=list(true_values.keys()), ordered=True)
coverage_wide = coverage_wide.sort_values("Index")

print("\nCoverage rates table:\n")
print(coverage_wide.to_string(index=False))

# ============================
# Step 6: Count full-Inf CIs
# ============================
inf_count_by_index = ci_data.melt(
    id_vars=["n", "iteration"],
    var_name="IndexBound",
    value_name="Value"
)

inf_count_by_index["Index"] = inf_count_by_index["IndexBound"].str.replace(r"_[LU]$", "", regex=True)
inf_count_by_index["Bound"] = np.where(inf_count_by_index["IndexBound"].str.endswith("_L"), "Lower", "Upper")

inf_count_by_index = inf_count_by_index.drop(columns="IndexBound").pivot_table(
    index=["n", "iteration", "Index"],
    columns="Bound",
    values="Value",
    aggfunc="first"
).reset_index()

inf_count_by_index = inf_count_by_index[
    np.isinf(inf_count_by_index["Lower"]) & np.isinf(inf_count_by_index["Upper"])
]

inf_count_by_index = (
    inf_count_by_index.groupby(["n", "Index"])
    .size()
    .reset_index(name="Num_Full_Inf")
    .pivot(index="Index", columns="n", values="Num_Full_Inf")
    .fillna(0)
    .astype(int)
    .reset_index()
)

print("\nNumber of observations with both CI limits equal to Inf, by index and sample size:\n")
print(inf_count_by_index.to_string(index=False))

# ============================
# POINT ESTIMATORS
# ============================
est_data = pd.read_csv("all_est_long_probit.csv")

desired_order = ["DEIN", "DNNE", "DNNT", "EIN", "IEIN", "INNT", "NNE", "NNT"]

bad_estimates = est_data.melt(
    id_vars=["n", "iteration"],
    var_name="Index",
    value_name="Estimate"
)

bad_estimates = bad_estimates[
    (np.isinf(bad_estimates["Estimate"])) | (bad_estimates["Estimate"] < 1)
]

bad_estimates = (
    bad_estimates.groupby(["Index", "n"])
    .size()
    .reset_index(name="Num_Bad")
    .query("Index in @desired_order")
)

bad_estimates_wide = (
    bad_estimates.pivot(index="Index", columns="n", values="Num_Bad")
    .fillna(0)
    .astype(int)
    .reindex(desired_order)
    .reset_index()
)

print("\nNumber of negative or infinite point estimates by sample size and index:\n")
print(bad_estimates_wide.to_string(index=False))

# Proportion check
prop = bad_estimates_wide.drop(columns="Index").sum().sum() / (9 * 100)
print("\nProportion of bad estimates:", prop)