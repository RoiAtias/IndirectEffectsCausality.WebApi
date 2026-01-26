import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1. Read the CSV
all_est_long = pd.read_csv("all_est_long_probit.csv")

# 2. Reshape to long format (equivalent to pivot_longer)
est_long_long = all_est_long.melt(
    id_vars=["n", "iteration"],
    value_vars=["INNE", "IEIN", "INNT", "DNNE", "DEIN", "DNNT", "NNE", "EIN", "NNT"],
    var_name="Index",
    value_name="Estimate"
)

# 3. Define true values
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
true_values = {k: round(v, 2) for k, v in true_values.items()}

# 4. Truncate: if < 1 → Inf; if > 3 × true_value → NaN
def truncate(row):
    val = row["Estimate"]
    true_val = true_values[row["Index"]]
    if val < 1:
        return np.inf
    elif val > 3 * true_val:
        return np.nan
    else:
        return val

est_long_long["Estimate"] = est_long_long.apply(truncate, axis=1)

# 5. Ensure sample size is categorical with sorted levels
est_long_long["n"] = pd.Categorical(est_long_long["n"], ordered=True)

# 6. Define 3x3 layout indices
indices_order = [
    ["INNE", "IEIN", "INNT"],
    ["DNNE", "DEIN", "DNNT"],
    ["NNE", "EIN", "NNT"]
]

# 7. Create plots in a 3x3 grid
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(3, 3, figure=fig)

for i, row in enumerate(indices_order):
    for j, index in enumerate(row):
        ax = fig.add_subplot(gs[i, j])
        sns.boxplot(
            data=est_long_long[est_long_long["Index"] == index],
            x="n", y="Estimate", color="lightgrey", ax=ax
        )
        ax.axhline(y=true_values[index], color="red", linestyle="--", linewidth=1)
        ax.set_title(index, fontsize=10)
        ax.set_xlabel("Sample size")
        ax.set_ylabel("Estimator")

# 8. Add two-line title
fig.suptitle(
    "Point Estimators of the Indirect, Direct, and Marginal Indices \n"
    "in Double Probit Model as a Function of the Sample Size",
    fontsize=12
)

# 9. Save as PNG
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("probit_estimator_boxplots.png", dpi=200)

# 10. Show on screen
plt.show()