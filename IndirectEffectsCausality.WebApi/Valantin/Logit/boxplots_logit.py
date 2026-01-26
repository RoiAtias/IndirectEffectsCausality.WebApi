import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Read the CSV
all_est_long = pd.read_csv("all_est_long_logit.csv")

# 2. Reshape to long format
est_long_long = all_est_long.melt(
    id_vars=["n"],
    value_vars=["INNE", "IEIN", "INNT", "DNNE", "DEIN", "DNNT", "NNE", "EIN", "NNT"],
    var_name="Index",
    value_name="Estimate"
)

# 3. Define true values
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

true_values = {k: round(v, 2) for k, v in true_values.items()}

# 4. Truncate based on condition
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

# 5. Ensure sample size is treated as categorical
est_long_long["n"] = pd.Categorical(est_long_long["n"], categories=sorted(est_long_long["n"].unique()), ordered=True)

# 6. Define indices order for 3x3 layout
indices_order = [
    ["INNE", "IEIN", "INNT"],
    ["DNNE", "DEIN", "DNNT"],
    ["NNE",  "EIN",  "NNT"]
]

# 7. Create 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for i, row in enumerate(indices_order):
    for j, index in enumerate(row):
        ax = axes[i, j]
        sns.boxplot(
            data=est_long_long[est_long_long["Index"] == index],
            x="n", y="Estimate",
            color="lightgrey",
            ax=ax
        )
        # add true value line
        ax.axhline(true_values[index], color="red", linestyle="--", linewidth=1)
        ax.set_title(index, fontsize=12)
        ax.set_xlabel("Sample size")
        ax.set_ylabel("Estimator")

# 8. Add main title
fig.suptitle(
    "Point Estimators of the Indirect, Direct, and Marginal Indices \n in Double Logit Model as a Function of the Sample Size",
    fontsize=14
)

# 9. Save figure
plt.savefig("logit_estimator_boxplots.png", dpi=10)
plt.show()