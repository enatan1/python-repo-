import pandas as pd
import numpy as np
import os

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

# -------------------------------
# 1. Data Loading & Matrix Prep
# -------------------------------
path = os.path.expanduser("~/Desktop/DATA.xlsx")
df = pd.read_excel(path)

# Filter for 1984
df_1984 = df[df["YearCollected"] == 1984].copy()

parasite_cols = df_1984.columns.drop(
    ["YearCollected", "Host", "HostAbundance"]  # adjust if needed
)

# Binary presence
presence = (df_1984[parasite_cols] > 0).astype(int)

# Prevalence matrix: Host × Parasite
prevalence_1984 = presence.groupby(df_1984["Host"]).mean()

host_degree = (prevalence_1984 > 0).sum(axis=1)

host_abundance = (
    df_1984
    .groupby("Host")["HostAbundance"]
    .mean()
)


df_long = (
    prevalence_1984
    .reset_index()
    .melt(id_vars="Host", var_name="Parasite", value_name="prevalence")
)

df_long["host_degree"] = df_long["Host"].map(host_degree)
df_long["host_abundance"] = df_long["Host"].map(host_abundance)


# Drop rows with missing predictors
df_long = df_long.dropna()

# Log transforms (standard in disease ecology)
df_long["log_degree"] = np.log1p(df_long["host_degree"])
df_long["log_abundance"] = np.log(df_long["host_abundance"])


glm = smf.glm(
    formula="prevalence ~ log_degree + log_abundance",
    data=df_long,
    family=sm.families.Binomial()
)

result = glm.fit()
print(result.summary())


# Predictions on observed data (baseline check)
df_long["predicted"] = result.predict(df_long)

rmse = np.sqrt(mean_squared_error(df_long["prevalence"], df_long["predicted"]))
r2 = r2_score(df_long["prevalence"], df_long["predicted"])
rho, _ = spearmanr(df_long["prevalence"], df_long["predicted"])

print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
print(f"Spearman ρ: {rho:.4f}")
