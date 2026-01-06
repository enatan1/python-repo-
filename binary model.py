import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from pandasgui import show
# -------------------------------
# 1. Data Loading & Matrix Prep
# -------------------------------
path = os.path.expanduser("~/Desktop/DATA.xlsx")
df = pd.read_excel(path)

# Filter for the specific year (1984)
df_1984 = df[df["YearCollected"] == 1984].copy()
parasite_cols = df_1984.columns.drop(["YearCollected", "Host"])
presence_matrix = (df_1984[parasite_cols] > 0).astype(int)
prevalence_1984 = presence_matrix.groupby(df_1984["Host"]).mean()
#show(prevalence_1984)
A = prevalence_1984.values
hosts = prevalence_1984.index.to_numpy()
parasites = prevalence_1984.columns.to_numpy()

# Build full edge list
pairs = []
for i, h in enumerate(hosts):
    for j, p in enumerate(parasites):
        pairs.append((i, j, h, p, A[i, j]))

pairs_df = pd.DataFrame(pairs, columns=["i", "j", "host", "parasite", "y"])

# --- BINARY TRANSFORMATION ---
# Convert continuous prevalence into 0 (Absence) or 1 (Presence)
pairs_df["y_bin"] = (pairs_df["y"] > 0).astype(int)
show(pairs_df)
# -------------------------------
# 2. Stratified CV Setup
# -------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_accuracies = []
all_f1s = []
all_aucs = []
all_y_test = []
all_y_pred = []

print(f"Starting 5-Fold Stratified Binary Classification...")
print(f"Target distribution: {pairs_df['y_bin'].value_counts(normalize=True).to_dict()}")

