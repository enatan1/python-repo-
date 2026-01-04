import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr



path = "~/Desktop/DATA.xlsx"
df = pd.read_excel(path)

df_1984 = df[df["YearCollected"] == 1984].copy()
parasite_cols = df_1984.columns.drop(["YearCollected", "Host"])


presence = (df_1984[parasite_cols] > 0).astype(int)
prevalence_1984 = presence.groupby(df_1984["Host"]).mean()

A = prevalence_1984.values
hosts = prevalence_1984.index.to_numpy()
parasites = prevalence_1984.columns.to_numpy()

nH, nP = A.shape


# -------------------------------
# 2. Build full edge list (labels)
# -------------------------------

pairs = []
for i, h in enumerate(hosts):
    for j, p in enumerate(parasites):
        pairs.append((i, j, h, p, A[i, j]))

pairs_df = pd.DataFrame(
    pairs, columns=["i", "j", "host", "parasite", "y"]
)

# -------------------------------
# 3. Train / test split (edges)
# -------------------------------

train_idx, test_idx = train_test_split(
    pairs_df.index,
    test_size=0.2,
    random_state=42
)

train_pairs = pairs_df.loc[train_idx].copy()
test_pairs  = pairs_df.loc[test_idx].copy()



A_train = A.copy()
A_train[
    test_pairs["i"].to_numpy(),
    test_pairs["j"].to_numpy()
] = 0.0

k = 5

# PCA for hosts (rows)
pca_hosts = PCA(n_components=k, random_state=42)
Zh = pca_hosts.fit_transform(A_train)

# PCA for parasites (columns)
pca_parasites = PCA(n_components=k, random_state=42)
Zp = pca_parasites.fit_transform(A_train.T)



host_degree   = (A_train > 0).sum(axis=1)
host_strength = A_train.sum(axis=1)

parasite_degree   = (A_train > 0).sum(axis=0)
parasite_strength = A_train.sum(axis=0)

host_degree_log   = np.log1p(host_degree)
host_strength_log = np.log1p(host_strength)

parasite_degree_log   = np.log1p(parasite_degree)
parasite_strength_log = np.log1p(parasite_strength)

# -------------------------------
# 6. Feature construction
# -------------------------------

def featurize(pairs_df):
    ii = pairs_df["i"].to_numpy()
    jj = pairs_df["j"].to_numpy()

    X = pd.DataFrame({
        # PCA geometry
        "dot": np.einsum("ij,ij->i", Zh[ii], Zp[jj]),
        "distance": np.linalg.norm(Zh[ii] - Zp[jj], axis=1),

        # marginals
        "host_degree": host_degree[ii],
        "host_strength": host_strength[ii],
        "parasite_degree": parasite_degree[jj],
        "parasite_strength": parasite_strength[jj],

        "host_degree_log": host_degree_log[ii],
        "host_strength_log": host_strength_log[ii],
        "parasite_degree_log": parasite_degree_log[jj],
        "parasite_strength_log": parasite_strength_log[jj],

        # ratios
        "degree_ratio": host_degree[ii] / (parasite_degree[jj] + 1e-6),
        "strength_ratio": host_strength[ii] / (parasite_strength[jj] + 1e-6),
    })

    for d in range(k):
        X[f"h_pc_{d}"] = Zh[ii, d]
        X[f"p_pc_{d}"] = Zp[jj, d]

    return X

# -------------------------------
# 7. Train / test matrices
# -------------------------------

X_train = featurize(train_pairs)
y_train = train_pairs["y"].to_numpy()

X_test  = featurize(test_pairs)
y_test  = test_pairs["y"].to_numpy()

# -------------------------------
# 8. Random Forest
# -------------------------------

rf = RandomForestRegressor(
    n_estimators=400,
    min_samples_leaf=10,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# -------------------------------
# 9. Metrics
# -------------------------------

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
rho, _ = spearmanr(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.3f}")
print(f"Spearman ρ: {rho:.3f}")



plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0,1],[0,1],'k--')
plt.xlabel("Observed prevalence")
plt.ylabel("Predicted prevalence")
plt.title("Observed vs Predicted (PCA-based RF)")
plt.tight_layout()
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(5,4))
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("Observed prevalence")
plt.ylabel("Residual")
plt.title("Residuals vs Observed")
plt.tight_layout()
plt.show()

print('checking importance')

import shap

X_shap = X_test.sample(n=min(1000, len(X_test)), random_state=42)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_shap)

shap.summary_plot(
    shap_values,
    X_shap,
    plot_type="dot",
    show=True
)


print('phylo + CK and StratifiedKFold')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from Bio import Phylo

# -------------------------------
# 1. Phylogenetic Tree Processing
# -------------------------------
tree_path = os.path.expanduser("~/Desktop/hosts_timetree.nwk")


def clean_n(name):
    if name is None: return ""
    return str(name).strip().replace(" ", "_").replace("'", "").replace('"', "")


tree = Phylo.read(tree_path, "newick")
terminals = [clade.name for clade in tree.get_terminals() if clade.name]
hosts_in_tree = {clean_n(name): name for name in terminals}

dist_lookup = {}
for clean_h1, raw_h1 in hosts_in_tree.items():
    dist_lookup[clean_h1] = {}
    for clean_h2, raw_h2 in hosts_in_tree.items():
        dist_lookup[clean_h1][clean_h2] = tree.distance(raw_h1, raw_h2)

# -------------------------------
# 2. Data Loading & Matrix Prep
# -------------------------------
path = os.path.expanduser("~/Desktop/DATA.xlsx")
df = pd.read_excel(path)

df_1984 = df[df["YearCollected"] == 1984].copy()
parasite_cols = df_1984.columns.drop(["YearCollected", "Host"])
presence = (df_1984[parasite_cols] > 0).astype(int)
prevalence_1984 = presence.groupby(df_1984["Host"]).mean()

A = prevalence_1984.values
hosts = prevalence_1984.index.to_numpy()
parasites = prevalence_1984.columns.to_numpy()

pairs = []
for i, h in enumerate(hosts):
    for j, p in enumerate(parasites):
        pairs.append((i, j, h, p, A[i, j]))

pairs_df = pd.DataFrame(pairs, columns=["i", "j", "host", "parasite", "y"])

# -------------------------------
# 3. Stratified K-Fold CV Setup
# -------------------------------
# Create binary labels for stratification (Presence vs Absence)
stratify_labels = (pairs_df["y"] > 0).astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_rhos = []
cv_r2s = []
all_y_test = []
all_y_pred = []

print("Starting 5-Fold Stratified Cross-Validation...")

for fold, (train_idx, test_idx) in enumerate(skf.split(pairs_df, stratify_labels)):
    train_pairs = pairs_df.loc[train_idx].copy()
    test_pairs = pairs_df.loc[test_idx].copy()

    # Masking: Create training matrix by zeroing out test edges for this fold
    A_train = A.copy()
    A_train[test_pairs["i"].values, test_pairs["j"].values] = 0.0

    # 4. Feature Engineering (Inside Fold to prevent Leakage)
    k = 5
    pca_h = PCA(n_components=k, random_state=42).fit(A_train)
    Zh = pca_h.transform(A_train)
    pca_p = PCA(n_components=k, random_state=42).fit(A_train.T)
    Zp = pca_p.transform(A_train.T)

    h_deg = (A_train > 0).sum(axis=1)
    p_deg = (A_train > 0).sum(axis=0)

    # Pre-calculate carrier lists for this fold's training data
    carrier_map = {}
    for j_idx in range(len(parasites)):
        carrying_indices = np.where(A_train[:, j_idx] > 0)[0]
        carrier_map[j_idx] = [clean_n(hosts[idx]) for idx in carrying_indices]


    def featurize_cv(df_subset):
        ii, jj = df_subset["i"].values, df_subset["j"].values
        h_names = df_subset["host"].values

        phylo_dists = []
        for h_name, j_idx in zip(h_names, jj):
            h_c = clean_n(h_name)
            carriers = carrier_map[j_idx]
            if h_c not in dist_lookup:
                phylo_dists.append(200.0)
            elif not carriers:
                phylo_dists.append(100.0)
            else:
                d = [dist_lookup[h_c].get(c, 150.0) for c in carriers if c in dist_lookup]
                phylo_dists.append(np.mean(d) if d else 150.0)

        X = pd.DataFrame({
            "dot_product": np.einsum("ij,ij->i", Zh[ii], Zp[jj]),
            "pca_distance": np.linalg.norm(Zh[ii] - Zp[jj], axis=1),
            "phylo_dist_carriers": phylo_dists,
            "host_degree": h_deg[ii],
            "parasite_degree": p_deg[jj],
            "pref_attachment": h_deg[ii] * p_deg[jj]
        })
        for d in range(k):
            X[f"h_pc_{d}"], X[f"p_pc_{d}"] = Zh[ii, d], Zp[jj, d]
        return X


    X_train_fold = featurize_cv(train_pairs)
    y_train_fold = train_pairs["y"].values
    X_test_fold = featurize_cv(test_pairs)
    y_test_fold = test_pairs["y"].values

    # 5. Train & Predict
    weights = np.where(y_train_fold > 0, 40, 1.0)
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(X_train_fold, y_train_fold, sample_weight=weights)

    preds = np.maximum(0, rf.predict(X_test_fold))

    # 6. Store Fold Metrics
    rho, _ = spearmanr(y_test_fold, preds)
    r2 = r2_score(y_test_fold, preds)
    cv_rhos.append(rho)
    cv_r2s.append(r2)
    all_y_test.extend(y_test_fold)
    all_y_pred.extend(preds)

    print(f"Fold {fold + 1}: Spearman ρ = {rho:.3f}, R² = {r2:.3f}")

# -------------------------------
# 7. Final Results & Visualization
# -------------------------------
print(f"\n--- Final CV Metrics ---")
print(f"Mean Spearman ρ: {np.mean(cv_rhos):.3f} (+/- {np.std(cv_rhos):.3f})")
print(f"Mean Global R²:  {np.mean(cv_r2s):.3f} (+/- {np.std(cv_r2s):.3f})")

plt.figure(figsize=(6, 5))
plt.scatter(all_y_test, all_y_pred, alpha=0.4, color='red')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.title("Combined CV: Observed vs Predicted")
plt.show()


print('checking second model')

import pandas as pd
import numpy as np
import os

# Set path to your Excel file
path = os.path.expanduser("~/Desktop/DATA.xlsx")


def check_prevalence_stats():
    # 1. Load Data
    df = pd.read_excel(path)

    # 2. Filter for 1984
    df_1984 = df[df["YearCollected"] == 1984].copy()

    # 3. Identify Parasite Columns
    parasite_cols = df_1984.columns.drop(["YearCollected", "Host"])

    # 4. Calculate Prevalence Matrix
    # (Grouping by Host to get mean infection rates)
    presence = (df_1984[parasite_cols] > 0).astype(int)
    prevalence_matrix = presence.groupby(df_1984["Host"]).mean()

    # 5. Extract Values and Count
    A = prevalence_matrix.values
    total_pairs = A.size
    positive_pairs = np.count_nonzero(A > 0)
    zero_pairs = total_pairs - positive_pairs

    # 6. Print Summary
    print("-" * 30)
    print(f"STAGE 1 (Classifier) TOTAL DATA: {total_pairs} pairs")
    print(f"STAGE 2 (Regressor) POSITIVE DATA: {positive_pairs} pairs")
    print("-" * 30)
    print(f"Zero-to-Positive Ratio: {zero_pairs / positive_pairs:.2f} : 1")
    print(f"Positive Density: {(positive_pairs / total_pairs) * 100:.2f}%")

    if positive_pairs < 100:
        print("\nWARNING: Very low positive sample size.")
        print("The Hurdle regressor may struggle to find patterns.")
    else:
        print("\nSUCCESS: Sufficient data found for a Hurdle approach.")


check_prevalence_stats()



