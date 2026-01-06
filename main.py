import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import os

#for PC
#path = "~/Desktop/DATA.xlsx"
#df = pd.read_excel(path)

#for mac
# -------------------------------
# 1. Data Loading & Matrix Prep
# -------------------------------
path = os.path.expanduser("~/Desktop/DATA.xlsx")
df = pd.read_excel(path)

# Filter for the specific year (1984)
df_1984 = df[df["YearCollected"] == 1984].copy()
parasite_cols = df_1984.columns.drop(["YearCollected", "Host"])
presence = (df_1984[parasite_cols] > 0).astype(int)
prevalence_1984 = presence.groupby(df_1984["Host"]).mean()

A = prevalence_1984.values
hosts = prevalence_1984.index.to_numpy()
parasites = prevalence_1984.columns.to_numpy()

# Build full edge list for modeling
pairs = []
for i, h in enumerate(hosts):
    for j, p in enumerate(parasites):
        pairs.append((i, j, h, p, A[i, j]))

pairs_df = pd.DataFrame(pairs, columns=["i", "j", "host", "parasite", "y"])
print(pairs_df.head())


# -------------------------------
# 2. Stratified CV Setup
# -------------------------------
stratify_labels = (pairs_df["y"] > 0).astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_rhos = []
all_r2s = []  # Lisst to store R^2 for each fold
all_y_test = []
all_y_pred = []
fold_importances = []

print("Starting 5-Fold Stratified CV...")

for fold, (train_idx, test_idx) in enumerate(skf.split(pairs_df, stratify_labels)):
    train_pairs = pairs_df.loc[train_idx].copy()
    test_pairs = pairs_df.loc[test_idx].copy()

    # Masking test values to prevent leakage
    A_train = A.copy()
    A_train[test_pairs["i"].values, test_pairs["j"].values] = 0.0

    # 3. Feature Engineering: PCA & Degrees
    k = 5
    pca_h = PCA(n_components=k, random_state=42).fit(A_train)
    Zh = pca_h.transform(A_train)
    pca_p = PCA(n_components=k, random_state=42).fit(A_train.T)
    Zp = pca_p.transform(A_train.T)
    h_deg = (A_train > 0).sum(axis=1)
    p_deg = (A_train > 0).sum(axis=0)


    def featurize_network(df_subset):
        ii, jj = df_subset["i"].values, df_subset["j"].values
        dot = np.einsum("ij,ij->i", Zh[ii], Zp[jj])
        pca_dist = np.linalg.norm(Zh[ii] - Zp[jj], axis=1)

        X = pd.DataFrame({
            "dot_product": dot,
            "pca_distance": pca_dist,
            "host_degree": h_deg[ii],
            "parasite_degree": p_deg[jj],
            "pref_attachment": h_deg[ii] * p_deg[jj],
            "p_last_pc_strength": np.linalg.norm(Zp[jj, -2:], axis=1)
        })
        for d in range(k):
            X[f"h_pc_{d}"], X[f"p_pc_{d}"] = Zh[ii, d], Zp[jj, d]
        return X


    X_train_f = featurize_network(train_pairs)
    X_test_f = featurize_network(test_pairs)
    y_train_f, y_test_f = train_pairs["y"].values, test_pairs["y"].values

    # 4. Model Training (Weighted)
    weights = np.where(y_train_f > 0, 1.87, 1.0)
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features='sqrt', random_state=42)
    rf.fit(X_train_f, y_train_f, sample_weight=weights)

    # 5. Prediction & Fold Metrics
    preds = np.maximum(0, rf.predict(X_test_f))

    # Calculate R^2 and Rho for this specific fold
    fold_rho, _ = spearmanr(y_test_f, preds)
    fold_r2 = r2_score(y_test_f, preds)

    all_rhos.append(fold_rho)
    all_r2s.append(fold_r2)

    all_y_test.extend(y_test_f)
    all_y_pred.extend(preds)
    fold_importances.append(rf.feature_importances_)

    print(f"Fold {fold + 1}: Spearman ρ = {fold_rho:.3f}, R² = {fold_r2:.3f}")

# -------------------------------
# 6. Final Summary
# -------------------------------
print(f"\n--- Final CV Metrics Summary ---")
print(f"Mean Spearman ρ: {np.mean(all_rhos):.4f} (+/- {np.std(all_rhos):.4f})")
print(f"Mean R² Score:   {np.mean(all_r2s):.4f} (+/- {np.std(all_r2s):.4f})")

# Visualization

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
mean_imp = np.mean(fold_importances, axis=0)
indices = np.argsort(mean_imp)
plt.barh(range(len(indices)), mean_imp[indices])
plt.yticks(range(len(indices)), [X_train_f.columns[i] for i in indices])
plt.title("Feature Importance")

plt.subplot(1, 2, 2)
plt.scatter(all_y_test, all_y_pred, alpha=0.5, color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.title(f"Observed vs Predicted (Aggregate R²: {r2_score(all_y_test, all_y_pred):.3f})")
plt.show()
