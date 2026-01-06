import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

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

for fold, (train_idx, test_idx) in enumerate(skf.split(pairs_df, pairs_df["y_bin"])):
    train_pairs = pairs_df.loc[train_idx].copy()
    test_pairs = pairs_df.loc[test_idx].copy()

    # Masking test values to prevent leakage during feature engineering
    A_train = A.copy()
    A_train[test_pairs["i"].values, test_pairs["j"].values] = 0.0

    # 3. Feature Engineering: PCA & Degrees (Network Features)
    k = 5
    pca_h = PCA(n_components=k, random_state=42).fit(A_train)
    Zh = pca_h.transform(A_train)

    pca_p = PCA(n_components=k, random_state=42).fit(A_train.T)
    Zp = pca_p.transform(A_train.T)

    h_deg = (A_train > 0).sum(axis=1)
    p_deg = (A_train > 0).sum(axis=0)


    def featurize_binary(df_subset):
        ii, jj = df_subset["i"].values, df_subset["j"].values
        dot = np.einsum("ij,ij->i", Zh[ii], Zp[jj])  # Interaction Strength
        pca_dist = np.linalg.norm(Zh[ii] - Zp[jj], axis=1)  # Latent Distance

        X = pd.DataFrame({
            "dot_product": dot,
            "pca_distance": pca_dist,
            "host_degree": h_deg[ii],
            "parasite_degree": p_deg[jj],
            "pref_attachment": h_deg[ii] * p_deg[jj],
            "p_last_pc_strength": np.linalg.norm(Zp[jj, -2:], axis=1)
        })
        # Add raw PC components as features
        for d in range(k):
            X[f"h_pc_{d}"], X[f"p_pc_{d}"] = Zh[ii, d], Zp[jj, d]
        return X


    X_train_f = featurize_binary(train_pairs)
    X_test_f = featurize_binary(test_pairs)
    y_train_f, y_test_f = train_pairs["y_bin"].values, test_pairs["y_bin"].values

    # 4. Model Training (Random Forest Classifier)
    # Use class_weight='balanced' to handle the ~2:1 imbalance ratio automatically
    clf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train_f, y_train_f)

    # 5. Prediction & Metrics
    preds = clf.predict(X_test_f)
    probs = clf.predict_proba(X_test_f)[:, 1]  # Probability of Presence

    all_accuracies.append(accuracy_score(y_test_f, preds))
    all_f1s.append(f1_score(y_test_f, preds))
    all_aucs.append(roc_auc_score(y_test_f, probs))

    all_y_test.extend(y_test_f)
    all_y_pred.extend(preds)

    print(f"Fold {fold + 1}: Accuracy = {all_accuracies[-1]:.3f}, AUC = {all_aucs[-1]:.3f}")

# -------------------------------
# 6. Final Summary Metrics
# -------------------------------
print(f"\n--- Final CV Binary Metrics ---")
print(f"Mean Accuracy: {np.mean(all_accuracies):.4f} (+/- {np.std(all_accuracies):.4f})")
print(f"Mean F1-Score: {np.mean(all_f1s):.4f} (+/- {np.std(all_f1s):.4f})")
print(f"Mean ROC-AUC:  {np.mean(all_aucs):.4f} (+/- {np.std(all_aucs):.4f})")

# Confusion Matrix Visualization
cm = confusion_matrix(all_y_test, all_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Absence (0)", "Presence (1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Aggregate Confusion Matrix (5 Folds)")
plt.show()



print('K=5 and t=0.39')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

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

A = prevalence_1984.values
hosts = prevalence_1984.index.to_numpy()
parasites = prevalence_1984.columns.to_numpy()

# Build full edge list
pairs = []
for i, h in enumerate(hosts):
    for j, p in enumerate(parasites):
        pairs.append((i, j, h, p, A[i, j]))

pairs_df = pd.DataFrame(pairs, columns=["i", "j", "host", "parasite", "y"])

# BINARY TRANSFORMATION
pairs_df["y_bin"] = (pairs_df["y"] > 0).astype(int)

# -------------------------------
# 2. Stratified CV Setup
# -------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_y_test = []
all_y_pred = []

# DEFINED OPTIMIZED PARAMETERS
BEST_K = 5
THRESHOLD = 0.39

print(f"Starting 5-Fold Stratified Binary Classification (K={BEST_K}, T={THRESHOLD})...")

for fold, (train_idx, test_idx) in enumerate(skf.split(pairs_df, pairs_df["y_bin"])):
    train_pairs = pairs_df.loc[train_idx].copy()
    test_pairs = pairs_df.loc[test_idx].copy()

    # Masking test values to prevent leakage
    A_train = A.copy()
    A_train[test_pairs["i"].values, test_pairs["j"].values] = 0.0

    # 3. Feature Engineering: PCA & Degrees
    pca_h = PCA(n_components=BEST_K, random_state=42).fit(A_train)
    Zh = pca_h.transform(A_train)

    pca_p = PCA(n_components=BEST_K, random_state=42).fit(A_train.T)
    Zp = pca_p.transform(A_train.T)

    h_deg = (A_train > 0).sum(axis=1)
    p_deg = (A_train > 0).sum(axis=0)


    def featurize_binary(df_subset):
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
        for d in range(BEST_K):
            X[f"h_pc_{d}"], X[f"p_pc_{d}"] = Zh[ii, d], Zp[jj, d]
        return X


    X_train_f = featurize_binary(train_pairs)
    X_test_f = featurize_binary(test_pairs)
    y_train_f, y_test_f = train_pairs["y_bin"].values, test_pairs["y_bin"].values

    # 4. Model Training
    clf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train_f, y_train_f)

    # 5. Prediction with Manual Threshold
    probs = clf.predict_proba(X_test_f)[:, 1]
    fold_preds = (probs >= THRESHOLD).astype(int)

    # Calculate fold metrics
    acc = accuracy_score(y_test_f, fold_preds)
    auc = roc_auc_score(y_test_f, probs)

    all_y_test.extend(y_test_f)
    all_y_pred.extend(fold_preds)

    print(f"Fold {fold + 1}: Accuracy = {acc:.3f}, AUC = {auc:.3f}")

# -------------------------------
# 6. Confusion Matrix Visualization
# -------------------------------

cm = confusion_matrix(all_y_test, all_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Absence (0)", "Presence (1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (K={BEST_K}, T={THRESHOLD})")
plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

# 1. Load your data
path = os.path.expanduser("~/Desktop/DATA.xlsx")
df = pd.read_excel(path)

# Filter for the specific year (1984) and create matrix
df_1984 = df[df["YearCollected"] == 1984].copy()
parasite_cols = df_1984.columns.drop(["YearCollected", "Host"])
prevalence_1984 = (df_1984[parasite_cols] > 0).astype(int).groupby(df_1984["Host"]).mean()
A = prevalence_1984.values

# 2. Perform PCA for all possible components
max_k = min(A.shape)
pca = PCA(n_components=max_k, random_state=42)
pca.fit(A)

# Individual and Cumulative Variance
exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)
k_range = np.arange(1, max_k + 1)

# 3. Plotting the results
plt.figure(figsize=(8, 5))
plt.plot(k_range, cum_var, 'bo-', linewidth=2, label='Cumulative Information')
plt.bar(k_range, exp_var, alpha=0.3, color='gray', label='Individual PC Contribution')

# Highlighting your tested K values
if 5 <= max_k:
    plt.axvline(x=5, color='red', linestyle='--', label=f'K=5 ({cum_var[4]:.1%})')
if 20 <= max_k:
    plt.axvline(x=20, color='green', linestyle='--', label=f'K=20 ({cum_var[19]:.1%})')

plt.xlabel('Number of Components (K)', fontsize=12)
plt.ylabel('Percentage of Data Described (Variance)', fontsize=12)
plt.title('PCA Information Coverage: How much K describes the data?', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print(f"Total information described by K=5:  {cum_var[4]:.1%}")
print(f"Total information described by K=20: {cum_var[19]:.1%}")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Initialize PCA to reduce the matrix to 2 dimensions
# This allows us to draw a 2D map of your nodes
pca = PCA(n_components=2)

# 2. Run PCA on the masked training matrix
# Each row in A_train represents a node and its connections
node_features = pca.fit_transform(A_train)

# 3. Put the results into a DataFrame for easy plotting
pca_df = pd.DataFrame(data=node_features, columns=['PC1', 'PC2'])

# 4. Optional: Add node degrees to color the points (highly connected nodes vs. lonely nodes)
pca_df['degree'] = A_train.sum(axis=1)

# 5. Create the scatter plot
plt.figure(figsize=(10, 7))
plot = sns.scatterplot(
    x='PC1', y='PC2',
    hue='degree',      # Colors the points based on how many connections they have
    palette='viridis',
    data=pca_df,
    alpha=0.6,
    edgecolor=None
)
print('clutring')

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Run PCA on your training matrix
pca = PCA(n_components=2)
node_features = pca.fit_transform(A_train) # A_train is your masked matrix

# 2. Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(node_features[:, 0], node_features[:, 1], alpha=0.7)

# 3. Add labels and grid
plt.title('PCA Node Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, linestyle='--', alpha=0.5)

# 4. Display the plot (Use this in PyCharm!)
plt.show()


print('3d')

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection

# 1. Change PCA to 3 components
pca = PCA(n_components=3)
node_features_3d = pca.fit_transform(A_train) # Use your masked matrix

# 2. Setup the 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3. Create the 3D scatter plot
# We use the three columns from node_features_3d as X, Y, and Z
ax.scatter(
    node_features_3d[:, 0],
    node_fesatures_3d[:, 1],
    node_features_3d[:, 2],
    alpha=0.7,
    edgecolors='w'
)

# 4. Add labels for the 3 axes
ax.set_title('3D PCA Node Clusters')
ax.set_xlabel('PC1 (Width)')
ax.set_ylabel('PC2 (Height)')
ax.set_zlabel('PC3 (Depth)')

# 5. Show the interactive plot
# In PyCharm, this will allow you to click and drag to rotate the graph!
plt.show()