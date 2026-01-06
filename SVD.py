import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from pandasgui import show
from sklearn.model_selection import StratifiedShuffleSplit

# -------------------------------
# 1. Data Loading & Matrix Prep
# -------------------------------
path = os.path.expanduser("~/Desktop/DATA.xlsx") #the path for the file
df = pd.read_excel(path) #reading the file

# Filter for the specific year (1984)
df_1984 = df[df["YearCollected"] == 1984].copy() #filtering the whole data so it will contain only the year 1984
parasite_cols = df_1984.columns.drop(["YearCollected", "Host"]) #removing the columns Host and YearCollected
presence_matrix = (df_1984[parasite_cols] > 0).astype(int) #calculating how much each Host have individuals
prevalence_1984 = presence_matrix.groupby(df_1984["Host"]).mean() # calculating the prevalence for each Host - Parasite pair
#show(prevalence_1984)


X = prevalence_1984.values  # hosts × parasites

U, S, VT = np.linalg.svd(X, full_matrices=False) # calculating SVD components


#plotting the sigma values
plt.figure(figsize=(6,4))
plt.plot(S, marker='o')
plt.yscale("log")
plt.xlabel("Component index")
plt.ylabel("Singular value (σ)")
plt.title("Singular values – Host × Parasite prevalence (1984)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#the plot shows how many SVD components are needed to capture most of the interaction structure.
energy = S**2
cum_energy = np.cumsum(energy) / np.sum(energy)

plt.figure(figsize=(6,4))
plt.plot(cum_energy, marker='o')
plt.axhline(0.9, color='r', linestyle='--', label='90% energy')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained energy")
plt.title("Cumulative energy – SVD (1984)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#split the data to 80/20 by how much the links there is


X = prevalence_1984.values
hosts = prevalence_1984.index.to_numpy() #getting the hosts names
parasites = prevalence_1984.columns.to_numpy() #getting the parasite names

n_hosts, n_parasites = X.shape

# Build list of all cells
cell_df = pd.DataFrame({
    "host_idx": np.repeat(np.arange(n_hosts), n_parasites),
    "parasite_idx": np.tile(np.arange(n_parasites), n_hosts),
})

# Cell values - the prevalence value, and labels - binary 1/0
cell_df["value"] = X[cell_df["host_idx"], cell_df["parasite_idx"]]
cell_df["label"] = (cell_df["value"] > 0).astype(int)

split_data= StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(split_data.split(cell_df, cell_df["label"])) #splitting the data into 80 and 20

train_cells = cell_df.iloc[train_idx].copy()
test_cells  = cell_df.iloc[test_idx].copy()

def check_split(df, name):
    n = len(df)
    n_pos = df["label"].sum()
    print(f"{name}:")
    print(f"  total cells: {n}")
    print(f"  positives:   {n_pos} ({n_pos/n:.3f})")

check_split(cell_df,   "All")
check_split(train_cells, "Train")
check_split(test_cells,  "Test")



print('gr')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ===============================
# 1. Load data and build prevalence matrix
# ===============================

path = os.path.expanduser("~/Desktop/DATA.xlsx")
df = pd.read_excel(path)

df_1984 = df[df["YearCollected"] == 1984].copy()

parasite_cols = df_1984.columns.drop(["YearCollected", "Host"])

presence_matrix = (df_1984[parasite_cols] > 0).astype(int)
prevalence_1984 = presence_matrix.groupby(df_1984["Host"]).mean()

X = prevalence_1984.values
hosts = prevalence_1984.index.to_numpy()
parasites = prevalence_1984.columns.to_numpy()

n_hosts, n_parasites = X.shape

# ===============================
# 2. Convert matrix to cell list
# ===============================

cell_df = pd.DataFrame({
    "host_idx": np.repeat(np.arange(n_hosts), n_parasites),
    "parasite_idx": np.tile(np.arange(n_parasites), n_hosts),
})

cell_df["value"] = X[cell_df["host_idx"], cell_df["parasite_idx"]]
cell_df["label"] = (cell_df["value"] > 0).astype(int)  # binary target

# ===============================
# 3. Stratified cell-wise split (80/20)
# ===============================

splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(cell_df, cell_df["label"]))

train_cells = cell_df.iloc[train_idx].copy()
test_cells  = cell_df.iloc[test_idx].copy()

# Sanity check
def check_split(df, name):
    n = len(df)
    n_pos = df["label"].sum()
    print(f"{name}:")
    print(f"  total cells: {n}")
    print(f"  positives:   {n_pos} ({n_pos/n:.3f})")

check_split(cell_df, "All")
check_split(train_cells, "Train")
check_split(test_cells, "Test")

# ===============================
# 4. Build training matrix for SVD
# ===============================

X_train = np.zeros_like(X)

X_train[
    train_cells["host_idx"].to_numpy(),
    train_cells["parasite_idx"].to_numpy()
] = train_cells["value"].to_numpy()

# ===============================
# 5. SVD on TRAIN ONLY
# ===============================

U, S, Vt = np.linalg.svd(X_train, full_matrices=False)

# choose number of components
k = 5

host_embed = U[:, :k] @ np.diag(S[:k])        # hosts × k
parasite_embed = Vt[:k, :].T @ np.diag(S[:k]) # parasites × k

# ===============================
# 6. Build ML feature matrices
# ===============================

def build_features(cell_df, host_embed, parasite_embed):
    X_feat = np.zeros((len(cell_df), 2 * host_embed.shape[1]))
    y = cell_df["label"].to_numpy()

    for i, row in enumerate(cell_df.itertuples()):
        h = row.host_idx
        p = row.parasite_idx
        X_feat[i] = np.concatenate([host_embed[h], parasite_embed[p]])

    return X_feat, y

X_train_feat, y_train = build_features(train_cells, host_embed, parasite_embed)
X_test_feat,  y_test  = build_features(test_cells,  host_embed, parasite_embed)

# ===============================
# 7. Train binary classifier
# ===============================

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

clf.fit(X_train_feat, y_train)

# ===============================
# 8. Evaluation
# ===============================

y_prob = clf.predict_proba(X_test_feat)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)
baseline = y_test.mean()

print("\nEvaluation:")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC:  {pr_auc:.3f}")
print(f"Baseline prevalence: {baseline:.3f}")

# Optional: confusion matrix at 0.5 threshold
y_pred = (y_prob >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion matrix (threshold = 0.5)")
plt.show()
