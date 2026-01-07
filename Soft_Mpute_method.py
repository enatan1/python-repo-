import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
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

binary_1984 = (prevalence_1984 > 0).astype(int)
print(binary_1984)
#show(binary_1984)

#splitting the data using StratifiedShuffleSplit that will split the data by the cells



X = binary_1984.values  # extract the binary interaction matrix as a NumPy array


hosts = binary_1984.index.to_numpy()  # store host species identifiers (row labels) as a NumPy array


parasites = binary_1984.columns.to_numpy() # Store parasite species identifiers (column labels) as a NumPy array


n_hosts, n_parasites = X.shape # Get the dimensions of the interaction matrix


# Build a DataFrame where each row represents one host–parasite cell
# host_idx repeats each host index for every parasite
# parasite_idx tiles parasite indices across hosts
cell_df = pd.DataFrame({
    "host_idx": np.repeat(np.arange(n_hosts), n_parasites),
    "parasite_idx": np.tile(np.arange(n_parasites), n_hosts),
})


# assign a binary label to each host–parasite pair
# label = 1 if interaction observed, 0 if never observed
cell_df["label"] = X[cell_df["host_idx"], cell_df["parasite_idx"]]


# Initialize a stratified shuffle split
# This will preserve the proportion of positive (1) and negative (0) cells
# while splitting the data into 90% train and 10% test
splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.10,   # 10% of cells used for testing
    random_state=42   # fixed seed for reproducibility
)


# Perform the stratified split
# train_idx and test_idx are row indices into cell_df
train_idx, test_idx = next(splitter.split(cell_df, cell_df["label"]))

# Extract training cells (90%) using the stratified indices
train_cells = cell_df.iloc[train_idx].copy()

# Extract test cells (10%) using the stratified indices
test_cells  = cell_df.iloc[test_idx].copy()



# Initialize an empty binary matrix for training data
# All values start as 0 (interaction absent)
X_train = np.zeros_like(X)

# Initialize an empty binary matrix for test data
X_test  = np.zeros_like(X)


# Fill the training matrix with observed interactions
# Only cells selected for training retain their original binary value
X_train[
    train_cells["host_idx"].to_numpy(),
    train_cells["parasite_idx"].to_numpy()
] = train_cells["label"].to_numpy()


# Fill the test matrix with held-out interactions
# These cells are used only for evaluation
X_test[
    test_cells["host_idx"].to_numpy(),
    test_cells["parasite_idx"].to_numpy()
] = test_cells["label"].to_numpy()


