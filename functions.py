import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

def mask_test_cells(binary_df, X_test, random_state=42):
    """
    Mask test interactions in a binary host–parasite matrix.

    - Masks all test positives (1s in X_test)
    - Masks the same *proportion* of zeros as the proportion of masked 1s
    - Prints counts and proportions of masked 1s and 0s

    Parameters
    ----------
    binary_df : pd.DataFrame
        Binary interaction matrix (hosts × parasites).
    X_test : np.ndarray
        Binary matrix with test interactions marked as 1.
        Must have the same shape as binary_df.
    random_state : int
        Seed for reproducible random masking.

    Returns
    -------
    masked_df : pd.DataFrame
        Copy of binary_df with selected cells masked as NA.
    """

    rng = np.random.default_rng(random_state)

    # Copy to avoid modifying original data
    masked_df = binary_df.copy()
    A = binary_df.values

    # ---- Identify positives and zeros ----
    pos_idx = np.argwhere(A == 1)
    zero_idx = np.argwhere(A == 0)

    n_pos_total = len(pos_idx)
    n_zero_total = len(zero_idx)

    # ---- Identify test positives ----
    test_pos_idx = np.argwhere(X_test == 1)
    n_test_pos = len(test_pos_idx)

    if n_test_pos == 0:
        raise ValueError("No test positives found in X_test.")

    # ---- Compute masking proportion ----
    prop_masked = n_test_pos / n_pos_total
    n_zero_mask = int(np.floor(prop_masked * n_zero_total))

    # ---- Mask test positives ----
    for i, j in test_pos_idx:
        masked_df.iat[i, j] = pd.NA

    # ---- Randomly mask zeros according to proportion ----
    zero_mask_idx = zero_idx[
        rng.choice(len(zero_idx), size=n_zero_mask, replace=False)
    ]

    for i, j in zero_mask_idx:
        masked_df.iat[i, j] = pd.NA

    # ---- Reporting ----
    print("Masking summary:")
    print(f"  Total positives: {n_pos_total}")
    print(f"  Masked positives: {n_test_pos} ({prop_masked:.3f})")
    print(f"  Total zeros: {n_zero_total}")
    print(f"  Masked zeros: {n_zero_mask} ({n_zero_mask / n_zero_total:.3f})")

    return masked_df