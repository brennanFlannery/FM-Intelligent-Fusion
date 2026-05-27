import numpy as np
import pandas as pd
import math
from scipy import stats
from tqdm import tqdm

def pick_best_uncorrelated_features(
    data,
    classes,
    idx_pool,
    num_features=100000000,
    correlation_factor=0.6,
    correlation_metric='spearman'
):
    # -- Input validation -----------------------------------------------
    unique_classes = np.unique(classes)
    if set(unique_classes) != {1, -1}:
        print(
            'Error: class labels must be exactly [1, -1]. provided', unique_classes
        )
        return [], []

    if data.shape[0] != len(classes):
        print('Error: number of samples in data and classes must match')
        return [], []

    # -- Parameter sanitization -----------------------------------------
    correlation_metric = correlation_metric.lower()
    if correlation_metric not in ('spearman', 'pearson'):
        print(
            "Warning: invalid correlation_metric, defaulting to 'spearman'"
        )
        correlation_metric = 'spearman'

    if correlation_factor > 1 or correlation_factor < 0:
        print(
            'Warning: correlation_factor must be between 0 and 1. resetting to 0.6'
        )
        correlation_factor = 0.6

    # -- Prepare data and masks -----------------------------------------
    df = pd.DataFrame(data)
    mask_pos = classes == 1
    mask_neg = classes == -1

    # -- Remove columns with NaNs ---------------------------------------
    non_na_cols = df.columns[df.notna().all(axis=0)]

    # -- Remove low-variance columns ------------------------------------
    threshold = math.floor(0.1 * df.shape[0])
    valid_counts = df[non_na_cols].nunique()
    valid_cols = valid_counts[valid_counts > threshold].index

    # -- Intersect with idx_pool ----------------------------------------
    idx_pool = np.array(idx_pool, dtype=int)
    idx_agree = np.array(
        [i for i in idx_pool if i in valid_cols],
        dtype=int
    )
    if idx_agree.size == 0:
        print('Error: no features remain after quality filtering')
        return [], []

    # -- Compute p-values for discrimination ----------------------------
    p_values = np.array([
        stats.ttest_ind(
            df.loc[mask_pos, j], df.loc[mask_neg, j]
        )[1]
        if correlation_metric == 'pearson'
        else stats.ranksums(
            df.loc[mask_pos, j], df.loc[mask_neg, j]
        )[1]
        for j in idx_agree
    ])
    print("Pvalues calculated")
    # Keep copies for final p-value mapping
    copy_idx = idx_agree.copy()
    copy_pvals = p_values.copy()

    # -- Precompute full correlation matrix once ------------------------
    corr_df = df.iloc[:, idx_agree].corr(method=correlation_metric).abs()
    corr_matrix = corr_df.values

    # -- Iteratively select top features while removing correlated ones ----
    selected = []
    while len(selected) < num_features and idx_agree.size > 0:
        # choose feature with smallest p-value
        best_idx = np.argmin(p_values)
        feature = idx_agree[best_idx]
        selected.append(feature)

        # identify correlated features including itself
        to_remove = np.where(corr_matrix[best_idx, :] > correlation_factor)[0]

        # mask out correlated features for next round
        keep_mask = np.ones(idx_agree.shape[0], dtype=bool)
        keep_mask[to_remove] = False

        idx_agree = idx_agree[keep_mask]
        p_values = p_values[keep_mask]
        corr_matrix = corr_matrix[keep_mask][:, keep_mask]

    # -- Notify if fewer selected than requested ------------------------
    if len(selected) < num_features:
        print(
            f'Too many correlated features. only {len(selected)} returned.'
        )

    # -- Prepare return values ------------------------------------------
    selected = sorted(set(selected))
    pvals_map = dict(zip(copy_idx, copy_pvals))
    selected_pvals = [pvals_map[i] for i in selected]

    return selected, selected_pvals
