import os
import argparse
import random
import itertools
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from scipy.linalg import orthogonal_procrustes
from typing import Tuple, List, Dict
from tqdm import tqdm


def load_matched_embeddings(
    model1_dir: str,
    model2_dir: str,
    sample_size: int = 50000,
    seed: int = 0,
    feature_key: str = "features",
    coord_key: str = "coords"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load matching tile embeddings from two model directories, aligning by coords.
    Randomly sample up to `sample_size` tile pairs across all common slides.

    Returns:
        X (np.ndarray): (N, d1) array of embeddings from model1
        Y (np.ndarray): (N, d2) array of embeddings from model2
    """
    random.seed(seed)
    np.random.seed(seed)

    files1 = {fname: os.path.join(model1_dir, fname)
              for fname in os.listdir(model1_dir) if fname.endswith(".h5")}
    files2 = {fname: os.path.join(model2_dir, fname)
              for fname in os.listdir(model2_dir) if fname.endswith(".h5")}

    common_slides = sorted(set(files1.keys()) & set(files2.keys()))
    random.shuffle(common_slides)

    collected_X: List[np.ndarray] = []
    collected_Y: List[np.ndarray] = []
    total_collected = 0

    with tqdm(total=sample_size, desc="Loading matching tiles", unit="tiles") as pbar:
        for slide in common_slides:
            with h5py.File(files1[slide], 'r') as f1, h5py.File(files2[slide], 'r') as f2:
                coords1 = f1[coord_key][:]
                feats1 = f1[feature_key][:]
                coords2 = f2[coord_key][:]
                feats2 = f2[feature_key][:]

            struct_dtype = np.dtype([('x', coords1.dtype), ('y', coords1.dtype)])
            c1 = coords1.view(struct_dtype).reshape(-1)
            c2 = coords2.view(struct_dtype).reshape(-1)

            common_coords, idx1, idx2 = np.intersect1d(c1, c2, return_indices=True)
            if len(common_coords) == 0:
                continue

            matched_feats1 = feats1[idx1]
            matched_feats2 = feats2[idx2]

            remaining = sample_size - total_collected
            if len(common_coords) > remaining:
                perm = np.random.permutation(len(common_coords))[:remaining]
                matched_feats1 = matched_feats1[perm]
                matched_feats2 = matched_feats2[perm]
                collected_X.append(matched_feats1)
                collected_Y.append(matched_feats2)
                total_collected += remaining
                pbar.update(remaining)
                break
            else:
                collected_X.append(matched_feats1)
                collected_Y.append(matched_feats2)
                total_collected += len(common_coords)
                pbar.update(len(common_coords))
                if total_collected >= sample_size:
                    break

    if total_collected == 0:
        raise ValueError(f"No matching tiles between {model1_dir} and {model2_dir}.")

    X = np.vstack(collected_X)[:sample_size]
    Y = np.vstack(collected_Y)[:sample_size]
    return X, Y


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cross_cov = Xc.T @ Yc
    hsic_xy = np.linalg.norm(cross_cov, ord='fro') ** 2
    cov_xx = Xc.T @ Xc
    cov_yy = Yc.T @ Yc
    hsic_xx = np.linalg.norm(cov_xx, ord='fro')
    hsic_yy = np.linalg.norm(cov_yy, ord='fro')
    return float(hsic_xy / (hsic_xx * hsic_yy + 1e-12))


def compute_svcca(
    X: np.ndarray,
    Y: np.ndarray,
    var_thres: float = 0.99,
    max_components: int = 50
) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Ux, Sx, _ = np.linalg.svd(Xc, full_matrices=False)
    var_ratios_x = (Sx**2) / np.sum(Sx**2)
    cum_x = np.cumsum(var_ratios_x)
    kx = min(int(np.searchsorted(cum_x, var_thres) + 1), max_components)
    Uy, Sy, _ = np.linalg.svd(Yc, full_matrices=False)
    var_ratios_y = (Sy**2) / np.sum(Sy**2)
    cum_y = np.cumsum(var_ratios_y)
    ky = min(int(np.searchsorted(cum_y, var_thres) + 1), max_components)
    Xr = Ux[:, :kx]
    Yr = Uy[:, :ky]
    n_cca = min(kx, ky)
    cca = CCA(n_components=n_cca)
    cca.fit(Xr, Yr)
    Xs, Ys = cca.transform(Xr, Yr)
    corrs = np.array([np.corrcoef(Xs[:, i], Ys[:, i])[0, 1] for i in range(n_cca)])
    return float(corrs.mean())


def compute_procrustes(
    X: np.ndarray,
    Y: np.ndarray,
    pca_dim: int = 50
) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    pca_x = PCA(n_components=min(pca_dim, Xc.shape[1]))
    pca_y = PCA(n_components=min(pca_dim, Yc.shape[1]))
    Xp = pca_x.fit_transform(Xc)
    Yp = pca_y.fit_transform(Yc)
    d_min = min(Xp.shape[1], Yp.shape[1])
    Xp = Xp[:, :d_min]
    Yp = Yp[:, :d_min]
    R, _ = orthogonal_procrustes(Xp, Yp)
    Yp_aligned = Xp @ R
    dist = np.linalg.norm(Yp_aligned - Yp, ord='fro') / (np.linalg.norm(Yp, ord='fro') + 1e-12)
    return float(dist)


def compute_knn_overlap(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 10
) -> Tuple[float, np.ndarray]:
    nbrs_x = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    nbrs_y = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(Y)
    _, indices_x = nbrs_x.kneighbors(X)
    _, indices_y = nbrs_y.kneighbors(Y)
    jaccard_scores = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        set_x = set(indices_x[i, 1:])
        set_y = set(indices_y[i, 1:])
        inter = len(set_x & set_y)
        union = len(set_x | set_y)
        jaccard_scores[i] = inter / (union + 1e-12)
    return float(np.mean(jaccard_scores)), jaccard_scores


def compute_ridge_r2(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 1.0
) -> Tuple[float, float]:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    ridge_xy = Ridge(alpha=alpha, fit_intercept=False)
    ridge_xy.fit(Yc, Xc)
    X_pred = ridge_xy.predict(Yc)
    ss_res_xy = np.sum((Xc - X_pred) ** 2)
    ss_tot_x = np.sum(Xc ** 2)
    r2_y2x = 1 - (ss_res_xy / (ss_tot_x + 1e-12))
    ridge_yx = Ridge(alpha=alpha, fit_intercept=False)
    ridge_yx.fit(Xc, Yc)
    Y_pred = ridge_yx.predict(Xc)
    ss_res_yx = np.sum((Yc - Y_pred) ** 2)
    ss_tot_y = np.sum(Yc ** 2)
    r2_x2y = 1 - (ss_res_yx / (ss_tot_y + 1e-12))
    return float(r2_y2x), float(r2_x2y)


def plot_radar(metrics: dict, labels: List[str], output_path: str):
    """Create and save a radar plot for the provided metric values."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
    values = [metrics[l] for l in labels]
    if 'procrustes' in labels:
        idx = labels.index('procrustes')
        values[idx] = 1.0 / (1.0 + values[idx])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=20)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_title("Radar Plot: " + os.path.splitext(os.path.basename(output_path))[0], fontsize=24, fontweight='bold')
    ax.spines['polar'].set_visible(False)
    fig.savefig(output_path, dpi=500)
    plt.close(fig)


def plot_embedding_scatter(X: np.ndarray, Y: np.ndarray, dir1: str, dir2: str, output_path: str, sample=5000):
    """Project embeddings to 2D with separate PCA and scatter, labeling by model names."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
    n = min(sample, X.shape[0], Y.shape[0])
    idx_X = np.random.choice(X.shape[0], size=n, replace=False)
    idx_Y = np.random.choice(Y.shape[0], size=n, replace=False)
    X_sample = X[idx_X]
    Y_sample = Y[idx_Y]
    pca_X = PCA(n_components=2)
    proj_X = pca_X.fit_transform(X_sample)
    pca_Y = PCA(n_components=2)
    proj_Y = pca_Y.fit_transform(Y_sample)
    name1 = os.path.basename(os.path.normpath(dir1))
    name2 = os.path.basename(os.path.normpath(dir2))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(proj_X[:, 0], proj_X[:, 1], s=5, alpha=0.6, label=name1)
    ax.scatter(proj_Y[:, 0], proj_Y[:, 1], s=5, alpha=0.6, label=name2)
    ax.set_title(f"PCA Scatter: {name1} vs {name2}", fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(output_path, dpi=500)
    plt.close(fig)


def plot_knn_histogram(jaccard_scores: np.ndarray, output_path: str):
    """Plot and save a histogram of tile-wise Jaccard scores with customized style."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
    fig, ax = plt.subplots()
    ax.hist(jaccard_scores, bins=30, color='dodgerblue', edgecolor='black')
    ax.set_xlabel('Jaccard Index (k-NN overlap)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=24, fontweight='bold')
    ax.set_title('Histogram of k-NN Jaccard Scores', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(output_path, dpi=500)
    plt.close(fig)


def plot_neighborhoods(X: np.ndarray, Y: np.ndarray, jaccard_scores: np.ndarray, output_path: str, k=10, probes=3):
    """Plot fine-grained neighborhoods for a few random probe tiles, separately for each model."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
    rng = np.random.RandomState(0)
    indices = rng.choice(X.shape[0], size=probes, replace=False)
    fig, axs = plt.subplots(probes, 2, figsize=(8, 4 * probes))
    for i, probe_idx in enumerate(indices):
        # Model A (X) neighborhood
        nbrs_x = NearestNeighbors(n_neighbors=k + 1).fit(X)
        dist_x, idx_x = nbrs_x.kneighbors(X[probe_idx].reshape(1, -1))
        pts_idx_x = np.concatenate(([probe_idx], idx_x[0][1:]))
        pca_x = PCA(n_components=2)
        proj_x = pca_x.fit_transform(X[pts_idx_x])
        ax_x = axs[i, 0] if probes > 1 else axs[0]
        ax_x.scatter(proj_x[0, 0], proj_x[0, 1], color='black', s=50, label='Probe tile')
        ax_x.scatter(proj_x[1:, 0], proj_x[1:, 1], color='blue', s=20, label='A neighbors')
        ax_x.set_title(f'Model A Neighborhood of tile {probe_idx}', fontsize=24, fontweight='bold')
        ax_x.tick_params(axis='both', labelsize=20)
        ax_x.spines['top'].set_visible(False)
        ax_x.spines['right'].set_visible(False)
        # Model B (Y) neighborhood
        nbrs_y = NearestNeighbors(n_neighbors=k + 1).fit(Y)
        dist_y, idx_y = nbrs_y.kneighbors(Y[probe_idx].reshape(1, -1))
        pts_idx_y = np.concatenate(([probe_idx], idx_y[0][1:]))
        pca_y = PCA(n_components=2)
        proj_y = pca_y.fit_transform(Y[pts_idx_y])
        ax_y = axs[i, 1] if probes > 1 else axs[1]
        ax_y.scatter(proj_y[0, 0], proj_y[0, 1], color='black', s=50, label='Probe tile')
        ax_y.scatter(proj_y[1:, 0], proj_y[1:, 1], color='red', s=20, label='B neighbors')
        ax_y.set_title(f'Model B Neighborhood of tile {probe_idx}', fontsize=24, fontweight='bold')
        ax_y.tick_params(axis='both', labelsize=20)
        ax_y.spines['top'].set_visible(False)
        ax_y.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=500)
    plt.close(fig)


def plot_radial_average_metrics(results_df: pd.DataFrame, output_path: str):
    """Create radial plot of average metrics per model (excluding procrustes)."""
    # Model color mapping
    model_colors = {
        'features_conch_v15': '#0070C0',
        'features_musk': '#FFC000',
        'features_hoptimus1': '#C00000',
        'features_virchow2': '#7030A0',
        'features_gigapath': '#00B050'
    }
    
    # Get unique model names
    all_models = set(results_df['model1'].unique()) | set(results_df['model2'].unique())
    
    # Compute average metrics for each model across all comparisons
    metrics = ['cka', 'svcca', 'knn', 'ridge_avg']
    metric_labels = ['CKA', 'SVCCA', 'k-NN', 'Ridge R²']
    
    model_avg_metrics = {}
    for model in all_models:
        model_metrics = []
        for metric in ['cka', 'svcca', 'knn']:
            # Get all comparisons involving this model
            mask = (results_df['model1'] == model) | (results_df['model2'] == model)
            values = results_df[mask][metric].values
            model_metrics.append(np.nanmean(values))
        
        # For ridge, average r2_y2x and r2_x2y
        mask = (results_df['model1'] == model) | (results_df['model2'] == model)
        r2_values = []
        if 'r2_y2x' in results_df.columns:
            r2_values.extend(results_df[mask]['r2_y2x'].values)
        if 'r2_x2y' in results_df.columns:
            r2_values.extend(results_df[mask]['r2_x2y'].values)
        model_metrics.append(np.nanmean(r2_values) if r2_values else np.nan)
        
        model_avg_metrics[model] = model_metrics
    
    # Create radial plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for model, values in model_avg_metrics.items():
        values_plot = values + values[:1]  # Complete the circle
        color = model_colors.get(model, '#CCCCCC')
        ax.plot(angles, values_plot, 'o-', linewidth=3, color=color, markersize=8)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=20, fontweight='normal')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=20)
    ax.set_title('Average Metrics Across All Comparisons', fontsize=24, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_procrustes_heatmap(results_df: pd.DataFrame, output_path: str):
    """Create lower-triangle heatmap of procrustes distances between models."""
    # Model color mapping for reference (not used directly in heatmap but for consistency)
    model_colors = {
        'features_conch_v15': '#0070C0',
        'features_musk': '#FFC000',
        'features_hoptimus1': '#C00000',
        'features_virchow2': '#7030A0',
        'features_gigapath': '#00B050'
    }
    
    # Get unique models
    all_models = sorted(set(results_df['model1'].unique()) | set(results_df['model2'].unique()))
    
    # Create symmetric matrix of procrustes distances
    n_models = len(all_models)
    procrustes_matrix = np.full((n_models, n_models), np.nan)
    
    for idx, row in results_df.iterrows():
        if 'procrustes' in row and not np.isnan(row['procrustes']):
            i = all_models.index(row['model1'])
            j = all_models.index(row['model2'])
            procrustes_matrix[i, j] = row['procrustes']
            procrustes_matrix[j, i] = row['procrustes']
    
    # Set diagonal to 0 (distance to self)
    np.fill_diagonal(procrustes_matrix, 0)
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(procrustes_matrix, dtype=bool), k=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
    
    # Create heatmap with lower triangle only
    im = ax.imshow(np.ma.array(procrustes_matrix, mask=mask), 
                   cmap='cool', aspect='equal', vmin=0, vmax=np.nanmax(procrustes_matrix))
    
    # Remove tick labels and ticks - we'll use colored boxes instead
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    
    # Add colored boxes for each model on both axes
    box_size = 0.8  # Size of the colored boxes (same as heatmap cells)
    
    # X-axis colored boxes (bottom)
    for i, model in enumerate(all_models):
        color = model_colors.get(model, '#CCCCCC')
        # Position boxes below the heatmap
        rect = Rectangle((i - box_size/2, -1 - box_size/2), box_size, box_size, 
                        linewidth=1, edgecolor='black', facecolor=color, 
                        clip_on=False, transform=ax.transData)
        ax.add_patch(rect)
    
    # Y-axis colored boxes (left)
    for i, model in enumerate(all_models):
        color = model_colors.get(model, '#CCCCCC')
        # Position boxes to the left of the heatmap
        rect = Rectangle((-1 - box_size/2, i - box_size/2), box_size, box_size, 
                        linewidth=1, edgecolor='black', facecolor=color, 
                        clip_on=False, transform=ax.transData)
        ax.add_patch(rect)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Procrustes Distance', fontsize=24, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def load_random_embeddings(
    model_dir: str,
    sample_size: int = 5000,
    seed: int = 0,
    feature_key: str = "features"
) -> np.ndarray:
    """
    Randomly sample up to `sample_size` embeddings from a single model directory (across all slides).
    Returns array of shape (N, D).
    """
    random.seed(seed)
    np.random.seed(seed)
    files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".h5")]
    random.shuffle(files)
    collected: List[np.ndarray] = []
    total = 0
    for fp in tqdm(files, desc=f"Sampling tiles from {os.path.basename(model_dir)}", leave=False):
        with h5py.File(fp, 'r') as f:
            feats = f[feature_key][:]
        n_feats = feats.shape[0]
        remaining = sample_size - total
        if n_feats > remaining:
            idx = np.random.choice(n_feats, size=remaining, replace=False)
            collected.append(feats[idx])
            total += remaining
            break
        else:
            collected.append(feats)
            total += n_feats
            if total >= sample_size:
                break
    if total == 0:
        raise ValueError(f"No embeddings found in {model_dir} to sample.")
    X = np.vstack(collected)[:sample_size]
    return X


def plot_all_models_scatter(model_dirs: List[str], output_path: str, sample_per_model: int, pca_dim: int):
    """Create a single 2D scatter for all models combined."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
    samples = []
    labels = []
    print("\nGenerating all-models scatter plot...")
    for idx, md in enumerate(tqdm(model_dirs, desc="Loading models for scatter", leave=False)):
        X = load_random_embeddings(md, sample_size=sample_per_model, seed=idx)
        # Reduce to common dimension
        pca = PCA(n_components=min(pca_dim, X.shape[1]))
        Xr = pca.fit_transform(X)
        samples.append(Xr)
        labels += [os.path.basename(os.path.normpath(md))] * Xr.shape[0]
    combined = np.vstack(samples)
    # Final PCA to 2D
    pca2 = PCA(n_components=2)
    proj = pca2.fit_transform(combined)
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    start = 0
    colors = plt.cm.tab10.colors
    for i, md in enumerate(model_dirs):
        name = os.path.basename(os.path.normpath(md))
        count = min(sample_per_model, load_random_embeddings(md, sample_per_model).shape[0])
        pts = proj[start:start+count]
        ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.6, label=name, color=colors[i % len(colors)])
        start += count
    ax.set_title("PCA Scatter: All Models", fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(output_path, dpi=500)
    plt.close(fig)

# === Consensus Clustering Functions ===
from sklearn.cluster import KMeans

def cluster_once(X: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    """Perform a single k-means clustering run on data X."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    return km.fit_predict(X)


def compute_consensus_matrix(
    X: np.ndarray,
    n_clusters: int,
    subsample_frac: float = 0.8,
    n_runs: int = 100,
    random_state: int = 0
) -> np.ndarray:
    """
    Build an (N,N) consensus matrix by running k-means on random subsamples of X.
    M[i,j] is the fraction of runs where i and j co-clustered.
    """
    N = X.shape[0]
    counts = np.zeros((N, N), dtype=int)
    consensus = np.zeros((N, N), dtype=int)
    rng = np.random.RandomState(random_state)
    for run in range(n_runs):
        idx = rng.choice(N, size=int(subsample_frac * N), replace=False)
        labels = cluster_once(X[idx], n_clusters=n_clusters, random_state=random_state + run)
        for pi, i in enumerate(idx):
            for pj, j in enumerate(idx):
                counts[i, j] += 1
                if labels[pi] == labels[pj]:
                    consensus[i, j] += 1
    M = np.zeros((N, N), dtype=float)
    nonzero = counts > 0
    M[nonzero] = consensus[nonzero] / counts[nonzero]
    return M


def summarize_model_agreement(
    consensus: np.ndarray,
    model_labels: List[str]
) -> Tuple[float, float]:
    """
    Compute mean within-model and between-model consensus.
    """
    N = len(model_labels)
    within = []
    between = []
    for i in range(N):
        for j in range(i):
            if model_labels[i] == model_labels[j]:
                within.append(consensus[i, j])
            else:
                between.append(consensus[i, j])
    return float(np.mean(within)), float(np.mean(between))
 
# def run_consensus_clustering(
    # model_dirs: List[str],
    # sample_per_model: int,
    # pca_dim: int,
    # cluster_counts: List[int],
    # subsample_frac: float,
    # n_runs: int,
    # random_state: int,
    # out_dir: str
# ) -> dict:
    # """
    # For each K in cluster_counts, sample embeddings, reduce to common PCA space,
    # compute consensus clustering, and return both overall and per-model consensus scores.

    # Returns:
        # results: dict mapping K -> {
            # 'overall': (within_mean, between_mean),
            # 'per_model': {model_name: within_model_mean, ...}
        # }
    # """
    # # Load and reduce embeddings for each model
    # samples = []
    # labels = []
    # for idx, md in enumerate(model_dirs):
        # X_md = load_random_embeddings(md, sample_size=sample_per_model, seed=random_state + idx)
        # pca = PCA(n_components=min(pca_dim, X_md.shape[1]))
        # samples.append(pca.fit_transform(X_md))
        # labels += [os.path.basename(os.path.normpath(md))] * samples[-1].shape[0]
    # X_all = np.vstack(samples)

    # results = {}
    # for K in cluster_counts:
        # consensus = compute_consensus_matrix(
            # X_all,
            # n_clusters=K,
            # subsample_frac=subsample_frac,
            # n_runs=n_runs,
            # random_state=random_state
        # )
        # # overall within/between
        # w_all, b_all = summarize_model_agreement(consensus, labels)
        # # per-model within
        # per_model = {}
        # unique_models = sorted(set(labels))
        # label_array = np.array(labels)
        # N = len(labels)
        # for m in unique_models:
            # idxs = np.where(label_array == m)[0]
            # # collect consensus entries for i<j both in idxs
            # vals = []
            # for i in idxs:
                # for j in idxs:
                    # if j < i:
                        # vals.append(consensus[i, j])
            # per_model[m] = float(np.mean(vals)) if vals else np.nan
        # results[K] = {
            # 'overall': (float(w_all), float(b_all)),
            # 'per_model': per_model
        # }
    # return results

def run_consensus_clustering(
    model_dirs: List[str],
    sample_per_model: int,
    pca_dim: int,
    cluster_counts: List[int],
    subsample_frac: float,
    n_runs: int,
    random_state: int,
    out_dir: str
) -> Dict[int, Dict]:
    """
    For each K in cluster_counts, sample embeddings by model (slide-level),
    reduce to common PCA space, compute consensus clustering, and return:
      - overall: (within_all, between_all)
      - per_model: {model_name: within_model}
      - per_pair: { "modelA_vs_modelB": between_pair }
    """
    # 1) Load & PCA‐reduce all slides per model
    samples, labels = [], []
    for md in model_dirs:
        X_md = load_random_embeddings(md)
        pca = PCA(n_components=min(pca_dim, X_md.shape[1]))
        Xr = pca.fit_transform(X_md)
        samples.append(Xr)
        labels += [os.path.basename(md)] * Xr.shape[0]
    X_all = np.vstack(samples)

    # 2) Precompute indices
    labels = np.array(labels)
    models = sorted(set(labels))
    idx_by_model = {m: np.where(labels == m)[0] for m in models}
    # all unordered model‐pairs
    model_pairs = [(models[i], models[j])
                   for i in range(len(models)) for j in range(i+1, len(models))]

    results = {}
    for K in cluster_counts:
        M = compute_consensus_matrix(
            X_all,
            n_clusters=K,
            subsample_frac=subsample_frac,
            n_runs=n_runs,
            random_state=random_state
        )
        # overall within vs between
        w_all, b_all = summarize_model_agreement(M, labels.tolist())

        # per‐model within
        per_model = {}
        for m, idxs in idx_by_model.items():
            vals = [M[i,j] for i in idxs for j in idxs if j < i]
            per_model[m] = float(np.mean(vals)) if vals else np.nan

        # per‐pair between
        per_pair = {}
        for mA, mB in model_pairs:
            idxsA, idxsB = idx_by_model[mA], idx_by_model[mB]
            vals = [M[i,j] for i in idxsA for j in idxsB]
            key = f"{mA}_vs_{mB}"
            per_pair[key] = float(np.mean(vals)) if vals else np.nan

        results[K] = {
            'overall': (w_all, b_all),
            'per_model': per_model,
            'per_pair': per_pair
        }

    return results
    
# proceed to main
def main():
    parser = argparse.ArgumentParser(
        description="Compare tile embeddings across multiple foundation model folders."
    )
    parser.add_argument(
        "--model_dirs", type=str, nargs='+', required=True,
        help="List of folders, each containing slide .h5 files for one foundation model"
    )
    parser.add_argument(
        "--metrics", type=str, nargs='+', default=['all'],
        choices=['cka','svcca','procrustes','knn','ridge','all'],
        help="Metrics to compute (any subset or 'all')"
    )
    parser.add_argument(
        "--sample_size", type=int, default=50000,
        help="Max number of tiles to sample per pair"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--feature_key", type=str, default="features",
        help="H5 key for embedding array"
    )
    parser.add_argument(
        "--coord_key", type=str, default="coords",
        help="H5 key for coordinate array"
    )
    parser.add_argument(
        "--pca_dim", type=int, default=50,
        help="PCA dimension for Procrustes"
    )
    parser.add_argument(
        "--var_thres", type=float, default=0.99,
        help="Variance threshold for SVCCA"
    )
    parser.add_argument(
        "--knn_k", type=int, default=10,
        help="Number of neighbors for k-NN overlap"
    )
    parser.add_argument(
        "--ridge_alpha", type=float, default=1.0,
        help="Alpha (regularization) for Ridge"
    )
    parser.add_argument(
        "--output", type=str, default="comparison_results.xlsx",
        help="Path to the output xlsx file"
    )
    parser.add_argument(
        "--make_figures", action='store_true',
        help="If set, generate and save visualizations for each model pair"
    )
    # consensus-specific arguments (COMMENTED OUT)
    # parser.add_argument(
    #     "--consensus_clusters", type=int, default=5, help="Number of clusters for consensus clustering"
    # )
    # parser.add_argument(
    #     "--consensus_frac", type=float, default=0.3, help="Subsample fraction for consensus clustering"
    # )
    # parser.add_argument(
    #     "--consensus_runs", type=int, default=25, help="Number of runs for consensus clustering"
    # )
    
    args = parser.parse_args()

    model_dirs = args.model_dirs
    pairs = list(itertools.combinations(model_dirs, 2))
    results = []

    print(f"\nComparing {len(pairs)} model pairs...")
    for dir1, dir2 in tqdm(pairs, desc="Processing model pairs"):
        name1 = os.path.basename(os.path.normpath(dir1))
        name2 = os.path.basename(os.path.normpath(dir2))
        base_name = f"{name1}_vs_{name2}"
        try:
            X, Y = load_matched_embeddings(
                dir1, dir2,
                sample_size=args.sample_size,
                seed=args.seed,
                feature_key=args.feature_key,
                coord_key=args.coord_key
            )
        except ValueError as e:
            print(f"Skipping pair {name1} vs {name2}: {e}")
            continue

        metric_set = set(args.metrics)
        if 'all' in metric_set:
            metric_set = {'cka','svcca','procrustes','knn','ridge'}

        res = {'model1': name1, 'model2': name2}
        if 'cka' in metric_set:
            res['cka'] = compute_cka(X, Y)
        if 'svcca' in metric_set:
            res['svcca'] = compute_svcca(X, Y, var_thres=args.var_thres, max_components=args.pca_dim)
        if 'procrustes' in metric_set:
            res['procrustes'] = compute_procrustes(X, Y, pca_dim=args.pca_dim)
        if 'knn' in metric_set:
            mean_jaccard, jaccard_scores = compute_knn_overlap(X, Y, k=args.knn_k)
            res['knn'] = mean_jaccard
        else:
            jaccard_scores = None
        if 'ridge' in metric_set:
            r2_y2x, r2_x2y = compute_ridge_r2(X, Y, alpha=args.ridge_alpha)
            res['r2_y2x'] = r2_y2x
            res['r2_x2y'] = r2_x2y

        results.append(res)

        if args.make_figures:
            out_dir = os.path.dirname(args.output) or '.'
            metrics_for_radar = {k: res[k] for k in ['cka','svcca','procrustes','knn','r2_y2x']}
            radar_path = os.path.join(out_dir, f"{base_name}_radar.png")
            plot_radar(metrics_for_radar, ['cka','svcca','procrustes','knn','r2_y2x'], radar_path)
            if jaccard_scores is not None:
                hist_path = os.path.join(out_dir, f"{base_name}_knn_hist.png")
                plot_knn_histogram(jaccard_scores, hist_path)
            neigh_path = os.path.join(out_dir, f"{base_name}_neighborhoods.png")
            plot_neighborhoods(X, Y, jaccard_scores, neigh_path, k=args.knn_k)

    if len(results) == 0:
        print("No valid comparisons computed.")
        return

    df = pd.DataFrame(results)
    
    # Generate additional summary figures if requested
    if args.make_figures:
        out_dir = os.path.dirname(args.output) or '.'
        
        print("\nGenerating summary visualizations...")
        
        # All models scatter plot
        all_scatter_path = os.path.join(out_dir, "all_models_scatter.png")
        plot_all_models_scatter(model_dirs, all_scatter_path, sample_per_model=5000, pca_dim=args.pca_dim)
        print(f"✓ Saved all-models scatter: {all_scatter_path}")
        
        # Radial plot of average metrics (excluding procrustes)
        print("Generating radial plot...")
        radial_path = os.path.join(out_dir, "radial_average_metrics.png")
        plot_radial_average_metrics(df, radial_path)
        print(f"✓ Saved radial plot: {radial_path}")
        
        # Procrustes heatmap (lower triangle only)
        print("Generating procrustes heatmap...")
        heatmap_path = os.path.join(out_dir, "procrustes_heatmap.png")
        plot_procrustes_heatmap(df, heatmap_path)
        print(f"✓ Saved procrustes heatmap: {heatmap_path}")
        
    # write pairwise
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Pairwise', index=False)
        # CONSENSUS CLUSTERING COMMENTED OUT
        # # consensus clustering summary if requested
        # if 'consensus' in args.metrics or 'all' in args.metrics:
        #  cons = run_consensus_clustering(
        #     model_dirs=model_dirs,
        #     sample_per_model=5000,
        #     pca_dim=args.pca_dim,
        #     cluster_counts=[args.consensus_clusters],
        #     subsample_frac=args.consensus_frac,
        #     n_runs=args.consensus_runs,
        #     random_state=args.seed,
        #     out_dir=out_dir
        # )
        # # flatten and save
        # rows = []
        # for K, stats in cons.items():
        #     w_all, b_all = stats['overall']
        #     rows.append({'K': K, 'type': 'within_all', 'model': 'ALL', 'value': w_all})
        #     rows.append({'K': K, 'type': 'between_all', 'model': 'ALL', 'value': b_all})

        #     # per-model
        #     for m, v in stats['per_model'].items():
        #         rows.append({'K': K, 'type': 'within_model',  'model': m,     'value': v})

        #     # per-pair
        #     for pair_key, v in stats['per_pair'].items():
        #         rows.append({'K': K, 'type': 'between_pair', 'model': pair_key, 'value': v})

        # summary_df = pd.DataFrame(rows)
        # summary_df.to_excel(writer, sheet_name='Consensus', index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Analysis complete!")
    print(f"✓ Saved tile-level comparisons to: {args.output}")
    print(f"{'='*60}")
    
if __name__ == "__main__":
    main()
