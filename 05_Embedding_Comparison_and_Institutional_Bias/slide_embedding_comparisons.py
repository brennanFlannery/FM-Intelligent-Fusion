import os
import argparse
import h5py
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict
from tqdm import tqdm
import warnings


def ensure_arial_font():
    """
    Ensure Arial font is available and set it as the default.
    Raises an error if Arial is not found with instructions to install it.
    """
    # Try to refresh font cache (different methods for different matplotlib versions)
    try:
        fm._load_fontmanager(try_read_cache=False)
    except:
        pass  # Cache refresh not critical
    
    # Get list of available fonts
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    
    if 'Arial' not in available_fonts:
        error_msg = """
        ╔═══════════════════════════════════════════════════════════════════╗
        ║  ERROR: Arial font not found on system                           ║
        ╠═══════════════════════════════════════════════════════════════════╣
        ║  Arial font is required for this script but is not installed.    ║
        ║                                                                   ║
        ║  EASY INSTALL (no sudo required):                                ║
        ║    bash install_arial.sh                                         ║
        ║                                                                   ║
        ║  Or manually:                                                    ║
        ║    mkdir -p ~/.fonts                                             ║
        ║    cd ~/.fonts                                                   ║
        ║    wget https://github.com/matomo-org/travis-scripts/raw/\\      ║
        ║         master/fonts/Arial.ttf                                   ║
        ║    fc-cache -fv ~/.fonts                                         ║
        ║    rm -rf ~/.cache/matplotlib                                    ║
        ║                                                                   ║
        ║  After installation, restart your Python session.                ║
        ║  See ARIAL_INSTALL_GUIDE.md for more details.                    ║
        ╚═══════════════════════════════════════════════════════════════════╝
        """
        raise RuntimeError(error_msg)
    
    # Set Arial as the font - MUST be done before creating any figures
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    
    print("✓ Arial font configured successfully")
    return True


def load_matched_slide_embeddings(
    model1_dir: str,
    model2_dir: str,
    feature_key: str = "features"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load slide-level embeddings from two model dirs by matching filenames.
    Returns X (M, d1) and Y (M, d2) where M = #common slides.
    """
    files1 = {f: os.path.join(model1_dir, f) for f in os.listdir(model1_dir) if f.endswith('.h5')}
    files2 = {f: os.path.join(model2_dir, f) for f in os.listdir(model2_dir) if f.endswith('.h5')}
    common = sorted(set(files1) & set(files2))
    if not common:
        raise ValueError(f"No common slide files between {model1_dir} and {model2_dir}")
    X_list, Y_list = [], []
    for fname in common:
        with h5py.File(files1[fname], 'r') as f1:
            emb1 = f1[feature_key][:]
        with h5py.File(files2[fname], 'r') as f2:
            emb2 = f2[feature_key][:]
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        X_list.append(emb1)
        Y_list.append(emb2)
    return np.vstack(X_list), np.vstack(Y_list)

# --- Metric functions (unchanged) ---
def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    cross = Xc.T @ Yc
    hsic_xy = np.linalg.norm(cross, 'fro')**2
    hsic_xx = np.linalg.norm(Xc.T @ Xc, 'fro')
    hsic_yy = np.linalg.norm(Yc.T @ Yc, 'fro')
    return hsic_xy / (hsic_xx * hsic_yy + 1e-12)

def compute_svcca(X: np.ndarray, Y: np.ndarray, var_thres: float=0.99, max_components: int=50) -> float:
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Ux, Sx, _ = np.linalg.svd(Xc, full_matrices=False)
    cumx = np.cumsum((Sx**2)/np.sum(Sx**2))
    kx = min(np.searchsorted(cumx, var_thres)+1, max_components)
    Uy, Sy, _ = np.linalg.svd(Yc, full_matrices=False)
    cumy = np.cumsum((Sy**2)/np.sum(Sy**2))
    ky = min(np.searchsorted(cumy, var_thres)+1, max_components)
    Xr, Yr = Ux[:,:kx], Uy[:,:ky]
    cca = CCA(n_components=min(kx, ky))
    Xs, Ys = cca.fit_transform(Xr, Yr)
    return float(np.mean([np.corrcoef(Xs[:,i], Ys[:,i])[0,1] for i in range(Xs.shape[1])]))

def compute_procrustes(X: np.ndarray, Y: np.ndarray, pca_dim: int=50) -> float:
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    px = PCA(n_components=min(pca_dim, Xc.shape[1])).fit_transform(Xc)
    py = PCA(n_components=min(pca_dim, Yc.shape[1])).fit_transform(Yc)
    d = min(px.shape[1], py.shape[1])
    px, py = px[:,:d], py[:,:d]
    R, _ = orthogonal_procrustes(px, py)
    return np.linalg.norm(px @ R - py, 'fro') / (np.linalg.norm(py, 'fro') + 1e-12)

def compute_knn_overlap(X: np.ndarray, Y: np.ndarray, k: int=10) -> float:
    nbrsX = NearestNeighbors(n_neighbors=k+1).fit(X)
    nbrsY = NearestNeighbors(n_neighbors=k+1).fit(Y)
    ix, iy = nbrsX.kneighbors(X)[1], nbrsY.kneighbors(Y)[1]
    js = []
    for i in range(len(X)):
        s1, s2 = set(ix[i,1:]), set(iy[i,1:])
        js.append(len(s1&s2)/len(s1|s2))
    return float(np.mean(js))

def compute_ridge_r2(X: np.ndarray, Y: np.ndarray, alpha: float=1.0) -> Tuple[float,float]:
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    r1 = Ridge(alpha=alpha, fit_intercept=False).fit(Yc, Xc)
    r2_y2x = 1 - np.sum((Xc - r1.predict(Yc))**2)/np.sum(Xc**2)
    r2 = Ridge(alpha=alpha, fit_intercept=False).fit(Xc, Yc)
    r2_x2y = 1 - np.sum((Yc - r2.predict(Xc))**2)/np.sum(Yc**2)
    return float(r2_y2x), float(r2_x2y)
    
def load_all_slide_embeddings(
    model_dir: str,
    feature_key: str = "features"
) -> np.ndarray:
    """
    Load every slide embedding from a model folder.
    Returns an (M, D) array of embeddings, one per .h5 file.
    """
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".h5"))
    out = []
    for fn in tqdm(files, desc=f"Loading slides from {os.path.basename(model_dir)}", leave=False):
        with h5py.File(os.path.join(model_dir, fn), "r") as f:
            emb = f[feature_key][:]
        out.append(emb.flatten())
    return np.vstack(out)
    
def plot_all_slide_scatter(model_dirs: List[str], output_path: str, pca_dim: int):
    """Create a 2D PCA scatter for all slide embeddings across models."""
    # Model color mapping
    model_colors = {
        'features_conch_v15': '#0070C0',
        'features_musk': '#FFC000',
        'features_hoptimus1': '#C00000',
        'features_virchow2': '#7030A0',
        'features_gigapath': '#00B050',
        'slide_features_titan': '#F6C6AD',
        'slide_features_madeleine': '#A6CAEC',
        'slide_features_chief': '#B4E5A2'
    }
    
    samples, labels = [], []
    print("\nGenerating all-models scatter plot...")
    for md in tqdm(model_dirs, desc="Loading models for scatter", leave=False):
        X = load_all_slide_embeddings(md)
        pca = PCA(n_components=min(pca_dim, X.shape[1]))
        Xr = pca.fit_transform(X)
        samples.append(Xr)
        labels += [os.path.basename(os.path.normpath(md))] * Xr.shape[0]
    combined = np.vstack(samples)
    proj2d = PCA(n_components=2).fit_transform(combined)
    fig, ax = plt.subplots(figsize=(6,6))
    start = 0
    default_colors = plt.cm.tab10.colors
    for i, md in enumerate(model_dirs):
        name = os.path.basename(os.path.normpath(md))
        cnt = samples[i].shape[0]
        pts = proj2d[start:start+cnt]
        color = model_colors.get(name, default_colors[i % len(default_colors)])
        ax.scatter(pts[:,0], pts[:,1], s=20, alpha=1.0, color=color)
        start += cnt
    ax.set_title('Slide PCA Scatter: All Models', fontsize=24, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=24, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(output_path, dpi=500)
    plt.close(fig)


def plot_slide_embedding_scatter(X: np.ndarray, Y: np.ndarray, name1: str, name2: str, output_path: str):
    """PCA scatter of slide embeddings for two models."""
    # Model color mapping
    model_colors = {
        'features_conch_v15': '#0070C0',
        'features_musk': '#FFC000',
        'features_hoptimus1': '#C00000',
        'features_virchow2': '#7030A0',
        'features_gigapath': '#00B050',
        'slide_features_titan': '#F6C6AD',
        'slide_features_madeleine': '#A6CAEC',
        'slide_features_chief': '#B4E5A2'
    }
    
    pca1 = PCA(n_components=2).fit(X)
    pca2 = PCA(n_components=2).fit(Y)
    proj1 = pca1.transform(X)
    proj2 = pca2.transform(Y)
    fig, ax = plt.subplots(figsize=(6,6))
    color1 = model_colors.get(name1, 'C0')
    color2 = model_colors.get(name2, 'C1')
    ax.scatter(proj1[:,0], proj1[:,1], s=20, alpha=1.0, color=color1)
    ax.scatter(proj2[:,0], proj2[:,1], s=20, alpha=1.0, color=color2)
    ax.set_title(f'Slide PCA Scatter: {name1} vs {name2}', fontsize=24, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=24, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(output_path, dpi=500)
    plt.close(fig)


def plot_slide_radar(metrics: dict, labels: List[str], output_path: str):
    """Radar plot for slide-level metrics."""
    # same as plot_radar
    values = [metrics[l] for l in labels]
    if 'procrustes' in labels:
        idx = labels.index('procrustes')
        values[idx] = 1.0 / (1.0 + values[idx])
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=20)
    ax.set_ylim(0,1)
    ax.set_title("Slide Radar: " + os.path.splitext(os.path.basename(output_path))[0], fontsize=24, fontweight='bold')
    ax.tick_params(axis='y', labelsize=20)
    ax.spines['polar'].set_visible(False)
    fig.savefig(output_path, dpi=500)
    plt.close(fig)


def plot_slide_knn_histogram(jaccard_scores: np.ndarray, output_path: str):
    """Histogram of slide-level k-NN Jaccard scores."""
    fig, ax = plt.subplots()
    ax.hist(jaccard_scores, bins=10, color='dodgerblue', edgecolor='black')
    ax.set_xlabel('Jaccard Index (k-NN overlap)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=24, fontweight='bold')
    ax.set_title('Slide k-NN Jaccard Histogram', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(output_path, dpi=500)
    plt.close(fig)


def plot_slide_neighborhoods(X: np.ndarray, Y: np.ndarray, jaccard_scores: np.ndarray, output_path: str, k=10, probes=3):
    """Neighborhoods for slide embeddings (few samples)."""
    indices = np.random.choice(X.shape[0], size=min(probes, X.shape[0]), replace=False)
    fig, axs = plt.subplots(len(indices), 2, figsize=(8,4*len(indices)))
    for i, idx in enumerate(indices):
        # X neighborhood
        nbrsX = NearestNeighbors(n_neighbors=k+1).fit(X)
        _, ix = nbrsX.kneighbors(X[idx].reshape(1,-1))
        ptsX = X[ix[0]]
        projX = PCA(n_components=2).fit_transform(ptsX)
        axX = axs[i,0] if len(indices)>1 else axs[0]
        axX.scatter(projX[0,0], projX[0,1], color='black', s=50)
        axX.scatter(projX[1:,0], projX[1:,1], color='blue', s=20)
        axX.set_title(f'Model A Slide Neighborhood {idx}', fontsize=24, fontweight='bold')
        axX.tick_params(axis='both', which='major', labelsize=20)
        axX.spines['top'].set_visible(False)
        axX.spines['right'].set_visible(False)
        # Y neighborhood
        nbrsY = NearestNeighbors(n_neighbors=k+1).fit(Y)
        _, iy = nbrsY.kneighbors(Y[idx].reshape(1,-1))
        ptsY = Y[iy[0]]
        projY = PCA(n_components=2).fit_transform(ptsY)
        axY = axs[i,1] if len(indices)>1 else axs[1]
        axY.scatter(projY[0,0], projY[0,1], color='black', s=50)
        axY.scatter(projY[1:,0], projY[1:,1], color='red', s=20)
        axY.set_title(f'Model B Slide Neighborhood {idx}', fontsize=24, fontweight='bold')
        axY.tick_params(axis='both', which='major', labelsize=20)
        axY.spines['top'].set_visible(False)
        axY.spines['right'].set_visible(False)
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
        'features_gigapath': '#00B050',
        'slide_features_titan': '#F6C6AD',
        'slide_features_madeleine': '#A6CAEC',
        'slide_features_chief': '#B4E5A2'
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
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Find max value across all models for adaptive y-axis
    all_values = []
    for model, values in model_avg_metrics.items():
        all_values.extend([v for v in values if not np.isnan(v)])
    max_value = np.max(all_values) if all_values else 1.0
    
    # Round up to nearest 0.1 and add 10% padding
    y_max = np.ceil((max_value * 1.1) * 10) / 10
    
    for model, values in model_avg_metrics.items():
        values_plot = values + values[:1]  # Complete the circle
        color = model_colors.get(model, '#CCCCCC')
        ax.plot(angles, values_plot, 'o-', linewidth=3, color=color, markersize=8, alpha=1.0)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=20, fontweight='normal')
    ax.set_ylim(0, y_max)
    
    # Generate adaptive tick marks
    n_ticks = 5
    tick_values = np.linspace(0, y_max, n_ticks + 1)[1:]  # Skip 0
    tick_labels = [f'{v:.1f}' for v in tick_values]
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels, fontsize=20)
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
        'features_gigapath': '#00B050',
        'slide_features_titan': '#F6C6AD',
        'slide_features_madeleine': '#A6CAEC',
        'slide_features_chief': '#B4E5A2'
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
    
    # Create figure with square aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create heatmap with lower triangle only - cool colormap, perfect squares
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
    
# CONSENSUS CLUSTERING FUNCTIONS COMMENTED OUT
# # Consensus clustering functions (unchanged) ...
# from sklearn.cluster import KMeans

# def cluster_once(X: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
#     """Perform a single k-means clustering run on data X."""
#     km = KMeans(n_clusters=n_clusters, random_state=random_state)
#     return km.fit_predict(X)


# def compute_consensus_matrix(
#     X: np.ndarray,
#     n_clusters: int,
#     subsample_frac: float = 0.8,
#     n_runs: int = 100,
#     random_state: int = 0
# ) -> np.ndarray:
#     """
#     Build an (N,N) consensus matrix by running k-means on random subsamples of X.
#     M[i,j] is the fraction of runs where i and j co-clustered.
#     """
#     N = X.shape[0]
#     counts = np.zeros((N, N), dtype=int)
#     consensus = np.zeros((N, N), dtype=int)
#     rng = np.random.RandomState(random_state)
#     for run in tqdm(range(n_runs), desc=f"Consensus runs (K={n_clusters})", leave=False):
#         idx = rng.choice(N, size=int(subsample_frac * N), replace=False)
#         labels = cluster_once(X[idx], n_clusters=n_clusters, random_state=random_state + run)
#         for pi, i in enumerate(idx):
#             for pj, j in enumerate(idx):
#                 counts[i, j] += 1
#                 if labels[pi] == labels[pj]:
#                     consensus[i, j] += 1
#     M = np.zeros((N, N), dtype=float)
#     nonzero = counts > 0
#     M[nonzero] = consensus[nonzero] / counts[nonzero]
#     return M


# def summarize_model_agreement(
#     consensus: np.ndarray,
#     model_labels: List[str]
# ) -> Tuple[float, float]:
#     """
#     Compute mean within-model and between-model consensus.
#     """
#     N = len(model_labels)
#     within = []
#     between = []
#     for i in range(N):
#         for j in range(i):
#             if model_labels[i] == model_labels[j]:
#                 within.append(consensus[i, j])
#             else:
#                 between.append(consensus[i, j])
#     return float(np.mean(within)), float(np.mean(between))
    
# def run_consensus_clustering(
#     model_dirs: List[str],
#     sample_per_model: int,
#     pca_dim: int,
#     cluster_counts: List[int],
#     subsample_frac: float,
#     n_runs: int,
#     random_state: int,
#     out_dir: str
# ) -> Dict[int, Dict]:
#     """
#     For each K in cluster_counts, sample embeddings by model (slide-level),
#     reduce to common PCA space, compute consensus clustering, and return:
#       - overall: (within_all, between_all)
#       - per_model: {model_name: within_model}
#       - per_pair: { "modelA_vs_modelB": between_pair }
#     """
#     # 1) Load & PCA‐reduce all slides per model
#     samples, labels = [], []
#     print("\nRunning consensus clustering...")
#     for md in tqdm(model_dirs, desc="Loading models for consensus", leave=False):
#         X_md = load_all_slide_embeddings(md)
#         pca = PCA(n_components=min(pca_dim, X_md.shape[1]))
#         Xr = pca.fit_transform(X_md)
#         samples.append(Xr)
#         labels += [os.path.basename(md)] * Xr.shape[0]
#     X_all = np.vstack(samples)

#     # 2) Precompute indices
#     labels = np.array(labels)
#     models = sorted(set(labels))
#     idx_by_model = {m: np.where(labels == m)[0] for m in models}
#     # all unordered model‐pairs
#     model_pairs = [(models[i], models[j])
#                    for i in range(len(models)) for j in range(i+1, len(models))]

#     results = {}
#     for K in tqdm(cluster_counts, desc="Computing consensus for K values"):
#         M = compute_consensus_matrix(
#             X_all,
#             n_clusters=K,
#             subsample_frac=subsample_frac,
#             n_runs=n_runs,
#             random_state=random_state
#         )
#         # overall within vs between
#         w_all, b_all = summarize_model_agreement(M, labels.tolist())

#         # per‐model within
#         per_model = {}
#         for m, idxs in idx_by_model.items():
#             vals = [M[i,j] for i in idxs for j in idxs if j < i]
#             per_model[m] = float(np.mean(vals)) if vals else np.nan

#         # per‐pair between
#         per_pair = {}
#         for mA, mB in model_pairs:
#             idxsA, idxsB = idx_by_model[mA], idx_by_model[mB]
#             vals = [M[i,j] for i in idxsA for j in idxsB]
#             key = f"{mA}_vs_{mB}"
#             per_pair[key] = float(np.mean(vals)) if vals else np.nan

#         results[K] = {
#             'overall': (w_all, b_all),
#             'per_model': per_model,
#             'per_pair': per_pair
#         }

#     return results
    
    
# === Main function ===
def main():
    parser = argparse.ArgumentParser(description="Slide-level embedding comparisons")
    parser.add_argument("--model_dirs", nargs='+', required=True)
    parser.add_argument("--metrics", nargs='+', default=['all'], choices=['cka','svcca','procrustes','knn','ridge','consensus','all'])
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--var_thres", type=float, default=0.99)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    # CONSENSUS CLUSTERING ARGUMENTS COMMENTED OUT
    # parser.add_argument("--consensus_K", nargs='+', type=int, default=[5,10,20])
    # parser.add_argument("--consensus_frac", type=float, default=0.8)
    # parser.add_argument("--consensus_runs", type=int, default=100)
    parser.add_argument("--make_figures", action='store_true', help="Generate slide-level figures")
    parser.add_argument("--output", type=str, default="slide_comparison.xlsx")
    args = parser.parse_args()
    
    # Configure Arial font globally BEFORE any plotting
    ensure_arial_font()

    model_dirs = args.model_dirs
    # slide-level: match all patients across every pair
    pairs = list(itertools.combinations(model_dirs, 2))
    results = []
    
    print(f"\nComparing {len(pairs)} model pairs...")
    for d1,d2 in tqdm(pairs, desc="Processing model pairs"):
        n1, n2 = os.path.basename(d1), os.path.basename(d2)
        X, Y = load_matched_slide_embeddings(d1,d2)
        res = {'model1':n1,'model2':n2}
        if 'cka' in args.metrics or 'all' in args.metrics:
            res['cka'] = compute_cka(X, Y)
        if 'svcca' in args.metrics or 'all' in args.metrics:
            res['svcca'] = compute_svcca(X, Y, args.var_thres, args.pca_dim)
        if 'procrustes' in args.metrics or 'all' in args.metrics:
            res['procrustes'] = compute_procrustes(X, Y, args.pca_dim)
        if 'knn' in args.metrics or 'all' in args.metrics:
            res['knn'] = compute_knn_overlap(X, Y, args.knn_k)
        if 'ridge' in args.metrics or 'all' in args.metrics:
            r2_y2x, r2_x2y = compute_ridge_r2(X, Y, args.ridge_alpha)
            res['r2_y2x'], res['r2_x2y'] = r2_y2x, r2_x2y
        results.append(res)

            # Slide-level pairwise figures
        if args.make_figures:
            fig_dir = os.path.dirname(args.output) or '.'
            # Pairwise PCA scatter
            scatter_path = os.path.join(fig_dir, f"{n1}_vs_{n2}_scatter.png")
            plot_slide_embedding_scatter(X, Y, n1, n2, scatter_path)
            # Pairwise radar
            radar_metrics = {k: res[k] for k in ['cka','svcca','procrustes','knn','r2_y2x'] if k in res}
            radar_path = os.path.join(fig_dir, f"{n1}_vs_{n2}_radar.png")
            plot_slide_radar(radar_metrics, list(radar_metrics.keys()), radar_path)
            
    if args.make_figures:
        fig_dir = os.path.dirname(args.output) or '.'
        
        print("\nGenerating summary visualizations...")
        
        # All models scatter plot
        all_scatter_path = os.path.join(fig_dir, "all_models_scatter.png")
        plot_all_slide_scatter(model_dirs, all_scatter_path, args.pca_dim)
        print(f"✓ Saved all-models scatter: {all_scatter_path}")
        
        # Radial plot of average metrics (excluding procrustes)
        print("Generating radial plot...")
        radial_path = os.path.join(fig_dir, "radial_average_metrics.png")
        plot_radial_average_metrics(pd.DataFrame(results), radial_path)
        print(f"✓ Saved radial plot: {radial_path}")
        
        # Procrustes heatmap (lower triangle only)
        print("Generating procrustes heatmap...")
        heatmap_path = os.path.join(fig_dir, "procrustes_heatmap.png")
        plot_procrustes_heatmap(pd.DataFrame(results), heatmap_path)
        print(f"✓ Saved procrustes heatmap: {heatmap_path}")

    df = pd.DataFrame(results)
    # write pairwise
    with pd.ExcelWriter(args.output, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Pairwise', index=False)
        # CONSENSUS CLUSTERING COMMENTED OUT
        # # consensus clustering summary if requested
        # if 'consensus' in args.metrics or 'all' in args.metrics:
        #     cons = run_consensus_clustering(
        #         model_dirs, sample_per_model=len(df), pca_dim=args.pca_dim,
        #         cluster_counts=args.consensus_K, subsample_frac=args.consensus_frac,
        #         n_runs=args.consensus_runs, random_state=0, out_dir='.'
        #     )
        #     # flatten and save
        #     rows = []
        #     for K, stats in cons.items():
        #         w_all, b_all = stats['overall']
        #         rows.append({'K': K, 'type': 'within_all', 'model': 'ALL', 'value': w_all})
        #         rows.append({'K': K, 'type': 'between_all', 'model': 'ALL', 'value': b_all})

        #         # per-model
        #         for m, v in stats['per_model'].items():
        #             rows.append({'K': K, 'type': 'within_model',  'model': m,     'value': v})

        #         # per-pair
        #         for pair_key, v in stats['per_pair'].items():
        #             rows.append({'K': K, 'type': 'between_pair', 'model': pair_key, 'value': v})

        #     summary_df = pd.DataFrame(rows)
        #     summary_df.to_excel(w, sheet_name='Consensus', index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Analysis complete!")
    print(f"✓ Saved slide-level comparisons to: {args.output}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
