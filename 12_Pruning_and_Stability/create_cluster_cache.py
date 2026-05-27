from __future__ import print_function

# Example usage:
# python create_cluster_cache.py \
#   --data_root_dir "/path/to/h5_features" \
#   --csv_path "/path/to/slides.csv" \
#   --cluster_cache_dir "/path/to/cluster_cache" \
#   --n_clusters 5 \
#   --sample_frac 0.1 \
#   --max_tiles_per_slide 1000 \
#   --local_k 10 \
#   --local_batch_size 256 \
#   --batch_size 10000 \
#   --chunk_size 20000 \
#   --seed 42

import argparse
import json
import os
import random

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_slide_h5_path(slide_id, data_dir):
    return os.path.join(data_dir, f"{slide_id}.h5")


def filter_slides_with_data(slide_ids, data_dir):
    """Return only slide_ids for which the corresponding .h5 file exists."""
    present = []
    missing = []
    for slide_id in slide_ids:
        path = get_slide_h5_path(slide_id, data_dir)
        if os.path.isfile(path):
            present.append(slide_id)
        else:
            missing.append(slide_id)
    return present, missing


def load_slide_embeddings(slide_id, data_dir):
    h5_path = get_slide_h5_path(slide_id, data_dir)
    with h5py.File(h5_path, "r") as hdf5_file:
        features = hdf5_file["features"][:]
        coords = hdf5_file["coords"][:]
    return features, coords


def stratified_sample_embeddings(
    slide_ids,
    data_dir,
    sample_frac,
    max_tiles_per_slide,
    local_k,
    local_batch_size,
    seed,
):
    samples = []
    n_total = len(slide_ids)
    for i, slide_id in enumerate(slide_ids):
        if (i + 1) % 25 == 0 or i == 0 or i == n_total - 1:
            print("  sampling slide {}/{}".format(i + 1, n_total))
        features, _ = load_slide_embeddings(slide_id, data_dir)
        n_tiles_total = len(features)
        if n_tiles_total == 0:
            continue

        n_tiles = min(n_tiles_total, max_tiles_per_slide)
        slide_seed = (seed + hash(slide_id)) % (2**32 - 1)
        rng = np.random.default_rng(slide_seed)
        tile_idx = rng.choice(n_tiles_total, size=n_tiles, replace=False)
        tile_feats = features[tile_idx]

        effective_k = min(local_k, n_tiles)
        local_batch = min(local_batch_size, n_tiles)
        local_centroids = fit_kmeans(tile_feats, effective_k, local_batch, seed)
        distances = cdist(tile_feats, local_centroids, "euclidean")
        local_cluster_ids = np.argmin(distances, axis=1)

        total_sample = max(1, int(n_tiles * sample_frac))
        counts = np.bincount(local_cluster_ids, minlength=effective_k)
        proportions = counts / counts.sum()
        samples_per_cluster = np.maximum(1, (proportions * total_sample).astype(int))

        diff = samples_per_cluster.sum() - total_sample
        if diff > 0:
            for idx in np.argsort(-samples_per_cluster):
                if diff == 0:
                    break
                if samples_per_cluster[idx] > 1:
                    samples_per_cluster[idx] -= 1
                    diff -= 1
        elif diff < 0:
            for idx in np.argsort(-samples_per_cluster):
                if diff == 0:
                    break
                samples_per_cluster[idx] += 1
                diff += 1

        slide_samples = []
        for cluster_id in range(effective_k):
            cluster_mask = local_cluster_ids == cluster_id
            cluster_tiles = tile_feats[cluster_mask]
            if len(cluster_tiles) == 0:
                continue
            n_sample = min(samples_per_cluster[cluster_id], len(cluster_tiles))
            indices = rng.choice(len(cluster_tiles), size=n_sample, replace=False)
            slide_samples.append(cluster_tiles[indices])

        if slide_samples:
            samples.append(np.concatenate(slide_samples, axis=0))

    if len(samples) == 0:
        raise ValueError("No embeddings found to sample")
    return np.concatenate(samples, axis=0)


def fit_kmeans(embeddings, n_clusters, batch_size, seed):
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=seed,
    )
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_


def assign_clusters_streaming(slide_id, data_dir, centroids, output_dir, chunk_size):
    h5_path = os.path.join(data_dir, f"{slide_id}.h5")
    cache_path = os.path.join(output_dir, f"{slide_id}_clusters.h5")
    with h5py.File(h5_path, "r") as f_in, h5py.File(cache_path, "w") as f_out:
        feats = f_in["features"]
        n_tiles = feats.shape[0]
        out = f_out.create_dataset("cluster_ids", shape=(n_tiles,), dtype="i8")
        for start in range(0, n_tiles, chunk_size):
            end = min(n_tiles, start + chunk_size)
            batch = feats[start:end]
            distances = cdist(batch, centroids, "euclidean")
            out[start:end] = np.argmin(distances, axis=1)
        f_out.create_dataset("centroids", data=centroids)


def assign_clusters(slide_id, data_dir, centroids, output_dir):
    h5_path = os.path.join(data_dir, f"{slide_id}.h5")
    cache_path = os.path.join(output_dir, f"{slide_id}_clusters.h5")
    with h5py.File(h5_path, "r") as f_in, h5py.File(cache_path, "w") as f_out:
        feats = f_in["features"][:]
        distances = cdist(feats, centroids, "euclidean")
        cluster_ids = np.argmin(distances, axis=1)
        f_out.create_dataset("cluster_ids", data=cluster_ids)
        f_out.create_dataset("centroids", data=centroids)


def save_centroids(centroids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "centroids.npy"), centroids)


def save_metadata(args, output_dir, n_embeddings, n_slides_present=None, n_slides_missing=None):
    meta = {
        "n_clusters": args.n_clusters,
        "sample_frac": args.sample_frac,
        "max_tiles_per_slide": args.max_tiles_per_slide,
        "local_k": args.local_k,
        "local_batch_size": args.local_batch_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_embeddings": n_embeddings,
        "data_root_dir": args.data_root_dir,
        "csv_path": args.csv_path,
    }
    if n_slides_present is not None:
        meta["n_slides_present"] = n_slides_present
    if n_slides_missing is not None:
        meta["n_slides_missing"] = n_slides_missing
    with open(os.path.join(output_dir, "cluster_cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main(args):
    os.makedirs(args.cluster_cache_dir, exist_ok=True)
    set_seed(args.seed)

    print("Reading CSV and checking for H5 files...")
    df = pd.read_csv(args.csv_path)
    if "slide_id" not in df.columns:
        raise ValueError("CSV must contain a 'slide_id' column")
    all_slide_ids = df["slide_id"].tolist()

    slide_ids, missing_ids = filter_slides_with_data(all_slide_ids, args.data_root_dir)
    if not slide_ids:
        raise ValueError(
            "No H5 files found for any slide in the CSV. "
            "Check that --data_root_dir contains <slide_id>.h5 for the slide_ids in --csv_path."
        )
    if missing_ids:
        n_missing = len(missing_ids)
        print("Skipping {} slide(s) with missing H5 data:".format(n_missing))
        for sid in missing_ids[:10]:
            print("  - {}".format(sid))
        if n_missing > 10:
            print("  ... and {} more".format(n_missing - 10))
        print("Processing {} slide(s) with present data.".format(len(slide_ids)))
        missing_path = os.path.join(args.cluster_cache_dir, "missing_slides.txt")
        with open(missing_path, "w") as f:
            for sid in missing_ids:
                f.write(sid + "\n")
        print("Missing slide IDs written to {}.".format(missing_path))

    print("Stratified sampling from {} slides (this may take a while)...".format(len(slide_ids)))
    embeddings = stratified_sample_embeddings(
        slide_ids,
        args.data_root_dir,
        args.sample_frac,
        args.max_tiles_per_slide,
        args.local_k,
        args.local_batch_size,
        args.seed,
    )
    print("Fitting global k-means (n_clusters={})...".format(args.n_clusters))
    centroids = fit_kmeans(embeddings, args.n_clusters, args.batch_size, args.seed)
    save_centroids(centroids, args.cluster_cache_dir)
    save_metadata(
        args,
        args.cluster_cache_dir,
        embeddings.shape[0],
        n_slides_present=len(slide_ids),
        n_slides_missing=len(missing_ids),
    )

    n_slides = len(slide_ids)
    print("Writing cluster assignments for {} slides...".format(n_slides))
    for idx, slide_id in enumerate(slide_ids):
        if (idx + 1) % 50 == 0 or idx == 0 or idx == n_slides - 1:
            print("  slide {}/{}".format(idx + 1, n_slides))
        if args.chunk_size > 0:
            assign_clusters_streaming(
                slide_id,
                args.data_root_dir,
                centroids,
                args.cluster_cache_dir,
                args.chunk_size,
            )
        else:
            assign_clusters(slide_id, args.data_root_dir, centroids, args.cluster_cache_dir)

    print("Done. Cache written to {}.".format(args.cluster_cache_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create cached cluster assignments for Sub-CLAM")
    parser.add_argument("--data_root_dir", type=str, required=True, help="Directory containing slide-level H5 features")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file with slide_id column")
    parser.add_argument("--cluster_cache_dir", type=str, required=True, help="Output directory for cluster cache")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of k-means clusters")
    parser.add_argument("--sample_frac", type=float, default=0.1, help="Fraction of tiles to sample per slide")
    parser.add_argument("--max_tiles_per_slide", type=int, default=1000, help="Max tiles per slide to sample")
    parser.add_argument("--local_k", type=int, default=10, help="Local clusters per slide for stratified sampling")
    parser.add_argument("--local_batch_size", type=int, default=256, help="MiniBatchKMeans batch size for local clustering")
    parser.add_argument("--batch_size", type=int, default=10000, help="MiniBatchKMeans batch size")
    parser.add_argument("--chunk_size", type=int, default=20000, help="Chunk size for streaming assignments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
