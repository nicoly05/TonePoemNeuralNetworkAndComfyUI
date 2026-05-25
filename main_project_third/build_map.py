import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from dataset import LatentDataset

N_CLUSTERS  = 7       
MAP_PATH    = "models/latent_map.pt"
PLOT_PATH   = "models/latent_map_pca.png"


def build_map(n_clusters: int = N_CLUSTERS) -> None:
    os.makedirs("models", exist_ok=True)

    # ── Load latents ──────────────────────────────────────────────────────────
    ds = LatentDataset()
    latents_t = ds.latents                     
    latents   = latents_t.numpy()              
    filenames = ds.files
    N         = len(latents)

    print(f"Loaded {N} latent vectors, dim={latents.shape[1]}")

    # ── K-means clustering ────────────────────────────────────────────────────
    print(f"Running k-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = kmeans.fit_predict(latents)        
    centers = kmeans.cluster_centers_           

    # Print cluster sizes
    for c in range(n_clusters):
        count = np.sum(labels == c)
        print(f"  Cluster {c}: {count} sounds")

    # ── Save map ──────────────────────────────────────────────────────────────
    torch.save({
        "latents":        latents_t,
        "filenames":      filenames,
        "cluster_labels": torch.from_numpy(labels).long(),
        "cluster_centers":torch.from_numpy(centers).float(),
        "n_clusters":     n_clusters,
    }, MAP_PATH)
    print(f"\nLatent map saved to {MAP_PATH}")

    # ── PCA visualisation ─────────────────────────────────────────────────────
    print("Generating PCA plot...")
    pca   = PCA(n_components=2)
    pts   = pca.fit_transform(latents)           
    c_pts = pca.transform(centers)          

    cmap  = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6))

    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(pts[mask, 0], pts[mask, 1],
                   color=cmap(c), alpha=0.7, s=40, label=f"Cluster {c}")

    ax.scatter(c_pts[:, 0], c_pts[:, 1],
               color="black", marker="X", s=150, zorder=5, label="Centroids")

    for c in range(n_clusters):
        ax.annotate(str(c), c_pts[c], fontsize=11, fontweight="bold",
                    ha="center", va="center", color="white",
                    bbox=dict(boxstyle="circle,pad=0.3", fc=cmap(c), ec="black"))

    ax.set_title(f"Latent Space — {n_clusters} Clusters (PCA projection)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"PCA plot saved to {PLOT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=N_CLUSTERS,
                        help="Number of k-means clusters (default: 7)")
    args = parser.parse_args()
    build_map(n_clusters=args.clusters)
