from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

import umap
import numpy as np
import matplotlib.pyplot as plt

def reduce_and_cluster(X, n_clusters=3):
    """
    Scales the data, reduces dimensions with UMAP, and performs KMeans clustering.

    :param X: Input feature matrix (2D array-like)
    :param n_clusters: Number of clusters for KMeans
    :return: Reduced features (X_umap) and cluster labels
    """
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply UMAP for dimensionality reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # Apply KMeans for clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Adjust n_clusters as needed
    labels = kmeans.fit_predict(X_umap)

    # Visualize UMAP clusters with labels
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f"UMAP Clustering with KMeans (n_clusters={n_clusters})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.show()

    return X_umap, labels

def anomaly_scores(X, threshold_value=0.9):
    """
    Computes anomaly scores using Local Outlier Factor (LOF).
    
    :param X: Input feature matrix (2D array-like)
    :param threshold_value: Threshold for marking anomalies (0 to 1)
    :return: Tuple (normalized scores, anomaly flags)
    """
    if len(X) < 3:
        return [0] * len(X), [False] * len(X)

    lof = LocalOutlierFactor(n_neighbors=min(5, len(X)-1), contamination='auto')
    lof.fit(X)

    raw_scores = lof.negative_outlier_factor_
    norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

    anomalies = [score > threshold_value for score in norm_scores]

    print(f"Anomaly scores range: {norm_scores.min()} to {norm_scores.max()}")

    return norm_scores.tolist(), anomalies


def determine_optimal_clusters(X, max_clusters=10):
    """
    Uses the Elbow Method to determine the optimal number of clusters for KMeans.
    
    :param X: Input feature matrix (2D array-like)
    :param max_clusters: Maximum number of clusters to test
    :return: None, but plots the elbow graph
    """
    distortions = []
    K_range = range(1, max_clusters + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(K_range, distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method to Determine Optimal Clusters')
    plt.show()
