"""
Mini Project 3: Iris Flower Clustering Project
------------------------------------------------
Skill Gain: Unsupervised ML + Visualization

Tasks covered:
1. Perform K-Means on the Iris dataset and visualize clusters.
2. Apply PCA to reduce dataset dimensions.
3. Apply K-Means (k=3).
4. Visualize clusters (scatter plot).
5. Compare predicted clusters vs true labels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score
from scipy.stats import mode

# ---------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------
df = pd.read_csv("Iris.csv")

feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[feature_cols].values
y_true_labels = df["Species"].values

# Encode species names to integers 0,1,2 for comparison later
species_order = sorted(df["Species"].unique())
species_to_int = {name: i for i, name in enumerate(species_order)}
y_true = np.array([species_to_int[s] for s in y_true_labels])

print("Dataset shape:", df.shape)
print("Species classes:", species_order)

# ---------------------------------------------------------
# 2. Standardize features
# ---------------------------------------------------------
# K-Means is distance-based, so features are scaled to have
# mean 0 and unit variance -- otherwise Petal measurements
# (larger numeric range) would dominate the distance metric.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 3. Apply K-Means (k=3) on the full 4D feature space
# ---------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# ---------------------------------------------------------
# 4. Apply PCA to reduce to 2 dimensions (for visualization only)
# ---------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)
print("Total variance captured by 2 PCs: {:.2f}%".format(
    pca.explained_variance_ratio_.sum() * 100))

# ---------------------------------------------------------
# 5. Visualize clusters (scatter plot in PCA space)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: K-Means predicted clusters
scatter1 = axes[0].scatter(
    X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=45,
    edgecolor="k", linewidth=0.4
)
centers_pca = pca.transform(kmeans.cluster_centers_)
axes[0].scatter(
    centers_pca[:, 0], centers_pca[:, 1], c="red", marker="X",
    s=200, edgecolor="black", linewidth=1.2, label="Centroids"
)
axes[0].set_title("K-Means Clusters (k=3) — PCA-reduced view")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: True species labels, same PCA coordinates
scatter2 = axes[1].scatter(
    X_pca[:, 0], X_pca[:, 1], c=y_true, cmap="viridis", s=45,
    edgecolor="k", linewidth=0.4
)
axes[1].set_title("True Species Labels — same PCA view")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
axes[1].grid(alpha=0.3)

handles, _ = scatter2.legend_elements()
axes[1].legend(handles, species_order, title="Species")

plt.tight_layout()
plt.savefig("clusters_vs_true_labels.png", dpi=150)
plt.show()
print("\nSaved figure: clusters_vs_true_labels.png")

# ---------------------------------------------------------
# 6. Compare predicted clusters vs true labels
# ---------------------------------------------------------
# K-Means cluster IDs (0,1,2) are arbitrary and don't necessarily
# line up with the species IDs (0,1,2). We remap each cluster to
# the species label that appears most often inside it, so accuracy
# is measured fairly.
labels_matched = np.zeros_like(clusters)
for cluster_id in np.unique(clusters):
    mask = clusters == cluster_id
    most_common_label = mode(y_true[mask], keepdims=True).mode[0]
    labels_matched[mask] = most_common_label

acc = accuracy_score(y_true, labels_matched)
ari = adjusted_rand_score(y_true, clusters)
cm = confusion_matrix(y_true, labels_matched)

print("\n" + "=" * 50)
print("CLUSTER vs TRUE LABEL COMPARISON")
print("=" * 50)
print(f"Accuracy after best cluster-to-label matching: {acc*100:.2f}%")
print(f"Adjusted Rand Index (label-independent agreement): {ari:.3f}")
print("\nConfusion matrix (rows = true species, cols = matched cluster):")
cm_df = pd.DataFrame(cm, index=species_order, columns=species_order)
print(cm_df)

# Confusion matrix heatmap
fig2, ax2 = plt.subplots(figsize=(5.5, 4.8))
im = ax2.imshow(cm, cmap="Blues")
ax2.set_xticks(range(3)); ax2.set_xticklabels(species_order, rotation=30, ha="right")
ax2.set_yticks(range(3)); ax2.set_yticklabels(species_order)
ax2.set_xlabel("Predicted cluster (matched to species)")
ax2.set_ylabel("True species")
ax2.set_title("Confusion Matrix: K-Means vs True Labels")
for i in range(3):
    for j in range(3):
        ax2.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("\nSaved figure: confusion_matrix.png")

print("\nDone.")
