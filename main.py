"""Task 2: Customer Segmentation (K-Means + bonus DBSCAN & spending analysis)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/Mall_Customers.csv"
OUT_DIR = "outputs"
INCOME = "Annual Income (k$)"
SCORE = "Spending Score (1-100)"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load + clean
df = pd.read_csv(DATA_PATH).drop_duplicates().dropna(subset=[INCOME, SCORE])
print("Shape:", df.shape)

# 2. Scaling + visual exploration
X = df[[INCOME, SCORE]]
X_scaled = StandardScaler().fit_transform(X)

plt.figure(figsize=(6, 5))
plt.scatter(df[INCOME], df[SCORE], alpha=0.7)
plt.xlabel(INCOME)
plt.ylabel(SCORE)
plt.title("Income vs spending score (raw)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exploration.png")
plt.close()

# 3. K-Means: find optimal k (elbow + silhouette)
ks = range(2, 11)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertias, marker="o")
ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(ks, sils, marker="o", color="green")
ax[1].set(title="Silhouette score", xlabel="k", ylabel="Score")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/optimal_k.png")
plt.close()

best_k = list(ks)[sils.index(max(sils))]
print(f"Best k by silhouette: {best_k} (score={max(sils):.3f})")

# Use k=5 (classic elbow for this dataset) if silhouette agrees closely, else best_k
K = 5 if abs(sils[3] - max(sils)) < 0.02 else best_k
print(f"Using k={K}")

kmeans = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X_scaled)
df["KMeans_Cluster"] = kmeans.labels_

# 4. Visualize clusters (2D)
centers = pd.DataFrame(
    StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_), columns=[INCOME, SCORE]
)
plt.figure(figsize=(7, 5))
plt.scatter(df[INCOME], df[SCORE], c=df["KMeans_Cluster"], cmap="tab10", alpha=0.8)
plt.scatter(centers[INCOME], centers[SCORE], c="black", marker="X", s=200, label="Centroids")
plt.xlabel(INCOME)
plt.ylabel(SCORE)
plt.title(f"K-Means clusters (k={K})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/kmeans_clusters.png")
plt.close()

# 5. BONUS A: DBSCAN
db = DBSCAN(eps=0.4, min_samples=5).fit(X_scaled)
df["DBSCAN_Cluster"] = db.labels_
n_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(f"\nDBSCAN: {n_db} clusters, {(db.labels_ == -1).sum()} noise points")

plt.figure(figsize=(7, 5))
plt.scatter(df[INCOME], df[SCORE], c=df["DBSCAN_Cluster"], cmap="tab10", alpha=0.8)
plt.xlabel(INCOME)
plt.ylabel(SCORE)
plt.title("DBSCAN clusters (-1 = noise)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/dbscan_clusters.png")
plt.close()

# 6. BONUS B: average spending per cluster
avg = df.groupby("KMeans_Cluster")[[INCOME, SCORE]].mean().round(2)
avg["Customers"] = df["KMeans_Cluster"].value_counts().sort_index()
print("\nAverage values per K-Means cluster:\n", avg)

plt.figure(figsize=(6, 4))
plt.bar(avg.index.astype(str), avg[SCORE], color="teal")
plt.xlabel("Cluster")
plt.ylabel("Avg spending score")
plt.title("Average spending score per cluster")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/avg_spending_per_cluster.png")
plt.close()

df.to_csv(f"{OUT_DIR}/customers_with_clusters.csv", index=False)
print(f"\nResults saved to ./{OUT_DIR}/")
