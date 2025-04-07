import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("data/Income_Survey_Dataset.csv")

# Plot histograms of income variables
income_vars = [
    "income_after_tax",
    "Total_income",
    "Salary_wages",
    "Earning",
    "Investment",
    "Self_emp_income.1",
    "Pension",
]

plt.figure(figsize=(15, 10))
for i, var in enumerate(income_vars):
    plt.subplot(3, 3, i + 1)
    plt.hist(df[var], bins=30)
    plt.title(f"Distribution of {var}")
    plt.tight_layout()
plt.savefig("output/clustering/income_histograms.png")

# Correlation matrix for income variables
plt.figure(figsize=(10, 8))
sns.heatmap(df[income_vars].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Income Variables")
plt.savefig("output/clustering/income_correlation_matrix.png")

# Select features for clustering
cluster_features = [
    "income_after_tax",
    "Total_income",
    "Investment",
    "Self_emp_income",
    "Pension",
    "Salary_wages",
]

# Handle missing values
X = df[cluster_features].fillna(0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check for outliers using boxplots
plt.figure(figsize=(12, 8))
sns.boxplot(data=pd.DataFrame(X_scaled, columns=cluster_features))
plt.title("Boxplot of Scaled Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/clustering/boxplot_scaled_features.png")
plt.close()


# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA result
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.title("PCA of Income Variables")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.savefig("output/clustering/pca_income.png")
plt.close()

# Explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Components")
plt.savefig("output/clustering/explained_variance.png")
plt.close()

# Elbow method to find optimal K
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.savefig("output/clustering/elbow_method.png")
plt.close()

# Silhouette analysis
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg}")

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.savefig("output/clustering/silhouette_analysis.png")
plt.close()

# Dendrogram to visualize hierarchical clustering
plt.figure(figsize=(12, 8))
dend = shc.dendrogram(shc.linkage(X_scaled, method="ward"))
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distance")
plt.axhline(y=6, color="r", linestyle="--")
plt.savefig("output/clustering/dendrogram.png")
plt.close()


# Apply K-Means with optimal K
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to original data
df["cluster"] = cluster_labels

# Visualize clusters in PCA space
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", alpha=0.6, s=50)
plt.colorbar(scatter, label="Cluster")
plt.title(f"K-Means Clustering (K={optimal_k}) Visualized with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

# Plot centroids in PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker="X", s=200, c="red", label="Centroids")
plt.legend()
plt.savefig("output/clustering/kmeans_clusters_pca.png")
plt.close()

# Analyze clusters
cluster_profiles = df.groupby("cluster")[cluster_features].mean()
print("Cluster Profiles (Mean Values):")
print(cluster_profiles)

# Visualize cluster profiles
plt.figure(figsize=(14, 8))
cluster_profiles.T.plot(kind="bar", ax=plt.gca())
plt.title("Mean Values of Features Across Clusters")
plt.ylabel("Mean Value")
plt.xticks(rotation=45)
plt.legend(title="Cluster")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("output/clustering/cluster_profiles.png")
plt.close()

# Boxplots for each feature by cluster
for feature in cluster_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="cluster", y=feature, data=df)
    plt.title(f"Distribution of {feature} by Cluster")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"output/clustering/boxplot_{feature}_by_cluster.png")
    plt.close()

# Analyze demographic composition of clusters
demographic_vars = ["Age_gap", "Gender", "Marital_status", "Highest_edu", "Work_ref", "Immigrant"]

for var in demographic_vars:
    plt.figure(figsize=(12, 6))
    cross_tab = pd.crosstab(df["cluster"], df[var], normalize="index") * 100
    cross_tab.plot(kind="bar", stacked=True)
    plt.title(f"Distribution of {var} within Clusters")
    plt.ylabel("Percentage (%)")
    plt.legend(title=var)
    plt.xticks(rotation=0)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"output/clustering/distribution_{var}_by_cluster.png")
    plt.close()

# Apply hierarchical clustering
hier_cluster = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
hier_labels = hier_cluster.fit_predict(X_scaled)

# Visualize hierarchical clusters in PCA space
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels, cmap="viridis", alpha=0.6, s=50)
plt.colorbar(scatter, label="Cluster")
plt.title(f"Hierarchical Clustering (K={optimal_k}) Visualized with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

# Compare K-means and Hierarchical clustering
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", alpha=0.6, s=50)
plt.title("K-Means Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels, cmap="viridis", alpha=0.6, s=50)
plt.title("Hierarchical Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.savefig("output/clustering/comparison_kmeans_hierarchical.png")
plt.close()

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Visualize DBSCAN clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap="viridis", alpha=0.6, s=50)
plt.colorbar(scatter, label="Cluster")
plt.title("DBSCAN Clustering Visualized with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.savefig("output/clustering/dbscan_clusters_pca.png")
plt.close()

# Count samples in each cluster
unique_clusters, counts = np.unique(dbscan_labels, return_counts=True)
print("DBSCAN Clusters:")
for cluster, count in zip(unique_clusters, counts):
    if cluster == -1:
        print(f"Noise points: {count}")
    else:
        print(f"Cluster {cluster}: {count} points")
