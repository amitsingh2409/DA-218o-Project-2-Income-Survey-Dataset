import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_context("talk")

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

# Fit a range of Gaussian Mixture Models
gmm_bic = []
gmm_aic = []
n_components_range = range(1, 11)
for n_components in n_components_range:
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=42, max_iter=200
    )
    gmm.fit(X_scaled)
    gmm_bic.append(gmm.bic(X_scaled))
    gmm_aic.append(gmm.aic(X_scaled))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(n_components_range, gmm_bic, marker="o")
plt.title("BIC by Components")
plt.xlabel("Number of Components")
plt.ylabel("BIC Score")

plt.subplot(1, 2, 2)
plt.plot(n_components_range, gmm_aic, marker="o")
plt.title("AIC by Components")
plt.xlabel("Number of Components")
plt.ylabel("AIC Score")
plt.tight_layout()
plt.savefig("output/clustering/gmm_aic_bic.png")
plt.close()

# Select optimal number of components based on BIC
optimal_components = n_components_range[np.argmin(gmm_bic)]
print(f"Optimal number of components based on BIC: {optimal_components}")

# Fit the optimal GMM model
gmm = GaussianMixture(n_components=optimal_components, covariance_type="full", random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_probs = gmm.predict_proba(X_scaled)

# Add GMM cluster labels to DataFrame
df["gmm_cluster"] = gmm_labels

# Visualize GMM clusters in PCA space
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap="viridis", alpha=0.6, s=50)
plt.colorbar(scatter, label="Cluster")
plt.title(f"GMM Clustering (Components={optimal_components}) with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.savefig("output/clustering/gmm_clusters_pca.png")
plt.close()

# Visualize GMM cluster probabilities
sample_size = min(1000, len(X_scaled))
indices = np.random.choice(range(len(X_scaled)), sample_size, replace=False)

plt.figure(figsize=(14, 10))
for i in range(optimal_components):
    plt.subplot(2, (optimal_components + 1) // 2, i + 1)
    plt.scatter(
        X_pca[indices, 0],
        X_pca[indices, 1],
        c=gmm_probs[indices, i],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=30,
    )
    plt.colorbar(label="Probability")
    plt.title(f"Cluster {i} Probability")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("output/clustering/gmm_cluster_probabilities.png")
plt.close()

# Fit a Bayesian Gaussian Mixture Model with Dirichlet process prior
vbgmm = BayesianGaussianMixture(
    n_components=10,  # Upper bound on number of components
    weight_concentration_prior=1 / 10,
    weight_concentration_prior_type="dirichlet_process",
    covariance_type="full",
    random_state=42,
    max_iter=200,
)
vbgmm.fit(X_scaled)
vbgmm_labels = vbgmm.predict(X_scaled)

# Get effective number of components
effective_components = np.sum(vbgmm.weights_ > 0.01)
print(f"Effective number of components in VBGMM: {effective_components}")
print("Component weights:", vbgmm.weights_)

# Add VBGMM cluster labels to DataFrame
df["vbgmm_cluster"] = vbgmm_labels

# Visualize VBGMM clusters in PCA space
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=vbgmm_labels, cmap="viridis", alpha=0.6, s=50)
plt.colorbar(scatter, label="Cluster")
plt.title("Variational Bayesian GMM Clustering with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.savefig("output/clustering/vbgmm_clusters_pca.png")
plt.close()
