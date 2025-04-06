import matplotlib
import pandas as pd
import seaborn as sns
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set plotting style
matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_context("talk")

df = pd.read_csv("data/Income_Survey_Dataset.csv")

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nData types and missing values:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check unique values in marital_status (our target variable)
print("\nUnique values in marital_status:")
print(df["Marital_status"].value_counts())

# Plot marital status distribution
plt.figure(figsize=(12, 6))
ax = sns.countplot(x="Marital_status", data=df, palette="viridis")
ax.set_title("Distribution of Marital Status", fontsize=16)
ax.set_xlabel("Marital Status", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
plt.xticks(rotation=45)

# Add percentage labels
total = len(df)
for p in ax.patches:
    percentage = f"{100 * p.get_height() / total:.1f}%"
    ax.annotate(
        percentage,
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        fontsize=12,
        color="black",
        xytext=(0, 10),
        textcoords="offset points",
    )
plt.tight_layout()
plt.savefig("output/marital_status_distribution.png")
plt.close()

# Age distribution by marital status
plt.figure(figsize=(14, 8))
ax = sns.boxplot(x="Marital_status", y="Age_gap", data=df, palette="coolwarm")
ax.set_title("Age Distribution by Marital Status", fontsize=16)
ax.set_xlabel("Marital Status", fontsize=14)
ax.set_ylabel("Age Category", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/age_distribution_by_marital_status.png")
plt.close()

# Income distribution by marital status
plt.figure(figsize=(14, 8))
ax = sns.boxplot(x="Marital_status", y="income_after_tax", data=df, palette="coolwarm")
ax.set_title("Income Distribution by Marital Status", fontsize=16)
ax.set_xlabel("Marital Status", fontsize=14)
ax.set_ylabel("Income After Tax", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/income_distribution_by_marital_status.png")
plt.close()

# Gender distribution by marital status
plt.figure(figsize=(12, 8))
gender_marital = pd.crosstab(df["Marital_status"], df["Gender"], normalize="index") * 100
gender_marital.plot(kind="bar", stacked=True, colormap="viridis")
plt.title("Gender Distribution by Marital Status", fontsize=16)
plt.xlabel("Marital Status", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="Gender")
plt.tight_layout()
plt.savefig("output/gender_distribution_by_marital_status.png")
plt.close()

# Education level by marital status
plt.figure(figsize=(16, 10))
edu_marital = pd.crosstab(df["Marital_status"], df["Highest_edu"], normalize="index") * 100
edu_marital.plot(kind="bar", stacked=True, colormap="viridis")
plt.title("Education Level by Marital Status", fontsize=16)
plt.xlabel("Marital Status", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="Education Level", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("output/education_level_by_marital_status.png")
plt.close()

# Correlation heatmap of numeric variables
plt.figure(figsize=(16, 14))
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
correlation_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Heatmap of Numeric Variables", fontsize=16)
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png")
plt.close()

# Create target variable
marital_status_mapping = {status: i for i, status in enumerate(df["Marital_status"].unique())}
df["marital_status_code"] = df["Marital_status"].map(marital_status_mapping)
unique_classes = len(marital_status_mapping)

# Select relevant features based on EDA
features = [
    "Highschool",
    "Highest_edu",
    "Work_ref",
    "Work_yearly",
    "Total_income",
    "income_after_tax",
    "Immigrant",
    "Family_mem",
    "Province",
    "Earning"
]

# Prepare features
X = df[features].copy()
y = df["marital_status_code"].values

# Scale numerical features
scaler = StandardScaler()
X_processed = scaler.fit_transform(X[features])

y_scaler = StandardScaler()
y_processed = y_scaler.fit_transform(y.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed, test_size=0.2, random_state=42, stratify=y
)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Number of unique classes: {unique_classes}")


# Get dimensions
n_train, n_features = X_train.shape
n_classes = unique_classes

# Define the PyMC model
with pm.Model() as multinomial_model:
    # Define priors for weights and intercept
    sigma_prior = pm.HalfCauchy("sigma_prior", beta=2.5)
    
    # Coefficients for each feature and class
    beta = pm.Normal("beta", 
                    mu=0, 
                    sigma=sigma_prior, 
                    shape=(n_features, n_classes - 1))
    
    # Intercepts for each class
    alpha = pm.Normal("alpha", 
                     mu=0, 
                     sigma=10, 
                     shape=n_classes - 1)
    
    # Model specification using softmax link function
    activation = pm.math.dot(X_train, beta) + alpha
    
    # Add reference class (first class with logits = 0)
    zero_vec = np.zeros((X_train.shape[0], 1))
    all_activation = pm.math.concatenate([zero_vec, activation], axis=1)
    
    # Apply softmax to get probabilities
    p = pm.math.softmax(all_activation)
    
    # Define the likelihood
    observed = pm.Categorical("observed", p=p, observed=y_train)
    
    # Sample from the posterior
    trace = pm.sample(
        1000,
        tune=1000,
        target_accept=0.95,
        return_inferencedata=True,
        cores=4
    )
    
    # Sample from the posterior predictive distribution
    ppc = pm.sample_posterior_predictive(trace)


# Assess model convergence and diagnostics
az.plot_trace(trace)
plt.tight_layout()
plt.savefig('trace_plot.png')
plt.close()

# Summary of parameter estimates
summary = az.summary(trace, round_to=2)
print("\nParameter Estimates Summary:")
print(summary)

# Plot parameter distributions
az.plot_forest(trace, var_names=['alpha', 'beta'], combined=True)
plt.tight_layout()
plt.savefig('parameter_forest_plot.png')
plt.close()

# Function to predict class using the trained model
def predict_class(X_data, trace):
    # Extract posterior samples
    alpha_samples = trace.posterior['alpha'].values
    beta_samples = trace.posterior['beta'].values
    
    # Calculate average over all posterior samples
    alpha_mean = alpha_samples.mean(axis=(0, 1))
    beta_mean = beta_samples.mean(axis=(0, 1))
    
    # Calculate activation
    activation = np.dot(X_data, beta_mean) + alpha_mean
    
    # Add zeros for reference class
    zero_vec = np.zeros((X_data.shape[0], 1))
    all_activation = np.concatenate([zero_vec, activation.reshape(X_data.shape[0], -1)], axis=1)
    
    # Apply softmax
    prob = np.exp(all_activation) / np.sum(np.exp(all_activation), axis=1, keepdims=True)
    
    # Return class with highest probability
    return np.argmax(prob, axis=1), prob

# Make predictions
y_pred, probabilities = predict_class(X_test, trace)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
class_names = list(marital_status_mapping.keys())
class_report = classification_report(y_test, y_pred, target_names=class_names)
print(class_report)

# Confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Plot ROC curves for each class (One-vs-Rest approach)
from sklearn.metrics import roc_curve, auc
from itertools import cycle

plt.figure(figsize=(12, 10))
colors = cycle(['blue', 'red', 'green', 'yellow', 'purple'])

# Convert true labels to one-hot encoding for ROC calculation
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=range(n_classes))

for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'ROC curve of class {class_names[i]} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.show()

# Get dimensions
n_train, n_features = X_train.shape
n_classes = unique_classes

# Define the PyMC model
with pm.Model() as multinomial_model:
    # Define priors for weights and intercept
    # Using hierarchical priors for better regularization
    sigma_prior = pm.HalfCauchy("sigma_prior", beta=2.5)

    # Coefficients for each feature and class
    beta = pm.Normal("beta", mu=0, sigma=sigma_prior, shape=(n_features, n_classes - 1))

    # Intercepts for each class
    alpha = pm.Normal("alpha", mu=0, sigma=10, shape=n_classes - 1)

    # Model specification (softmax activation for multinomial log regression)
    # Calculate the dot product of features and weights for each class
    activation = at.dot(X_train, beta) + alpha

    # The first class is treated as reference class (logits = 0)
    zero_vec = at.zeros((X_train.shape[0], 1))
    all_activation = at.concatenate([zero_vec, activation], axis=1)

    # Apply softmax to get probabilities
    p = at.softmax(all_activation, axis=1)

    # Define the likelihood
    observed = pm.Categorical("observed", p=p, observed=y_train)

    # Sample from the posterior
    trace = pm.sample(
        1000,  # Number of samples
        tune=1000,  # Number of tuning steps
        target_accept=0.95,  # Target acceptance rate
        return_inferencedata=True,
        cores=4,  # Use 4 cores for parallel sampling
    )

    # Sample from the posterior predictive distribution
    ppc = pm.sample_posterior_predictive(trace)


# Assess model convergence and diagnostics
az.plot_trace(trace)
plt.tight_layout()
plt.savefig("trace_plot.png")
plt.close()

# Summary of parameter estimates
summary = az.summary(trace, round_to=2)
print("\nParameter Estimates Summary:")
print(summary)

# Plot parameter distributions
az.plot_forest(trace, var_names=["alpha", "beta"], combined=True)
plt.tight_layout()
plt.savefig("parameter_forest_plot.png")
plt.close()


# Function to predict class using the trained model
def predict_class(X_data, trace):
    # Extract posterior samples
    alpha_samples = trace.posterior["alpha"].values
    beta_samples = trace.posterior["beta"].values

    # Calculate average over all posterior samples
    alpha_mean = alpha_samples.mean(axis=(0, 1))
    beta_mean = beta_samples.mean(axis=(0, 1))

    # Calculate activation
    activation = np.dot(X_data, beta_mean) + alpha_mean

    # Add zeros for reference class
    zero_vec = np.zeros((X_data.shape[0], 1))
    all_activation = np.concatenate([zero_vec, activation.reshape(X_data.shape[0], -1)], axis=1)

    # Apply softmax
    prob = np.exp(all_activation) / np.sum(np.exp(all_activation), axis=1, keepdims=True)

    # Return class with highest probability
    return np.argmax(prob, axis=1), prob


# Make predictions
y_pred, probabilities = predict_class(X_test, trace)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
class_names = list(marital_status_mapping.keys())
class_report = classification_report(y_test, y_pred, target_names=class_names)
print(class_report)

# Confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Plot ROC curves for each class (One-vs-Rest approach)
from sklearn.metrics import roc_curve, auc
from itertools import cycle

plt.figure(figsize=(12, 10))
colors = cycle(["blue", "red", "green", "yellow", "purple"])

# Convert true labels to one-hot encoding for ROC calculation
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=range(n_classes))

for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr,
        tpr,
        color=color,
        lw=2,
        label=f"ROC curve of class {class_names[i]} (area = {roc_auc:.2f})",
    )

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-class ROC")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.show()


# Check probability calibration
from sklearn.calibration import calibration_curve

plt.figure(figsize=(14, 10))
for i in range(n_classes):
    # Calculate calibration curve for each class
    prob_true, prob_pred = calibration_curve(
        y_test_bin[:, i], probabilities[:, i], n_bins=10, strategy="uniform"
    )

    # Plot calibration curve
    plt.plot(prob_pred, prob_true, marker="o", linewidth=1, label=f"Class {class_names[i]}")

# Plot perfectly calibrated line
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Plot (Reliability Diagram)")
plt.legend()
plt.tight_layout()
plt.savefig("calibration_plot.png")
plt.show()


# Perform posterior predictive checks
with multinomial_model:
    # Use a subset of the test data for posterior predictive checks
    ppc_test = pm.sample_posterior_predictive(
        trace, var_names=["observed"], predictions=True, extend_inferencedata=False
    )

# Plot posterior predictive distribution vs. observed data
fig, ax = plt.subplots(figsize=(12, 8))
az.plot_ppc(ppc, group="posterior", ax=ax)
plt.title("Posterior Predictive Check")
plt.tight_layout()
plt.savefig("posterior_predictive_check.png")
plt.show()


# Create 2D visualization of decision boundaries (using PCA for dimensionality reduction)
from sklearn.decomposition import PCA

# Apply PCA to reduce features to 2D
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)


# Create a mesh grid for decision boundary plotting
def make_meshgrid(x, y, h=0.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


# Create a 2D PyMC model for visualization
with pm.Model() as viz_model:
    # Define priors for weights and intercept
    beta_viz = pm.Normal("beta_viz", mu=0, sigma=1, shape=(2, n_classes - 1))
    alpha_viz = pm.Normal("alpha_viz", mu=0, sigma=10, shape=n_classes - 1)

    # Model specification
    activation = at.dot(X_train_2d, beta_viz) + alpha_viz
    zero_vec = at.zeros((X_train_2d.shape[0], 1))
    all_activation = at.concatenate([zero_vec, activation], axis=1)
    p = at.softmax(all_activation, axis=1)

    # Define the likelihood
    observed = pm.Categorical("observed", p=p, observed=y_train)

    # Sample from the posterior
    trace_viz = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True, cores=4)


# Function to predict class probabilities on 2D data
def predict_proba_2d(X_data, trace):
    beta_samples = trace.posterior["beta_viz"].values
    alpha_samples = trace.posterior["alpha_viz"].values

    beta_mean = beta_samples.mean(axis=(0, 1))
    alpha_mean = alpha_samples.mean(axis=(0, 1))

    activation = np.dot(X_data, beta_mean) + alpha_mean
    zero_vec = np.zeros((X_data.shape[0], 1))
    all_activation = np.concatenate([zero_vec, activation.reshape(X_data.shape[0], -1)], axis=1)

    prob = np.exp(all_activation) / np.sum(np.exp(all_activation), axis=1, keepdims=True)
    return prob


# Plot decision boundaries
xx, yy = make_meshgrid(X_train_2d[:, 0], X_train_2d[:, 1])
grid = np.c_[xx.ravel(), yy.ravel()]
probs = predict_proba_2d(grid, trace_viz)
Z = np.argmax(probs, axis=1).reshape(xx.shape)

plt.figure(figsize=(12, 10))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.viridis)

# Plot training points
for i, class_name in enumerate(class_names):
    idx = np.where(y_train == i)
    plt.scatter(X_train_2d[idx, 0], X_train_2d[idx, 1], c=f"C{i}", label=class_name, edgecolors="k")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Decision Boundaries (PCA 2D Projection)")
plt.legend()
plt.tight_layout()
plt.savefig("decision_boundaries.png")
plt.show()

# Plot probability contours for each class
plt.figure(figsize=(16, 12))
for i, class_name in enumerate(class_names):
    plt.subplot(2, (n_classes + 1) // 2, i + 1)

    # Get probabilities for class i
    Z_prob = probs[:, i].reshape(xx.shape)

    # Plot contour
    plt.contourf(xx, yy, Z_prob, alpha=0.8, cmap="viridis")
    plt.colorbar()

    # Plot points
    for j, other_class in enumerate(class_names):
        idx = np.where(y_train == j)
        plt.scatter(
            X_train_2d[idx, 0],
            X_train_2d[idx, 1],
            c=f"C{j}",
            label=other_class if i == 0 else None,
            alpha=0.5,
            edgecolors="k",
        )

    plt.title(f"Probability for Class: {class_name}")
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig("probability_contours.png")
plt.show()


# Analyze prediction uncertainty
# Get multiple predictions from different posterior samples
n_samples = 100
sample_indices = np.random.choice(
    trace.posterior.draw.size * trace.posterior.chain.size, size=n_samples, replace=False
)


# Get predictions for each sample
def predict_from_sample(X_data, trace, sample_idx):
    chain_idx = sample_idx // trace.posterior.draw.size
    draw_idx = sample_idx % trace.posterior.draw.size

    alpha_sample = trace.posterior["alpha"].values[chain_idx, draw_idx]
    beta_sample = trace.posterior["beta"].values[chain_idx, draw_idx]

    activation = np.dot(X_data, beta_sample) + alpha_sample
    zero_vec = np.zeros((X_data.shape[0], 1))
    all_activation = np.concatenate([zero_vec, activation.reshape(X_data.shape[0], -1)], axis=1)

    prob = np.exp(all_activation) / np.sum(np.exp(all_activation), axis=1, keepdims=True)
    return np.argmax(prob, axis=1), prob


# Collect predictions
all_preds = []
all_probs = []
for idx in sample_indices:
    pred, prob = predict_from_sample(X_test, trace, idx)
    all_preds.append(pred)
    all_probs.append(prob)

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Calculate uncertainty metrics
prediction_entropy = -np.sum(
    all_probs.mean(axis=0) * np.log(all_probs.mean(axis=0) + 1e-10), axis=1
)
prediction_std = np.std(all_preds, axis=0)

# Plot uncertainty histogram
plt.figure(figsize=(12, 8))
plt.hist(prediction_entropy, bins=30, alpha=0.7, color="blue")
plt.axvline(
    x=np.median(prediction_entropy),
    color="red",
    linestyle="--",
    label=f"Median: {np.median(prediction_entropy):.2f}",
)
plt.title("Prediction Entropy (Uncertainty) Distribution")
plt.xlabel("Entropy")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("uncertainty_histogram.png")
plt.show()

# Plot predictions with highest uncertainty
most_uncertain = np.argsort(prediction_entropy)[-10:]
most_certain = np.argsort(prediction_entropy)[:10]

plt.figure(figsize=(15, 12))
for i, idx in enumerate(most_uncertain):
    plt.subplot(2, 5, i + 1)

    # Get class probabilities for this example
    class_probs = all_probs[:, idx, :]

    # Plot histogram of predictions
    for j in range(n_classes):
        sns.kdeplot(class_probs[:, j], label=class_names[j])

    plt.title(f"High Uncertainty Example {i+1}\nTrue: {class_names[y_test[idx]]}")
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig("high_uncertainty_examples.png")
plt.show()

# Analyze prediction errors in relation to uncertainty
correct = y_pred == y_test
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=prediction_entropy,
    y=prediction_std,
    hue=correct,
    palette={True: "green", False: "red"},
    alpha=0.6,
)
plt.title("Prediction Uncertainty vs. Correctness")
plt.xlabel("Entropy")
plt.ylabel("Std Dev of Predictions")
plt.legend(title="Correct Prediction")
plt.tight_layout()
plt.savefig("uncertainty_vs_correctness.png")
plt.show()


# Compare with other models (optional)


# Train a frequentist logistic regression
logreg = LogisticRegression(max_iter=1000, multi_class="multinomial")
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
logreg_acc = accuracy_score(y_test, logreg_pred)

# Train a random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# Compare accuracies
models = ["Bayesian Multinomial", "Logistic Regression", "Random Forest"]
accuracies = [accuracy, logreg_acc, rf_acc]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

# Add accuracy values on top of bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc:.4f}", ha="center")

plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()
