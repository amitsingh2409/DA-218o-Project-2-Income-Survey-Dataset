import matplotlib
import pandas as pd
import seaborn as sns
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
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
plt.savefig("output/classification/marital_status_distribution.png")
plt.close()

# Age distribution by marital status
plt.figure(figsize=(14, 8))
ax = sns.boxplot(x="Marital_status", y="Age_gap", data=df, palette="coolwarm")
ax.set_title("Age Distribution by Marital Status", fontsize=16)
ax.set_xlabel("Marital Status", fontsize=14)
ax.set_ylabel("Age Category", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/classification/age_distribution_by_marital_status.png")
plt.close()

# Income distribution by marital status
plt.figure(figsize=(14, 8))
ax = sns.boxplot(x="Marital_status", y="income_after_tax", data=df, palette="coolwarm")
ax.set_title("Income Distribution by Marital Status", fontsize=16)
ax.set_xlabel("Marital Status", fontsize=14)
ax.set_ylabel("Income After Tax", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/classification/income_distribution_by_marital_status.png")
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
plt.savefig("output/classification/gender_distribution_by_marital_status.png")
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
plt.savefig("output/classification/education_level_by_marital_status.png")
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
plt.savefig("output/classification/correlation_heatmap.png")
plt.close()

# Create target variable
marital_status_mapping = {status: i for i, status in enumerate(df["Marital_status"].unique())}
df["marital_status_code"] = df["Marital_status"].map(marital_status_mapping)
unique_classes = len(marital_status_mapping)

# Select relevant features based on EDA
features = [
    "Age_gap",
    "Family_mem",
    "CFCOMP",
    "Work_ref",
    "Private_pension",
    "CPP_QPP",
    "Province",
    "MBMREGP",
    "income_after_tax",
    "Immigrant",
    "Guaranteed_income",
    "Weight",
    "Child_benefit",
    "Total_income",
    "RENTM",
]

# Prepare features
X = df[features].copy()
y = df["marital_status_code"].values

# Scale numerical features
scaler = StandardScaler()
X_processed = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Number of unique classes: {unique_classes}")


# Get dimensions
n_train, n_features = X_train.shape
n_classes = unique_classes

# Define the PyMC model
with pm.Model() as multinomial_model:
    # Simpler, more robust priors
    alpha = pm.Normal("alpha", mu=0, sigma=2, shape=(n_classes - 1))
    beta = pm.Normal("beta", mu=0, sigma=1, shape=(n_features, n_classes - 1))
    activation = pm.math.dot(X_train, beta) + alpha

    zeros = np.zeros((X_train.shape[0], 1))

    logits = pm.math.concatenate([zeros, activation], axis=1)

    p = pm.math.softmax(logits, axis=1)

    observed = pm.Categorical("observed", p=p, observed=y_train)

    trace = pm.sample(
        draws=500,
        tune=1000,
        chains=2,
        target_accept=0.9,
        init="adapt_diag",
        return_inferencedata=True,
        cores=1,
    )

az.plot_trace(trace)
plt.tight_layout()
plt.savefig("output/classification/trace_plot.png")
plt.close()

summary = az.summary(trace, round_to=2)
print("\nParameter Estimates Summary:")
print(summary)

az.plot_forest(trace, var_names=["alpha", "beta"], combined=True)
plt.tight_layout()
plt.savefig("output/classification/parameter_forest_plot.png")
plt.close()


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


y_pred, probabilities = predict_class(X_test, trace)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
class_names = [str(x) for x in list(marital_status_mapping.keys())]
class_report = classification_report(y_test, y_pred, target_names=class_names)
print(class_report)

plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("output/classification/confusion_matrix.png")
plt.close()

"""
Classification Report:
              precision    recall  f1-score   support

           3       0.66      0.82      0.73      1564
           1       0.77      0.90      0.83      5936
           4       0.60      0.67      0.63      2661
          96       1.00      1.00      1.00      2566
           2       0.21      0.00      0.00      1313
          99       0.23      0.02      0.04       489

    accuracy                           0.76     14529
   macro avg       0.58      0.57      0.54     14529
weighted avg       0.70      0.76      0.71     14529
"""

# Define the PyMC model with informative priors
with pm.Model() as informative_model:
    # Informative priors for intercepts
    alpha = pm.Normal("alpha", mu=[0, 0.5, -1, -0.5, -1], sigma=1, shape=(n_classes - 1))

    beta_mus = np.zeros((n_features, n_classes - 1))

    age_gap_idx = features.index("Age_gap")
    family_mem_idx = features.index("Family_mem")
    income_idx = features.index("income_after_tax")
    total_income_idx = features.index("Total_income")
    child_benefit_idx = features.index("Child_benefit")

    beta_mus[age_gap_idx, 0] = 0.8  # Older -> more likely married
    beta_mus[age_gap_idx, 2] = 0.5  # Older -> more likely widowed
    beta_mus[age_gap_idx, 4] = -0.5  # Younger -> more likely separated

    beta_mus[family_mem_idx, 0] = 0.7  # More family members -> married
    beta_mus[family_mem_idx, 3] = -0.4  # Fewer family members -> divorced

    # Prior means for income after tax
    beta_mus[income_idx, 0] = 0.5  # Higher income -> more likely married
    beta_mus[income_idx, 2] = -0.3  # Lower income -> more likely widowed

    # Child benefit more associated with certain marital statuses
    beta_mus[child_benefit_idx, 0] = 0.4  # Child benefit -> married
    beta_mus[child_benefit_idx, 4] = 0.3  # Child benefit -> separated

    beta = pm.Normal("beta", mu=beta_mus, sigma=0.5, shape=(n_features, n_classes - 1))

    activation = pm.math.dot(X_train, beta) + alpha

    zeros = np.zeros((X_train.shape[0], 1))

    logits = pm.math.concatenate([zeros, activation], axis=1)

    p = pm.math.softmax(logits, axis=1)

    observed = pm.Categorical("observed", p=p, observed=y_train)

    trace = pm.sample(
        draws=500,
        tune=1500,
        chains=2,
        target_accept=0.9,
        init="adapt_diag",
        return_inferencedata=True,
        cores=1,
    )


az.plot_trace(trace)
plt.tight_layout()
plt.savefig("output/classification/informative_trace_plot.png")
plt.close()

summary = az.summary(trace, round_to=2)
print("\nParameter Estimates Summary:")
print(summary)

az.plot_forest(trace, var_names=["alpha", "beta"], combined=True)
plt.tight_layout()
plt.savefig("output/classification/informative_parameter_forest_plot.png")
plt.close()


def predict_class_with_uncertainty(X_data, trace, samples=100):
    alpha_samples = trace.posterior["alpha"].values
    beta_samples = trace.posterior["beta"].values

    n_chains = alpha_samples.shape[0]
    n_draws = alpha_samples.shape[1]

    chain_indices = np.random.randint(0, n_chains, size=samples)
    draw_indices = np.random.randint(0, n_draws, size=samples)

    all_probs = np.zeros((X_data.shape[0], trace.posterior["alpha"].shape[2] + 1, samples))

    for i in range(samples):
        chain_idx = chain_indices[i]
        draw_idx = draw_indices[i]

        alpha_sample = alpha_samples[chain_idx, draw_idx]
        beta_sample = beta_samples[chain_idx, draw_idx]

        # Calculate activation
        activation = np.dot(X_data, beta_sample) + alpha_sample

        # Add zeros for reference class
        zero_vec = np.zeros((X_data.shape[0], 1))
        all_activation = np.concatenate([zero_vec, activation.reshape(X_data.shape[0], -1)], axis=1)

        # Apply softmax
        all_probs[:, :, i] = np.exp(all_activation) / np.sum(
            np.exp(all_activation), axis=1, keepdims=True
        )

    mean_probs = all_probs.mean(axis=2)

    predicted_classes = np.argmax(mean_probs, axis=1)

    epsilon = 1e-10  # To avoid log(0)
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=1)
    max_entropy = -np.log(1 / mean_probs.shape[1])  # Maximum possible entropy
    uncertainty = entropy / max_entropy  # Normalized uncertainty [0,1]

    # Calculate 95% credible intervals for each class probability
    credible_intervals = {}
    for class_idx in range(all_probs.shape[1]):
        lower_bound = np.percentile(all_probs[:, class_idx, :], 2.5, axis=1)
        upper_bound = np.percentile(all_probs[:, class_idx, :], 97.5, axis=1)
        credible_intervals[f"class_{class_idx}"] = (lower_bound, upper_bound)

    return predicted_classes, mean_probs, uncertainty, credible_intervals


y_pred, probs, _, _ = predict_class_with_uncertainty(X_test, trace)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
class_names = [str(x) for x in list(marital_status_mapping.keys())]
class_report = classification_report(y_test, y_pred, target_names=class_names)
print(class_report)

plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("output/classification/informative_confusion_matrix.png")
plt.close()


"""
Classification Report:
              precision    recall  f1-score   support

           3       0.66      0.82      0.73      1564
           1       0.77      0.90      0.83      5936
           4       0.60      0.67      0.64      2661
          96       1.00      1.00      1.00      2566
           2       0.20      0.00      0.00      1313
          99       0.23      0.02      0.04       489

    accuracy                           0.76     14529
   macro avg       0.58      0.57      0.54     14529
weighted avg       0.70      0.76      0.71     14529
"""
