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

    # Compute linear combinations for all non-reference classes
    activation = pm.math.dot(X_train, beta) + alpha

    # Create reference class column (zeros)
    zeros = np.zeros((X_train.shape[0], 1))

    # Concatenate zeros and activation for full probability matrix
    logits = pm.math.concatenate([zeros, activation], axis=1)

    # Apply softmax to get probabilities
    p = pm.math.softmax(logits, axis=1)

    # Categorical likelihood
    observed = pm.Categorical("observed", p=p, observed=y_train)

    # More conservative sampling to help with initialization
    trace = pm.sample(
        draws=500,
        tune=1000,
        chains=2,
        target_accept=0.9,
        init="adapt_diag",
        return_inferencedata=True,
        cores=1,
    )

# Assess model convergence and diagnostics
az.plot_trace(trace)
plt.tight_layout()
plt.savefig("output/classification/trace_plot.png")
plt.close()

# Summary of parameter estimates
summary = az.summary(trace, round_to=2)
print("\nParameter Estimates Summary:")
print(summary)

# Plot parameter distributions
az.plot_forest(trace, var_names=["alpha", "beta"], combined=True)
plt.tight_layout()
plt.savefig("output/classification/parameter_forest_plot.png")
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
class_names = [str(x) for x in list(marital_status_mapping.keys())]
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
