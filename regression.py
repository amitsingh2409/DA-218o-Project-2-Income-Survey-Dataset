import warnings
import matplotlib
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Set plotting style
matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_context("talk")

df = pd.read_csv("data/Income_Survey_Dataset.csv")

# Display basic information
print(f"Dataset shape: {df.shape}")
df.info()
df.head()

# Exploratory Data Analysis (EDA)
# Summary Statistics
# Basic statistics of numerical variables
numerical_summary = df.describe().T
print(numerical_summary)
# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])

# Distribution of Target Variable
plt.figure(figsize=(12, 6))
sns.histplot(df["Total_income"], kde=True)
plt.title("Distribution of Total Income")
plt.xlabel("Total Income")
plt.ylabel("Frequency")
plt.axvline(
    df["Total_income"].mean(),
    color="red",
    linestyle="--",
    label=f'Mean: {df["Total_income"].mean():.2f}',
)
plt.axvline(
    df["Total_income"].median(),
    color="green",
    linestyle="--",
    label=f'Median: {df["Total_income"].median():.2f}',
)
plt.legend()
plt.savefig("output/regression/total_income_distribution.png")
plt.close()

# Log transformation to handle skewness (if needed)
plt.figure(figsize=(12, 6))
sns.histplot(np.log1p(df["Total_income"]), kde=True)
plt.title("Distribution of Log-Transformed Total Income")
plt.xlabel("Log(Total Income + 1)")
plt.ylabel("Frequency")
plt.savefig("output/regression/log_transformed_income_distribution.png")
plt.close()

# Relationships with Potential Predictors
# Selecting numerical columns for correlation analysis
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Correlation heatmap
plt.figure(figsize=(16, 14))
corr_matrix = df[numerical_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, fmt=".2f", cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.savefig("output/regression/correlation_matrix.png")
plt.close()

# Identify the top correlated features with Total_income
top_corr = corr_matrix["Total_income"].sort_values(ascending=False)
print("Top correlations with Total_income:")
print(top_corr)


# Categorical Variable Analysis
# Function to plot categorical variables against target
def categorical_analysis(df, cat_col, target_col):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=cat_col, y=target_col, data=df)
    plt.title(f"{cat_col} vs {target_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"output/regression/{cat_col}_vs_{target_col}.png")
    plt.close()


# Analyze key categorical variables
categorical_cols = ["Province", "Gender", "Marital_status", "Highest_edu", "Work_ref", "Immigrant"]
for col in categorical_cols:
    if col in df.columns:
        categorical_analysis(df, col, "Total_income")

plt.figure(figsize=(12, 6))
sns.boxplot(x="Age_gap", y="Total_income", data=df)
plt.title("Total Income by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Total Income")
plt.tight_layout()
plt.savefig("output/regression/age_group_vs_total_income.png")
plt.close()

# Age gap and Income Relationship
plt.figure(figsize=(12, 6))
sns.boxplot(x="Age_gap", y="Total_income", data=df)
plt.title("Total Income by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Total Income")
plt.tight_layout()
plt.savefig("output/regression/age_group_vs_total_income.png")
plt.close()

# Pairwise Relationships
# Select a subset of important features for pairplot
important_features = ["Total_income", "Salary_wages", "Family_mem", "RENTM", "CFCOMP"]
if all(col in df.columns for col in important_features):
    sns.pairplot(df[important_features], height=2.5)
    plt.suptitle("Pairwise Relationships of Key Features", y=1.02)
    plt.tight_layout()
    plt.savefig("output/regression/pairwise_relationships.png")
    plt.close()

# Data Preparation
# Feature Selection
# Select top features based on correlation with target and EDA
target_correlations = df.corr()["Total_income"].abs().sort_values(ascending=False)
top_features = [
    "Salary_wages",
    "income_after_tax",
    "Self_emp_income.1",
    "Pension",
    "Investment",
    "CFCOMP",
    "Family_mem",
    "Marital_status",
    "Earning",
    "Child_benefit",
    "RENTM",
    "Age_gap",
    "Weight",
    "Private_pension",
    "Province",
]
print("Top 15 features by correlation magnitude:")
print(target_correlations[1:16])

# Select features for modeling
X = df[top_features]
y = df["Total_income"]

# Data Scaling
# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

# Convert back to DataFrame for readability
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train-Test Split
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_scaled, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Testing set: {X_test.shape}, {y_test.shape}")


# Building the PyMC Model
# Basic Linear Regression Model
def build_linear_model(X, y):
    with pm.Model() as linear_model:
        # Priors for unknown model parameters
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=X.shape[1])

        # Expected value of outcome
        mu = intercept + pm.math.dot(X, betas)

        # Likelihood (sampling distribution) of observations
        sigma = pm.HalfCauchy("sigma", beta=10)
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)

        # Sample from the posterior
        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    return linear_model, trace


# Build and sample from the model
linear_model, trace = build_linear_model(X_train.values, y_train)

# Model Evaluation
# Summarize the posterior distributions
summary = az.summary(trace)
print(summary)

# Plot posterior distributions
az.plot_posterior(trace)
plt.tight_layout()
plt.savefig("output/regression/regression_linear_model_posterior_distributions.png")
plt.close()

# Plot trace to check convergence
az.plot_trace(trace)
plt.tight_layout()
plt.savefig("output/regression/regression_linear_model_trace.png")
plt.close()


# Prediction function
def predict(trace, X):
    posterior_samples = trace.posterior["intercept"].values.flatten()
    posterior_betas = trace.posterior["betas"].values.reshape(-1, X.shape[1])

    # Get samples from posterior predictive distribution
    predictions = np.array(
        [
            sample_intercept + np.dot(X, sample_betas)
            for sample_intercept, sample_betas in zip(posterior_samples, posterior_betas)
        ]
    )

    # Return mean prediction
    return predictions.mean(axis=0)


# Make predictions
y_pred_train = predict(trace, X_train.values)
y_pred_test = predict(trace, X_test.values)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")

# Posterior Predictive Checks
# Plotting actual vs predicted values
plt.figure(figsize=(12, 10))

# Training data
plt.subplot(2, 1, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "k--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Training Data)")

# Testing data
plt.subplot(2, 1, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Testing Data)")

plt.tight_layout()
plt.savefig("output/regression/regression_linear_model_actual_vs_predicted.png")
plt.close()

# Residual analysis
y_train_residuals = y_train - y_pred_train
y_test_residuals = y_test - y_pred_test

plt.figure(figsize=(12, 10))

# Training residuals
plt.subplot(2, 2, 1)
plt.scatter(y_pred_train, y_train_residuals, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="-")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Training)")

# Testing residuals
plt.subplot(2, 2, 2)
plt.scatter(y_pred_test, y_test_residuals, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="-")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Testing)")

# Histogram of residuals
plt.subplot(2, 2, 3)
plt.hist(y_train_residuals, bins=30, alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals (Training)")

plt.subplot(2, 2, 4)
plt.hist(y_test_residuals, bins=30, alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals (Testing)")

plt.tight_layout()
plt.savefig("output/regression/regression_linear_model_residual_analysis.png")
plt.close()


# Advanced Modeling with PyMC
# Non-Linear Relationships
def build_nonlinear_model(X, y):
    with pm.Model() as nonlinear_model:
        # Priors for unknown model parameters
        intercept = pm.Normal("intercept", mu=0, sigma=10)

        # Coefficients for linear terms
        beta_linear = pm.Normal("beta_linear", mu=0, sigma=1, shape=X.shape[1])

        # Coefficients for squared terms (for non-linearity)
        beta_squared = pm.Normal("beta_squared", mu=0, sigma=0.5, shape=X.shape[1])

        # Expected value of outcome with squared terms
        mu = intercept + pm.math.dot(X, beta_linear) + pm.math.dot(X**2, beta_squared)

        # Likelihood (sampling distribution) of observations
        sigma = pm.HalfCauchy("sigma", beta=10)
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)

        # Sample from the posterior
        trace_nonlinear = pm.sample(1000, tune=1000, return_inferencedata=True)

    return nonlinear_model, trace_nonlinear


# Build non-linear model
nonlinear_model, trace_nonlinear = build_nonlinear_model(X_train.values, y_train)

# Evaluate non-linear model
az.summary(trace_nonlinear)


def predict_nonlinear(trace, X):
    # Extract posterior samples
    posterior_intercept = trace.posterior["intercept"].values.flatten()
    posterior_linear = trace.posterior["beta_linear"].values.reshape(-1, X.shape[1])
    posterior_squared = trace.posterior["beta_squared"].values.reshape(-1, X.shape[1])

    # Get predictions for each posterior sample
    predictions = np.array(
        [
            sample_intercept + np.dot(X, sample_linear) + np.dot(X**2, sample_squared)
            for sample_intercept, sample_linear, sample_squared in zip(
                posterior_intercept, posterior_linear, posterior_squared
            )
        ]
    )

    # Return mean prediction across all posterior samples
    return predictions.mean(axis=0)


# Make predictions
y_pred_train = predict_nonlinear(trace_nonlinear, X_train.values)
y_pred_test = predict_nonlinear(trace_nonlinear, X_test.values)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")


# Posterior Predictive Checks
# Plotting actual vs predicted values
plt.figure(figsize=(12, 10))

# Training data
plt.subplot(2, 1, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "k--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Training Data)")

# Testing data
plt.subplot(2, 1, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Testing Data)")

plt.tight_layout()
plt.savefig("output/regression/regression_non_linear_model_actual_vs_predicted.png")
plt.close()

# Residual analysis
y_train_residuals = y_train - y_pred_train
y_test_residuals = y_test - y_pred_test

plt.figure(figsize=(12, 10))

# Training residuals
plt.subplot(2, 2, 1)
plt.scatter(y_pred_train, y_train_residuals, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="-")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Training)")

# Testing residuals
plt.subplot(2, 2, 2)
plt.scatter(y_pred_test, y_test_residuals, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="-")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Testing)")

# Histogram of residuals
plt.subplot(2, 2, 3)
plt.hist(y_train_residuals, bins=30, alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals (Training)")

plt.subplot(2, 2, 4)
plt.hist(y_test_residuals, bins=30, alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals (Testing)")

plt.tight_layout()
plt.savefig("output/regression/regression_non_linear_model_residual_analysis.png")
plt.close()


# Robust Regression (t-distributed errors)
def build_robust_model(X, y):
    with pm.Model() as improved_robust_model:
        # More flexible priors
        intercept = pm.Normal("intercept", mu=0, sigma=20)
        beta_linear = pm.Normal("beta_linear", mu=0, sigma=2, shape=X.shape[1])
        beta_squared = pm.Normal("beta_squared", mu=0, sigma=1, shape=X.shape[1])

        # Add both linear and squared terms
        mu = intercept + pm.math.dot(X, beta_linear) + pm.math.dot(X**2, beta_squared)

        # More flexible degrees of freedom prior
        nu = pm.Gamma("nu", alpha=3, beta=0.1)
        sigma = pm.HalfNormal("sigma", sigma=10)

        # Student's t likelihood
        likelihood = pm.StudentT("likelihood", nu=nu, mu=mu, sigma=sigma, observed=y)

        # Increase number of samples and tune steps
        trace_robust = pm.sample(
            1000,
            tune=1000,
            target_accept=0.9,
            return_inferencedata=True,
        )

    return robust_model, trace_robust


def predict_robust(trace, X):
    posterior_intercept = trace.posterior["intercept"].values.flatten()
    posterior_linear = trace.posterior["beta_linear"].values.reshape(-1, X.shape[1])
    posterior_squared = trace.posterior["beta_squared"].values.reshape(-1, X.shape[1])

    predictions = np.array(
        [
            sample_intercept + np.dot(X, sample_linear) + np.dot(X**2, sample_squared)
            for sample_intercept, sample_linear, sample_squared in zip(
                posterior_intercept, posterior_linear, posterior_squared
            )
        ]
    )

    return predictions.mean(axis=0)

# Build robust model
robust_model, trace_robust = build_robust_model(X_train.values, y_train)


# Make predictions
y_pred_train = predict_robust(trace_robust, X_train.values)
y_pred_test = predict_robust(trace_robust, X_test.values)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")

# Posterior Predictive Checks
# Plotting actual vs predicted values
plt.figure(figsize=(12, 10))

# Training data
plt.subplot(2, 1, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "k--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Training Data)")

# Testing data
plt.subplot(2, 1, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Testing Data)")

plt.tight_layout()
plt.savefig("output/regression/regression_robust_model_actual_vs_predicted.png")
plt.close()

# Residual analysis
y_train_residuals = y_train - y_pred_train
y_test_residuals = y_test - y_pred_test

plt.figure(figsize=(12, 10))

# Training residuals
plt.subplot(2, 2, 1)
plt.scatter(y_pred_train, y_train_residuals, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="-")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Training)")

# Testing residuals
plt.subplot(2, 2, 2)
plt.scatter(y_pred_test, y_test_residuals, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="-")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Testing)")

# Histogram of residuals
plt.subplot(2, 2, 3)
plt.hist(y_train_residuals, bins=30, alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals (Training)")

plt.subplot(2, 2, 4)
plt.hist(y_test_residuals, bins=30, alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals (Testing)")

plt.tight_layout()
plt.savefig("output/regression/regression_robust_model_residual_analysis.png")
plt.close()

# Evaluate robust model
az.summary(trace_robust)


# Feature Importance Analysis
# Extract coefficient means
coef_means = az.summary(trace_nonlinear)["mean"].values[1 : len(top_features) + 1]

# Create feature importance DataFrame
feature_importance = pd.DataFrame({"Feature": top_features, "Coefficient": coef_means})

# Sort by absolute value
feature_importance["AbsCoef"] = np.abs(feature_importance["Coefficient"])
feature_importance = feature_importance.sort_values("AbsCoef", ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
colors = ["blue" if c > 0 else "red" for c in feature_importance["Coefficient"]]
plt.barh(feature_importance["Feature"], feature_importance["AbsCoef"], color=colors)
plt.xlabel("Absolute Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("output/regression/regression_feature_importance.png")
plt.close()
