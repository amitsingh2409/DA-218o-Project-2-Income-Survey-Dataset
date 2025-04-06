import matplotlib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import pymc as pm
import arviz as az
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

# 2. Exploratory Data Analysis (EDA)
# 2.1. Summary Statistics
# Basic statistics of numerical variables
numerical_summary = df.describe().T
print(numerical_summary)
# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])

# 2.2. Distribution of Target Variable
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
plt.savefig("output/total_income_distribution.png")
plt.close()

# Log transformation to handle skewness (if needed)
plt.figure(figsize=(12, 6))
sns.histplot(np.log1p(df["Total_income"]), kde=True)
plt.title("Distribution of Log-Transformed Total Income")
plt.xlabel("Log(Total Income + 1)")
plt.ylabel("Frequency")
plt.savefig("output/log_transformed_income_distribution.png")
plt.close()

# 2.3. Relationships with Potential Predictors
# Selecting numerical columns for correlation analysis
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Correlation heatmap
plt.figure(figsize=(16, 14))
corr_matrix = df[numerical_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, fmt=".2f", cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.savefig("output/correlation_matrix.png")
plt.close()

# Identify the top correlated features with Total_income
top_corr = corr_matrix["Total_income"].sort_values(ascending=False)
print("Top correlations with Total_income:")
print(top_corr)


# 2.4. Categorical Variable Analysis
# Function to plot categorical variables against target
def categorical_analysis(df, cat_col, target_col):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=cat_col, y=target_col, data=df)
    plt.title(f"{cat_col} vs {target_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"output/{cat_col}_vs_{target_col}.png")
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
plt.savefig("output/age_group_vs_total_income.png")
plt.close()

# 2.5. Age and Income Relationship
plt.figure(figsize=(12, 6))
sns.boxplot(x="Age_gap", y="Total_income", data=df)
plt.title("Total Income by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Total Income")
plt.tight_layout()
plt.savefig("output/age_group_vs_total_income.png")
plt.close()

# 2.6. Pairwise Relationships
# Select a subset of important features for pairplot
important_features = ["Total_income", "Age_gap", "Earning", "Work_yearly", "Total_hour_ref"]
if all(col in df.columns for col in important_features):
    sns.pairplot(df[important_features], height=2.5)
    plt.suptitle("Pairwise Relationships of Key Features", y=1.02)
    plt.tight_layout()
    plt.savefig("output/pairwise_relationships.png")
    plt.close()

# 3. Data Preparation

# 3.1. Feature Selection
# Select top features based on correlation with target
target_correlations = df.corr()["Total_income"].abs().sort_values(ascending=False)
top_features = target_correlations[1:16].index.tolist()  # Exclude target itself, take top 15

print("Top 15 features by correlation magnitude:")
print(target_correlations[1:16])

# Select features for modeling
X = df[top_features]
y = df["Total_income"]

# 3.5. Data Scaling
# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

# Convert back to DataFrame for readability
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 3.6. Train-Test Split
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_scaled, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Testing set: {X_test.shape}, {y_test.shape}")


# 4. Building the PyMC Model
# 4.1. Basic Linear Regression Model
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

# 4.2. Model Evaluation
# Summarize the posterior distributions
summary = az.summary(trace)
print(summary)

# Plot posterior distributions
az.plot_posterior(trace)
plt.tight_layout()
plt.savefig("output/posterior_distributions.png")
plt.close()

# Plot trace to check convergence
az.plot_trace(trace)
plt.tight_layout()
plt.savefig("output/trace_plot.png")
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

# 4.3. Posterior Predictive Checks
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
plt.savefig("output/actual_vs_predicted.png")
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
plt.savefig("output/residual_analysis.png")
plt.close()


# 5. Advanced Modeling with PyMC
# 5.1. Non-Linear Relationships
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


# 5.3. Robust Regression (t-distributed errors)
def build_robust_model(X, y):
    with pm.Model() as robust_model:
        # Priors for unknown model parameters
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])

        # Expected value of outcome
        mu = intercept + pm.math.dot(X, beta)

        # Use Student's t distribution for robustness against outliers
        nu = pm.Exponential("nu", lam=1 / 30)  # Degrees of freedom
        sigma = pm.HalfCauchy("sigma", beta=10)

        # Likelihood with Student's t distribution
        likelihood = pm.StudentT("likelihood", nu=nu, mu=mu, sigma=sigma, observed=y)

        # Sample from the posterior
        trace_robust = pm.sample(1000, tune=1000, return_inferencedata=True)

    return robust_model, trace_robust


# Build robust model
robust_model, trace_robust = build_robust_model(X_train.values, y_train)

# Evaluate robust model
az.summary(trace_robust)

# 6. Model Comparison
# Compare models using WAIC (Widely Applicable Information Criterion)
model_comparison = az.compare(
    {"linear": trace, "nonlinear": trace_nonlinear, "robust": trace_robust}
)

print("Model Comparison:")
print(model_comparison)

# Plot comparison
az.plot_compare(model_comparison)
plt.savefig("output/model_comparison.png")
plt.close()

# 7. Feature Importance Analysis
# Extract coefficient means
coef_means = az.summary(trace)["mean"].values[1 : len(top_features) + 1]

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
plt.savefig("output/feature_importance.png")
plt.close()
