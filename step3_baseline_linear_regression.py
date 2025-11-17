"""
STEP 3: BASELINE LINEAR REGRESSION MODELS (TASK A)
====================================================

This script trains simple linear regression models on all feature configurations:
1. Loads all 16 feature sets (4 smoothing levels × 4 lag configurations)
2. Trains LinearRegression model for each configuration
3. Computes metrics: RMSE, MAE, R² for training and validation sets
4. Identifies best configuration based on validation performance
5. Displays model coefficients and feature importance
6. Creates comprehensive visualizations

This addresses TASK A of the assignment:
"Use a linear regression model to find the relationship between past
closing values (delay characteristics) and the target (next closing price).
Choose how far in time you want to go -- do more testing. Be careful not
to raise the number of parameters too much. Indicate the parameters of the
model you calculated. Report the appropriate error metrics for the training
set and validation set."

Author: Statistical Methods of Machine Learning - Task 1
"""

import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_feature_set(features_path):
    """
    Loads a saved feature set from disk.

    Args:
        features_path (str): Path to .npz file containing features

    Returns:
        dict: Dictionary with X_train, X_val, y_train, y_val, feature_names

    Notes:
        - Features are already scaled (StandardScaler was applied)
        - Train/validation split is chronological (pre-2025 vs 2025)
    """
    data = np.load(features_path, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "y_train": data["y_train"],
        "y_val": data["y_val"],
        "feature_names": data["feature_names"],
    }


def compute_metrics(y_true, y_pred):
    """
    Computes regression metrics.

    Args:
        y_true (np.array): True target values
        y_pred (np.array): Predicted target values

    Returns:
        dict: Dictionary with RMSE, MAE, R² metrics

    Metric Explanations:

    1. RMSE (Root Mean Squared Error):
       - Square root of average squared differences
       - Formula: √(Σ(y_true - y_pred)² / n)
       - Units: Same as target (dollars)
       - Penalizes large errors more than small ones
       - Lower is better

    2. MAE (Mean Absolute Error):
       - Average absolute differences
       - Formula: Σ|y_true - y_pred| / n
       - Units: Same as target (dollars)
       - More robust to outliers than RMSE
       - Easier to interpret: "on average, predictions are off by $X"
       - Lower is better

    3. R² (Coefficient of Determination):
       - Proportion of variance explained by the model
       - Formula: 1 - (SS_res / SS_tot)
       - Range: (-∞, 1], typically [0, 1]
       - R² = 1: Perfect predictions
       - R² = 0: Model no better than predicting mean
       - R² < 0: Model worse than predicting mean
       - Higher is better

    Why All Three Metrics:
        - RMSE shows overall prediction error with emphasis on large mistakes
        - MAE shows typical prediction error
        - R² shows how well model captures variance (scale-independent)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def train_linear_regression(X_train, y_train, X_val, y_val):
    """
    Trains a simple linear regression model.

    Args:
        X_train: Training features (scaled)
        y_train: Training targets
        X_val: Validation features (scaled)
        y_val: Validation targets

    Returns:
        tuple: (model, train_metrics, val_metrics, predictions)

    Linear Regression Model:
        y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

        where:
        - y: Predicted close price
        - β₀: Intercept (bias term)
        - β₁, β₂, ..., βₙ: Coefficients for each feature
        - x₁, x₂, ..., xₙ: Feature values (lagged close prices and volumes)

    Optimization:
        Minimizes: Σ(y_true - y_pred)²
        This is called Ordinary Least Squares (OLS)
        Has closed-form solution: β = (X'X)⁻¹X'y

    No Regularization:
        Basic linear regression has no penalty term
        Can overfit if too many features relative to samples
        Coefficients can become very large
    """
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Compute metrics
    train_metrics = compute_metrics(y_train, y_train_pred)
    val_metrics = compute_metrics(y_val, y_val_pred)

    predictions = {"train": y_train_pred, "val": y_val_pred}

    return model, train_metrics, val_metrics, predictions


def extract_config_from_filename(filename):
    """
    Extracts smoothing type and n_lags from feature filename.

    Args:
        filename (str): e.g., "features_raw_6lags.npz"

    Returns:
        tuple: (smoothing_type, n_lags)

    Examples:
        "features_raw_3lags.npz" → ("raw", 3)
        "features_sigma1_6lags.npz" → ("sigma1", 6)
    """
    basename = os.path.basename(filename)
    # Remove "features_" prefix and ".npz" suffix
    config_str = basename.replace("features_", "").replace(".npz", "")

    # Split by '_' to separate smoothing and lags
    parts = config_str.split("_")

    if len(parts) == 2:
        smoothing_type = parts[0]
        n_lags_str = parts[1].replace("lags", "")
        n_lags = int(n_lags_str)
        return smoothing_type, n_lags
    else:
        return None, None


def train_all_configurations(features_dir="features"):
    """
    Trains linear regression models on all available feature sets.

    Args:
        features_dir (str): Directory containing feature files

    Returns:
        pd.DataFrame: Results for all configurations

    Strategy:
        - Test all combinations of preprocessing and lag windows
        - Track performance on both training and validation sets
        - Identify overfitting: large gap between train and validation error
        - Select best model based on validation performance (not training)
    """
    # Find all feature files
    feature_files = glob.glob(os.path.join(features_dir, "features_*.npz"))

    if len(feature_files) == 0:
        raise FileNotFoundError(f"No feature files found in {features_dir}/")

    print(f"Found {len(feature_files)} feature configurations")
    print()

    results = []

    for features_path in sorted(feature_files):
        # Extract configuration
        smoothing_type, n_lags = extract_config_from_filename(features_path)

        if smoothing_type is None:
            continue

        print(f"Training: smoothing={smoothing_type}, n_lags={n_lags}")

        # Load data
        data = load_feature_set(features_path)

        # Train model
        model, train_metrics, val_metrics, predictions = train_linear_regression(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"]
        )

        # Store results
        result = {
            "smoothing": smoothing_type,
            "n_lags": n_lags,
            "n_features": data["X_train"].shape[1],
            "n_train_samples": len(data["y_train"]),
            "n_val_samples": len(data["y_val"]),
            "train_rmse": train_metrics["RMSE"],
            "train_mae": train_metrics["MAE"],
            "train_r2": train_metrics["R2"],
            "val_rmse": val_metrics["RMSE"],
            "val_mae": val_metrics["MAE"],
            "val_r2": val_metrics["R2"],
            "model": model,
            "feature_names": data["feature_names"],
            "features_path": features_path,
        }

        results.append(result)

        print(
            f"  Training   - RMSE: ${train_metrics['RMSE']:.2f}, MAE: ${train_metrics['MAE']:.2f}, R²: {train_metrics['R2']:.4f}"
        )
        print(
            f"  Validation - RMSE: ${val_metrics['RMSE']:.2f}, MAE: ${val_metrics['MAE']:.2f}, R²: {val_metrics['R2']:.4f}"
        )
        print()

    return pd.DataFrame(results)


def analyze_best_configuration(results_df):
    """
    Identifies and analyzes the best performing configuration.

    Args:
        results_df (pd.DataFrame): Results from all configurations

    Returns:
        dict: Best configuration information

    Selection Criteria:
        - Primary: Lowest validation RMSE
        - Validation set represents future performance
        - Training metrics can be misleadingly good (overfitting)
    """
    # Sort by validation RMSE (lower is better)
    results_sorted = results_df.sort_values("val_rmse")

    best_config = results_sorted.iloc[0]

    print("=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"Smoothing: {best_config['smoothing']}")
    print(f"Number of Lags: {best_config['n_lags']}")
    print(f"Total Features: {best_config['n_features']}")
    print(f"Training Samples: {best_config['n_train_samples']}")
    print(f"Validation Samples: {best_config['n_val_samples']}")
    print()
    print("Training Metrics:")
    print(f"  RMSE: ${best_config['train_rmse']:.2f}")
    print(f"  MAE:  ${best_config['train_mae']:.2f}")
    print(f"  R²:   {best_config['train_r2']:.4f}")
    print()
    print("Validation Metrics:")
    print(f"  RMSE: ${best_config['val_rmse']:.2f}")
    print(f"  MAE:  ${best_config['val_mae']:.2f}")
    print(f"  R²:   {best_config['val_r2']:.4f}")
    print()

    # Analyze overfitting
    rmse_gap = best_config["val_rmse"] - best_config["train_rmse"]
    mae_gap = best_config["val_mae"] - best_config["train_mae"]
    r2_gap = best_config["train_r2"] - best_config["val_r2"]

    print("Overfitting Analysis:")
    print(f"  RMSE Gap (Val - Train): ${rmse_gap:.2f}")
    print(f"  MAE Gap (Val - Train):  ${mae_gap:.2f}")
    print(f"  R² Gap (Train - Val):   {r2_gap:.4f}")

    if rmse_gap > 50:
        print("  ⚠ Warning: Significant overfitting detected (large RMSE gap)")
    elif rmse_gap > 20:
        print("  ⚠ Moderate overfitting (moderate RMSE gap)")
    else:
        print("  ✓ Good generalization (small RMSE gap)")

    return best_config


def display_model_parameters(model, feature_names):
    """
    Displays the learned model parameters (coefficients).

    Args:
        model: Trained LinearRegression model
        feature_names: Names of features

    Model Equation:
        price = intercept + Σ(coefficient_i × feature_i)

    Interpretation (for scaled features):
        - Positive coefficient: feature increase → price increase
        - Negative coefficient: feature increase → price decrease
        - Larger absolute value: stronger influence
        - Features are scaled, so coefficients are directly comparable
    """
    print("=" * 80)
    print("MODEL PARAMETERS (LINEAR REGRESSION)")
    print("=" * 80)
    print()
    print(f"Intercept (β₀): ${model.intercept_:.2f}")
    print()
    print("Feature Coefficients:")
    print("-" * 60)
    print(f"{'Feature':<20} {'Coefficient':>15} {'Abs Value':>15}")
    print("-" * 60)

    # Create DataFrame for sorting
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_,
            "abs_coefficient": np.abs(model.coef_),
        }
    )

    # Sort by absolute value (most important first)
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    for _, row in coef_df.iterrows():
        print(
            f"{row['feature']:<20} {row['coefficient']:>15.4f} {row['abs_coefficient']:>15.4f}"
        )

    print("-" * 60)
    print()
    print("Interpretation:")
    print("  - Coefficients show the change in predicted price (in $)")
    print("    for a 1 standard deviation change in the feature")
    print("  - Positive: feature increase → price increase")
    print("  - Negative: feature increase → price decrease")
    print("  - Larger absolute value → stronger influence")
    print()

    # Highlight most important features
    print("Top 5 Most Influential Features:")
    for i, row in coef_df.head(5).iterrows():
        direction = "increases" if row["coefficient"] > 0 else "decreases"
        print(
            f"  {i+1}. {row['feature']}: {direction} price by ${abs(row['coefficient']):.2f} per std dev"
        )


def create_performance_comparison_plots(results_df, output_dir="results"):
    """
    Creates comprehensive visualizations comparing all configurations.

    Args:
        results_df (pd.DataFrame): Results from all configurations
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Validation RMSE by configuration
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Group by smoothing type
    for smoothing in ["raw", "sigma1", "sigma2", "sigma3"]:
        smoothing_data = results_df[results_df["smoothing"] == smoothing]

        if len(smoothing_data) == 0:
            continue

        ax = axes.flat[["raw", "sigma1", "sigma2", "sigma3"].index(smoothing)]

        x = smoothing_data["n_lags"]

        # Plot training and validation RMSE
        ax.plot(
            x,
            smoothing_data["train_rmse"],
            "o-",
            label="Training RMSE",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            x,
            smoothing_data["val_rmse"],
            "s-",
            label="Validation RMSE",
            linewidth=2,
            markersize=8,
        )

        ax.set_xlabel("Number of Lags", fontsize=11)
        ax.set_ylabel("RMSE ($)", fontsize=11)
        ax.set_title(
            f"Performance - Smoothing: {smoothing}", fontsize=12, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "baseline_performance_by_config.png")
    plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved performance comparison: {plot1_path}")
    plt.close()

    # Plot 2: R² comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for grouped bar chart
    smoothing_types = results_df["smoothing"].unique()
    n_lags_values = sorted(results_df["n_lags"].unique())

    x = np.arange(len(n_lags_values))
    width = 0.2

    for i, smoothing in enumerate(smoothing_types):
        smoothing_data = results_df[results_df["smoothing"] == smoothing].sort_values(
            "n_lags"
        )
        offset = width * (i - 1.5)
        ax.bar(x + offset, smoothing_data["val_r2"], width, label=smoothing, alpha=0.8)

    ax.set_xlabel("Number of Lags", fontsize=12)
    ax.set_ylabel("Validation R²", fontsize=12)
    ax.set_title("Validation R² Score by Configuration", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(n_lags_values)
    ax.legend(title="Smoothing")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "baseline_r2_comparison.png")
    plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved R² comparison: {plot2_path}")
    plt.close()


def create_actual_vs_predicted_plot(best_config, output_dir="results"):
    """
    Creates scatter plot of actual vs predicted prices for best model.

    Args:
        best_config: Best configuration from results
        output_dir (str): Directory to save plot
    """
    # Load data for best configuration
    data = load_feature_set(best_config["features_path"])

    # Get predictions
    model = best_config["model"]
    y_train_pred = model.predict(data["X_train"])
    y_val_pred = model.predict(data["X_val"])

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Training data
    ax1.scatter(data["y_train"], y_train_pred, alpha=0.6, s=30)
    ax1.plot(
        [data["y_train"].min(), data["y_train"].max()],
        [data["y_train"].min(), data["y_train"].max()],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    ax1.set_xlabel("Actual Close Price ($)", fontsize=11)
    ax1.set_ylabel("Predicted Close Price ($)", fontsize=11)
    ax1.set_title(
        f'Training Set - R²={best_config["train_r2"]:.4f}',
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Validation data
    ax2.scatter(data["y_val"], y_val_pred, alpha=0.6, s=30, color="orange")
    ax2.plot(
        [data["y_val"].min(), data["y_val"].max()],
        [data["y_val"].min(), data["y_val"].max()],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    ax2.set_xlabel("Actual Close Price ($)", fontsize=11)
    ax2.set_ylabel("Predicted Close Price ($)", fontsize=11)
    ax2.set_title(
        f'Validation Set - R²={best_config["val_r2"]:.4f}',
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f'Actual vs Predicted Prices - Best Model (smoothing={best_config["smoothing"]}, n_lags={best_config["n_lags"]})',
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "baseline_actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved actual vs predicted plot: {plot_path}")
    plt.close()


def save_results_table(results_df, output_dir="results"):
    """
    Saves detailed results table to CSV.

    Args:
        results_df (pd.DataFrame): Results from all configurations
        output_dir (str): Directory to save CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select columns for output
    output_df = results_df[
        [
            "smoothing",
            "n_lags",
            "n_features",
            "train_rmse",
            "train_mae",
            "train_r2",
            "val_rmse",
            "val_mae",
            "val_r2",
        ]
    ].copy()

    # Sort by validation RMSE
    output_df = output_df.sort_values("val_rmse")

    # Save to CSV
    csv_path = os.path.join(output_dir, "baseline_linear_regression_results.csv")
    output_df.to_csv(csv_path, index=False)
    print(f"✓ Saved results table: {csv_path}")

    return csv_path


def save_best_model(best_config, output_dir="models"):
    """
    Saves the best model for later use.

    Args:
        best_config: Best configuration dictionary
        output_dir (str): Directory to save model
    """
    os.makedirs(output_dir, exist_ok=True)

    model_info = {
        "model": best_config["model"],
        "smoothing": best_config["smoothing"],
        "n_lags": best_config["n_lags"],
        "feature_names": best_config["feature_names"],
        "features_path": best_config["features_path"],
        "train_rmse": best_config["train_rmse"],
        "val_rmse": best_config["val_rmse"],
        "train_r2": best_config["train_r2"],
        "val_r2": best_config["val_r2"],
    }

    model_path = os.path.join(output_dir, "best_baseline_linear_regression.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_info, f)

    print(f"✓ Saved best model: {model_path}")

    return model_path


def main():
    """
    Main execution function for baseline linear regression analysis.

    Workflow:
        1. Train models on all configurations
        2. Compare performance metrics
        3. Identify best configuration
        4. Display model parameters
        5. Create visualizations
        6. Save results and best model
    """
    print("=" * 80)
    print("NFLX STOCK PRICE PREDICTION - BASELINE LINEAR REGRESSION (TASK A)")
    print("=" * 80)
    print()

    # Train all configurations
    print("Training linear regression models on all configurations...")
    print()
    results_df = train_all_configurations()

    print("=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Total configurations tested: {len(results_df)}")
    print()

    # Analyze best configuration
    best_config = analyze_best_configuration(results_df)
    print()

    # Display model parameters
    display_model_parameters(best_config["model"], best_config["feature_names"])
    print()

    # Create visualizations
    print("Creating visualizations...")
    create_performance_comparison_plots(results_df)
    create_actual_vs_predicted_plot(best_config)
    print()

    # Save results
    print("Saving results...")
    save_results_table(results_df)
    save_best_model(best_config)
    print()

    print("=" * 80)
    print("✓ BASELINE LINEAR REGRESSION ANALYSIS COMPLETED")
    print("=" * 80)
    print()
    print("Key Findings:")
    print(f"  Best Smoothing: {best_config['smoothing']}")
    print(f"  Best Lag Window: {best_config['n_lags']} months")
    print(f"  Validation RMSE: ${best_config['val_rmse']:.2f}")
    print(f"  Validation R²: {best_config['val_r2']:.4f}")
    print()
    print("Next Step: Run step4_polynomial_regression_regularization.py for Task B")


if __name__ == "__main__":
    main()
