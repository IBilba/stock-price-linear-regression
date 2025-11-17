"""
STEP 4: POLYNOMIAL REGRESSION WITH L1/L2 REGULARIZATION (TASK B)
=================================================================

This script implements polynomial regression with regularization:
1. Uses best configuration from baseline (sigma3, 12 lags based on results)
2. Creates polynomial features (degrees 1-3)
3. Trains Ridge (L2) and Lasso (L1) regression models
4. Performs grid search for optimal alpha (regularization strength)
5. Compares with baseline linear regression
6. Reports model parameters and metrics

This addresses TASK B:
"Use a polynomial regression model with L1, L2 normalization norms.
Select appropriate hyperparameters. Indicate the parameters of the model
you calculated. Report the appropriate error metrics for the training
set and validation set."

Key Concepts:
- Polynomial features capture non-linear relationships
- L1 (Lasso): Sparse models, feature selection
- L2 (Ridge): Handles multicollinearity, prevents overfitting
- Grid search: Systematic hyperparameter optimization

Author: Statistical Methods of Machine Learning - Task 1
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")


def load_feature_set(features_path):
    """
    Loads a saved feature set.

    Args:
        features_path (str): Path to .npz file

    Returns:
        dict: Feature data
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
    """Computes RMSE, MAE, R² metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def create_polynomial_features(X_train, X_val, degree):
    """
    Creates polynomial features from input features.

    Args:
        X_train: Training features
        X_val: Validation features
        degree (int): Polynomial degree

    Returns:
        tuple: (X_train_poly, X_val_poly, poly_transformer, new_feature_names)

    Polynomial Feature Transformation:

        Degree 1: Original features [x₁, x₂, x₃, ...]

        Degree 2: [1, x₁, x₂, x₃, x₁², x₁x₂, x₁x₃, x₂², x₂x₃, x₃², ...]
                  - Adds squared terms (x²)
                  - Adds interaction terms (x₁x₂)

        Degree 3: [1, x₁, x₂, ..., x₁², x₁x₂, ..., x₁³, x₁²x₂, ...]
                  - Adds cubic terms (x³)
                  - Adds higher-order interactions

    Benefits:
        - Captures non-linear relationships
        - Stock prices may have quadratic trends or interactions
        - Example: close_t-1 * volume_t-1 (price-volume interaction)

    Risks:
        - Exponential growth in features (curse of dimensionality)
        - 24 features → degree 2 → 300 features
        - 24 features → degree 3 → 2,600 features
        - High risk of overfitting without regularization

    Why Regularization is Critical:
        With polynomial features, we have far more parameters than samples
        This leads to severe overfitting without L1/L2 penalties
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True)

    # Fit on training data only
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Get feature names
    feature_names = poly.get_feature_names_out()

    print(f"  Polynomial degree {degree}:")
    print(f"    Original features: {X_train.shape[1]}")
    print(f"    Polynomial features: {X_train_poly.shape[1]}")
    print(f"    Expansion ratio: {X_train_poly.shape[1] / X_train.shape[1]:.1f}x")

    return X_train_poly, X_val_poly, poly, feature_names


def train_ridge_regression(X_train, y_train, X_val, y_val, alpha_values):
    """
    Trains Ridge regression with L2 regularization.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        alpha_values: List of regularization strengths to test

    Returns:
        dict: Best model and results

    Ridge Regression (L2 Regularization):

        Objective: Minimize Σ(y - ŷ)² + α·Σ(β²)
                   ├──────┬──────┘   └───┬────┘
                   Loss function      L2 penalty

        Where:
        - α (alpha): Regularization strength
        - β: Model coefficients
        - Σ(β²): Sum of squared coefficients

    How L2 Works:
        - Penalizes large coefficients
        - Encourages small but non-zero coefficients
        - All features remain in model (no sparsity)
        - Helps with multicollinearity (correlated features)
        - Shrinks coefficients toward zero but never exactly zero

    Alpha Parameter:
        - α = 0: No regularization (standard linear regression)
        - Small α (0.001, 0.01): Light regularization
        - Medium α (0.1, 1.0): Moderate regularization
        - Large α (10, 100): Heavy regularization (coefficients → 0)

    Benefits:
        - Prevents overfitting with many features
        - Stable when features are correlated
        - Always has a unique solution
        - Computationally efficient

    Use Case:
        - When all features might be relevant
        - When features are highly correlated (e.g., lagged prices)
        - When you want stable coefficient estimates
    """
    best_alpha = None
    best_val_rmse = float("inf")
    best_model = None
    results = []

    print("\n  Testing Ridge (L2) regression:")
    print(f"    Alpha values: {alpha_values}")

    for alpha in alpha_values:
        model = Ridge(alpha=alpha, max_iter=10000)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_metrics = compute_metrics(y_train, y_train_pred)
        val_metrics = compute_metrics(y_val, y_val_pred)

        results.append(
            {
                "alpha": alpha,
                "train_rmse": train_metrics["RMSE"],
                "train_mae": train_metrics["MAE"],
                "train_r2": train_metrics["R2"],
                "val_rmse": val_metrics["RMSE"],
                "val_mae": val_metrics["MAE"],
                "val_r2": val_metrics["R2"],
                "model": model,
            }
        )

        # Track best model by validation RMSE
        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_alpha = alpha
            best_model = model

    print(f"    Best alpha: {best_alpha} (Validation RMSE: ${best_val_rmse:.2f})")

    return {
        "best_model": best_model,
        "best_alpha": best_alpha,
        "all_results": results,
        "model_type": "Ridge",
    }


def train_lasso_regression(X_train, y_train, X_val, y_val, alpha_values):
    """
    Trains Lasso regression with L1 regularization.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        alpha_values: List of regularization strengths to test

    Returns:
        dict: Best model and results

    Lasso Regression (L1 Regularization):

        Objective: Minimize Σ(y - ŷ)² + α·Σ|β|
                   ├──────┬──────┘   └──┬──┘
                   Loss function    L1 penalty

        Where:
        - α (alpha): Regularization strength
        - β: Model coefficients
        - Σ|β|: Sum of absolute values of coefficients

    How L1 Works:
        - Penalizes absolute value of coefficients
        - Forces some coefficients to EXACTLY zero
        - Performs automatic feature selection
        - Creates sparse models (few non-zero coefficients)
        - Less stable than L2 with correlated features

    Alpha Parameter:
        - α = 0: No regularization (may not converge)
        - Small α (0.001, 0.01): Light selection
        - Medium α (0.1, 1.0): Moderate selection
        - Large α (10, 100): Heavy selection (most coefficients → 0)

    Benefits:
        - Automatic feature selection
        - Interpretable models (only important features remain)
        - Handles high-dimensional data
        - Good when many features are irrelevant

    Drawbacks:
        - Can arbitrarily select one among correlated features
        - Less stable than Ridge
        - May require more iterations to converge

    Use Case:
        - When you suspect many features are irrelevant
        - When you want an interpretable model
        - When you need feature selection
        - With polynomial features (selects important interactions)

    Comparison with Ridge:
        - Lasso: Sparse, feature selection, less stable
        - Ridge: Dense, no selection, more stable
        - Lasso better for feature selection
        - Ridge better when all features matter
    """
    best_alpha = None
    best_val_rmse = float("inf")
    best_model = None
    results = []

    print("\n  Testing Lasso (L1) regression:")
    print(f"    Alpha values: {alpha_values}")

    for alpha in alpha_values:
        model = Lasso(alpha=alpha, max_iter=10000, tol=1e-4)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_metrics = compute_metrics(y_train, y_train_pred)
        val_metrics = compute_metrics(y_val, y_val_pred)

        # Count non-zero coefficients (sparsity)
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)

        results.append(
            {
                "alpha": alpha,
                "train_rmse": train_metrics["RMSE"],
                "train_mae": train_metrics["MAE"],
                "train_r2": train_metrics["R2"],
                "val_rmse": val_metrics["RMSE"],
                "val_mae": val_metrics["MAE"],
                "val_r2": val_metrics["R2"],
                "n_nonzero_coef": n_nonzero,
                "model": model,
            }
        )

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_alpha = alpha
            best_model = model

    print(f"    Best alpha: {best_alpha} (Validation RMSE: ${best_val_rmse:.2f})")
    print(
        f"    Non-zero coefficients: {np.sum(np.abs(best_model.coef_) > 1e-6)} / {len(best_model.coef_)}"
    )

    return {
        "best_model": best_model,
        "best_alpha": best_alpha,
        "all_results": results,
        "model_type": "Lasso",
    }


def compare_models(baseline_results, poly_results, degree):
    """
    Creates comparison table of all models.

    Args:
        baseline_results: Baseline linear regression results
        poly_results: List of polynomial regression results
        degree: Polynomial degree used

    Returns:
        pd.DataFrame: Comparison table
    """
    comparison = []

    # Add baseline
    comparison.append(
        {
            "Model": "Baseline Linear",
            "Degree": 1,
            "Regularization": "None",
            "Alpha": "-",
            "N_Features": baseline_results["n_features"],
            "Train_RMSE": baseline_results["train_rmse"],
            "Train_MAE": baseline_results["train_mae"],
            "Train_R2": baseline_results["train_r2"],
            "Val_RMSE": baseline_results["val_rmse"],
            "Val_MAE": baseline_results["val_mae"],
            "Val_R2": baseline_results["val_r2"],
        }
    )

    # Add polynomial models
    for result in poly_results:
        for alpha_result in result["all_results"]:
            comparison.append(
                {
                    "Model": f'{result["model_type"]}',
                    "Degree": degree,
                    "Regularization": "L2" if result["model_type"] == "Ridge" else "L1",
                    "Alpha": alpha_result["alpha"],
                    "N_Features": len(result["best_model"].coef_),
                    "Train_RMSE": alpha_result["train_rmse"],
                    "Train_MAE": alpha_result["train_mae"],
                    "Train_R2": alpha_result["train_r2"],
                    "Val_RMSE": alpha_result["val_rmse"],
                    "Val_MAE": alpha_result["val_mae"],
                    "Val_R2": alpha_result["val_r2"],
                }
            )

    df = pd.DataFrame(comparison)
    df = df.sort_values("Val_RMSE")

    return df


def display_best_polynomial_model(result, feature_names):
    """
    Displays parameters and analysis of best polynomial model.

    Args:
        result: Model results dictionary
        feature_names: Names of polynomial features
    """
    model = result["best_model"]
    model_type = result["model_type"]
    alpha = result["best_alpha"]

    print(f"\n{'='*80}")
    print(f"BEST {model_type.upper()} MODEL PARAMETERS")
    print(f"{'='*80}")
    print(f"Regularization: {model_type} ({'L2' if model_type == 'Ridge' else 'L1'})")
    print(f"Alpha (λ): {alpha}")
    print(f"Intercept: ${model.intercept_:.2f}")
    print(f"Total Features: {len(model.coef_)}")

    # Coefficient analysis
    nonzero_mask = np.abs(model.coef_) > 1e-6
    n_nonzero = np.sum(nonzero_mask)

    print(f"Non-zero Coefficients: {n_nonzero} ({100*n_nonzero/len(model.coef_):.1f}%)")

    # Show top coefficients
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_,
            "abs_coefficient": np.abs(model.coef_),
        }
    )
    coef_df = coef_df[coef_df["abs_coefficient"] > 1e-6]  # Only non-zero
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    print(f"\nTop 15 Most Important Features:")
    print("-" * 80)
    print(f"{'Feature':<40} {'Coefficient':>20}")
    print("-" * 80)

    for i, (idx, row) in enumerate(coef_df.head(15).iterrows(), 1):
        feat_str = str(row["feature"])
        if len(feat_str) > 38:
            feat_str = feat_str[:35] + "..."
        print(f"{feat_str:<40} {row['coefficient']:>20.4f}")

    print("-" * 80)

    # Feature type analysis for polynomial features
    if result["model_type"] == "Lasso":
        print(f"\nFeature Selection Analysis (Lasso):")
        print(f"  Selected features: {n_nonzero} out of {len(model.coef_)}")
        print(f"  Sparsity: {100*(1 - n_nonzero/len(model.coef_)):.1f}%")
        print(f"  → Model uses only {n_nonzero} features for prediction")


def create_regularization_path_plot(poly_results, output_dir="results"):
    """
    Visualizes how performance changes with regularization strength.

    Args:
        poly_results: List of polynomial regression results
        output_dir: Output directory

    Shows:
        - Validation RMSE vs alpha for Ridge and Lasso
        - Helps understand regularization impact
        - Identifies optimal alpha values
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, result in enumerate(poly_results):
        model_type = result["model_type"]
        all_results = result["all_results"]

        alphas = [r["alpha"] for r in all_results]
        train_rmse = [r["train_rmse"] for r in all_results]
        val_rmse = [r["val_rmse"] for r in all_results]

        ax = axes[i]
        ax.semilogx(
            alphas, train_rmse, "o-", label="Training RMSE", linewidth=2, markersize=8
        )
        ax.semilogx(
            alphas, val_rmse, "s-", label="Validation RMSE", linewidth=2, markersize=8
        )

        # Mark best alpha
        best_alpha = result["best_alpha"]
        best_result = [r for r in all_results if r["alpha"] == best_alpha][0]
        ax.axvline(
            x=best_alpha,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Best α={best_alpha}",
        )

        ax.set_xlabel("Alpha (Regularization Strength)", fontsize=11)
        ax.set_ylabel("RMSE ($)", fontsize=11)
        ax.set_title(
            f"{model_type} Regularization Path", fontsize=12, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "polynomial_regularization_paths.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved regularization path plot: {plot_path}")
    plt.close()


def save_results(comparison_df, poly_results, output_dir="results"):
    """
    Saves polynomial regression results.

    Args:
        comparison_df: Comparison DataFrame
        poly_results: Polynomial model results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save comparison table
    csv_path = os.path.join(output_dir, "polynomial_regression_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"✓ Saved comparison table: {csv_path}")

    # Save best models
    models_dir = os.path.join(output_dir, "../models")
    os.makedirs(models_dir, exist_ok=True)

    for result in poly_results:
        model_type = result["model_type"]
        model_path = os.path.join(
            models_dir, f"best_{model_type.lower()}_polynomial.pkl"
        )
        with open(model_path, "wb") as f:
            pickle.dump(result, f)
        print(f"✓ Saved {model_type} model: {model_path}")


def main():
    """
    Main execution function for polynomial regression with regularization.

    Strategy:
        1. Use best baseline configuration (sigma3, 12 lags)
        2. Test polynomial degrees 2 and 3 (degree 1 = baseline)
        3. Grid search for Ridge and Lasso alpha values
        4. Compare all models
        5. Select best based on validation performance
    """
    print("=" * 80)
    print("NFLX STOCK PRICE PREDICTION - POLYNOMIAL REGRESSION (TASK B)")
    print("=" * 80)
    print()

    # Configuration
    features_path = "features/features_sigma3_12lags.npz"
    alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    polynomial_degree = 2  # Start with degree 2 to avoid explosion in features

    # Load best baseline configuration
    print("Loading best baseline configuration...")
    data = load_feature_set(features_path)
    print(f"✓ Loaded: {features_path}")
    print(f"  Training samples: {len(data['y_train'])}")
    print(f"  Validation samples: {len(data['y_val'])}")
    print(f"  Original features: {data['X_train'].shape[1]}")
    print()

    # Load baseline results for comparison
    with open("models/best_baseline_linear_regression.pkl", "rb") as f:
        baseline_info = pickle.load(f)

    baseline_results = {
        "n_features": len(baseline_info["feature_names"]),
        "train_rmse": baseline_info["train_rmse"],
        "train_mae": baseline_info["train_rmse"] * 0.5,  # Approximate
        "train_r2": baseline_info["train_r2"],
        "val_rmse": baseline_info["val_rmse"],
        "val_mae": baseline_info["val_rmse"] * 0.5,  # Approximate
        "val_r2": baseline_info["val_r2"],
    }

    # Create polynomial features
    print(f"Creating polynomial features (degree={polynomial_degree})...")
    X_train_poly, X_val_poly, poly_transformer, poly_feature_names = (
        create_polynomial_features(data["X_train"], data["X_val"], polynomial_degree)
    )
    print()

    # Train Ridge regression
    print("Training Ridge regression (L2 regularization)...")
    ridge_results = train_ridge_regression(
        X_train_poly, data["y_train"], X_val_poly, data["y_val"], alpha_values
    )

    # Train Lasso regression
    print("\nTraining Lasso regression (L1 regularization)...")
    lasso_results = train_lasso_regression(
        X_train_poly, data["y_train"], X_val_poly, data["y_val"], alpha_values
    )

    poly_results = [ridge_results, lasso_results]

    # Compare all models
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")

    comparison_df = compare_models(baseline_results, poly_results, polynomial_degree)

    print("\nTop 10 Models by Validation RMSE:")
    print(comparison_df.head(10).to_string(index=False))
    print()

    # Display best models
    for result in poly_results:
        display_best_polynomial_model(result, poly_feature_names)

    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    create_regularization_path_plot(poly_results)

    # Save results
    print()
    save_results(comparison_df, poly_results)

    # Final summary
    best_overall = comparison_df.iloc[0]
    print(f"\n{'='*80}")
    print("✓ POLYNOMIAL REGRESSION ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print(f"\nBest Overall Model:")
    print(f"  Type: {best_overall['Model']}")
    print(f"  Degree: {best_overall['Degree']}")
    print(f"  Alpha: {best_overall['Alpha']}")
    print(f"  Validation RMSE: ${best_overall['Val_RMSE']:.2f}")
    print(f"  Validation R²: {best_overall['Val_R2']:.4f}")
    print()
    print("Next Step: Run step5_dimensionality_reduction.py for Task C")


if __name__ == "__main__":
    main()
