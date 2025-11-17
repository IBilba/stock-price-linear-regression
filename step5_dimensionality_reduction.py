"""
STEP 5: DIMENSIONALITY REDUCTION AND FEATURE SELECTION (TASK C)
================================================================

This script implements three dimensionality reduction approaches:
1. PCA (Principal Component Analysis) - Unsupervised transformation
2. CFS (Correlation-based Feature Selection) - Filter method
3. Sequential Forward Selection - Wrapper method

This addresses TASK C:
"Reduce the dimension by following PCA, CFA and a wrapper method of
your choice. Compare the results."

Each method reduces features differently:
- PCA: Creates new uncorrelated features (linear combinations)
- CFS: Selects features with high target correlation, low inter-correlation
- Wrapper: Uses model performance to select features

Author: Statistical Methods of Machine Learning - Task 1
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


def load_feature_set(features_path):
    """Loads saved feature set."""
    data = np.load(features_path, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "y_train": data["y_train"],
        "y_val": data["y_val"],
        "feature_names": data["feature_names"],
    }


def compute_metrics(y_true, y_pred):
    """Computes RMSE, MAE, R²."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def apply_pca(X_train, X_val, variance_threshold=0.95):
    """
    Applies PCA (Principal Component Analysis) for dimensionality reduction.

    Args:
        X_train: Training features (already scaled)
        X_val: Validation features (already scaled)
        variance_threshold: Minimum cumulative variance to retain (default: 95%)

    Returns:
        tuple: (X_train_pca, X_val_pca, pca_model, n_components)

    Principal Component Analysis (PCA):

        Goal: Transform features into uncorrelated principal components

        How it works:
        1. Finds directions of maximum variance in data
        2. Projects data onto these directions
        3. First PC captures most variance, second PC captures second-most, etc.
        4. PCs are orthogonal (uncorrelated) to each other

        Mathematical Foundation:
        - Eigenvalue decomposition of covariance matrix
        - PCs are eigenvectors, variance explained is eigenvalues
        - PC₁ = w₁₁·x₁ + w₁₂·x₂ + ... + w₁ₙ·xₙ (linear combination)

        Benefits:
        - Removes multicollinearity (lagged prices are highly correlated)
        - Reduces dimensionality while preserving variance
        - Noise reduction (minor components often represent noise)
        - Computational efficiency with fewer features

        Drawbacks:
        - Loss of interpretability (PCs are combinations, not original features)
        - Assumes linear relationships
        - Sensitive to feature scaling (already handled in our pipeline)

        For Stock Prediction:
        - Lagged prices (close_t-1, close_t-2, ...) are highly correlated
        - PCA can capture the "trend" in one component
        - Volume features may form separate components
        - Typically 3-5 components explain >95% variance

    Variance Threshold:
        - 0.95 (95%): Retains most information, moderate reduction
        - 0.90 (90%): More aggressive reduction
        - 0.99 (99%): Conservative, minimal reduction
    """
    # Apply PCA
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    n_components = X_train_pca.shape[1]
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(f"  Original features: {X_train.shape[1]}")
    print(f"  Selected components: {n_components}")
    print(f"  Variance explained: {cumulative_variance[-1]:.4f}")
    print(f"  Reduction: {X_train.shape[1] - n_components} features removed")
    print(f"\n  Variance per component:")
    for i, var in enumerate(explained_variance[:10], 1):  # Show first 10
        print(f"    PC{i}: {var:.4f} ({cumulative_variance[i-1]:.4f} cumulative)")

    return X_train_pca, X_val_pca, pca, n_components


def compute_cfs_merit(X, y, feature_indices):
    """
    Computes CFS (Correlation-based Feature Selection) merit score.

    Args:
        X: Feature matrix
        y: Target vector
        feature_indices: Indices of features to evaluate

    Returns:
        float: Merit score

    CFS Merit Formula:
        Merit(S) = (k · r̄cf) / √(k + k(k-1) · r̄ff)

        where:
        - k: Number of features in subset S
        - r̄cf: Average correlation between features and target (class)
        - r̄ff: Average correlation between features (feature-feature)

    Interpretation:
        - Numerator: Rewards high feature-target correlation
        - Denominator: Penalizes high feature-feature correlation (redundancy)
        - Higher merit = better feature subset
        - Balances relevance (to target) with redundancy (among features)

    Why CFS is Effective:
        - Selects features highly correlated with target
        - Avoids redundant features (e.g., close_t-1 and close_t-2 are similar)
        - Fast computation (correlation-based, no model training)
        - Works well for linear relationships

    For Stock Prediction:
        - Recent lags (close_t-1, close_t-2) highly correlated with target
        - These lags also highly correlated with each other
        - CFS might select close_t-1 and skip close_t-2 (redundant)
        - Volume features less correlated → may be included if adding new info
    """
    if len(feature_indices) == 0:
        return 0.0

    # Select features
    X_subset = X[:, feature_indices]
    k = len(feature_indices)

    # Compute feature-class correlations
    rcf_values = []
    for i in range(k):
        corr = np.corrcoef(X_subset[:, i], y)[0, 1]
        rcf_values.append(abs(corr))
    rcf_mean = np.mean(rcf_values)

    # Compute feature-feature correlations
    if k > 1:
        rff_values = []
        for i in range(k):
            for j in range(i + 1, k):
                corr = np.corrcoef(X_subset[:, i], X_subset[:, j])[0, 1]
                rff_values.append(abs(corr))
        rff_mean = np.mean(rff_values) if rff_values else 0.0
    else:
        rff_mean = 0.0

    # Compute merit
    numerator = k * rcf_mean
    denominator = np.sqrt(k + k * (k - 1) * rff_mean)

    if denominator == 0:
        return 0.0

    merit = numerator / denominator
    return merit


def apply_cfs(X_train, y_train, X_val, feature_names, max_features=None):
    """
    Applies CFS using forward selection based on merit.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        feature_names: Names of features
        max_features: Maximum features to select (None = automatic)

    Returns:
        tuple: (X_train_cfs, X_val_cfs, selected_indices, selected_names)

    CFS Algorithm (Forward Selection):
        1. Start with empty feature set
        2. For each remaining feature:
           - Try adding it to current set
           - Compute merit with this feature
        3. Add feature that most increases merit
        4. Repeat until merit stops improving
        5. Return best feature subset

    Stopping Criterion:
        - Merit stops increasing
        - Or max_features reached

    Time Complexity:
        - O(n² · m) where n = features, m = samples
        - Much faster than wrapper methods (no model training)
    """
    n_features = X_train.shape[1]

    # Initialize
    selected_indices = []
    remaining_indices = list(range(n_features))
    best_merit = 0.0

    print(f"  Starting CFS forward selection...")
    print(f"  Total features: {n_features}")

    # Forward selection loop
    iteration = 0
    while remaining_indices:
        iteration += 1

        # Try adding each remaining feature
        merits = []
        for idx in remaining_indices:
            candidate_indices = selected_indices + [idx]
            merit = compute_cfs_merit(X_train, y_train, candidate_indices)
            merits.append((merit, idx))

        # Find best feature to add
        current_best_merit, best_idx = max(merits, key=lambda x: x[0])

        # Check if merit improved
        if current_best_merit > best_merit:
            best_merit = current_best_merit
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

            print(
                f"    Iteration {iteration}: Added feature {best_idx} "
                f"({feature_names[best_idx]}), Merit: {best_merit:.4f}"
            )

            # Check max_features
            if max_features and len(selected_indices) >= max_features:
                break
        else:
            # Merit didn't improve, stop
            print(f"    Iteration {iteration}: No improvement, stopping")
            break

    # Extract selected features
    selected_indices = sorted(selected_indices)
    selected_names = [feature_names[i] for i in selected_indices]

    X_train_cfs = X_train[:, selected_indices]
    X_val_cfs = X_val[:, selected_indices]

    print(f"\n  CFS Results:")
    print(f"    Selected {len(selected_indices)} features")
    print(f"    Final merit: {best_merit:.4f}")
    print(f"    Selected features: {selected_names}")

    return X_train_cfs, X_val_cfs, selected_indices, selected_names


def apply_forward_selection(
    X_train, y_train, X_val, feature_names, n_features_to_select
):
    """
    Applies Sequential Forward Selection (wrapper method).

    Args:
        X_train, y_train: Training data
        X_val: Validation data
        feature_names: Feature names
        n_features_to_select: Number of features to select

    Returns:
        tuple: (X_train_selected, X_val_selected, selected_indices, selected_names)

    Sequential Forward Selection (Wrapper Method):

        Algorithm:
        1. Start with empty feature set
        2. For each remaining feature:
           - Train model with current features + this feature
           - Evaluate performance (cross-validation)
        3. Add feature that gives best performance
        4. Repeat until desired number of features selected

        Evaluation:
        - Uses actual model performance (not correlation)
        - Cross-validation to avoid overfitting
        - For regression: uses negative MSE as scoring metric

        Benefits:
        - Considers feature interactions
        - Optimizes for specific model (LinearRegression)
        - Usually gives best predictive performance

        Drawbacks:
        - Computationally expensive (trains many models)
        - Can overfit to training data
        - Sensitive to cross-validation splits
        - Greedy algorithm (may not find global optimum)

    Comparison with CFS:
        - CFS: Fast, correlation-based, no model training
        - Wrapper: Slow, model-based, considers interactions
        - CFS good for quick feature selection
        - Wrapper better for final model optimization

    Time Complexity:
        - O(n² · model_training_time)
        - With n=24 features, selects k=10 → ~170 models trained
    """
    print(f"  Starting Sequential Forward Selection...")
    print(f"  Target: {n_features_to_select} features")
    print(f"  This may take a minute...")

    # Create selector
    model = LinearRegression()
    selector = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features_to_select,
        direction="forward",
        scoring="neg_mean_squared_error",  # For regression
        cv=3,  # 3-fold cross-validation
        n_jobs=-1,  # Use all CPU cores
    )

    # Fit selector
    selector.fit(X_train, y_train)

    # Get selected features
    selected_mask = selector.get_support()
    selected_indices = np.where(selected_mask)[0]
    selected_names = [feature_names[i] for i in selected_indices]

    # Transform data
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]

    print(f"\n  Forward Selection Results:")
    print(f"    Selected {len(selected_indices)} features")
    print(f"    Selected features: {selected_names}")

    return X_train_selected, X_val_selected, selected_indices, selected_names


def train_and_evaluate(X_train, y_train, X_val, y_val, method_name):
    """
    Trains linear regression and evaluates on both sets.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        method_name: Name for reporting

    Returns:
        dict: Results including model and metrics
    """
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Metrics
    train_metrics = compute_metrics(y_train, y_train_pred)
    val_metrics = compute_metrics(y_val, y_val_pred)

    print(f"\n{method_name} - Model Performance:")
    print(f"  Training:")
    print(f"    RMSE: ${train_metrics['RMSE']:.2f}")
    print(f"    MAE:  ${train_metrics['MAE']:.2f}")
    print(f"    R²:   {train_metrics['R2']:.4f}")
    print(f"  Validation:")
    print(f"    RMSE: ${val_metrics['RMSE']:.2f}")
    print(f"    MAE:  ${val_metrics['MAE']:.2f}")
    print(f"    R²:   {val_metrics['R2']:.4f}")

    return {
        "method": method_name,
        "model": model,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "n_features": X_train.shape[1],
    }


def create_comparison_plot(results_list, output_dir="results"):
    """
    Creates visualization comparing all dimensionality reduction methods.

    Args:
        results_list: List of result dictionaries
        output_dir: Output directory
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = [r["method"] for r in results_list]
    n_features = [r["n_features"] for r in results_list]
    train_rmse = [r["train_metrics"]["RMSE"] for r in results_list]
    val_rmse = [r["val_metrics"]["RMSE"] for r in results_list]
    val_r2 = [r["val_metrics"]["R2"] for r in results_list]

    # Plot 1: Number of features
    ax = axes[0]
    bars = ax.bar(
        methods, n_features, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    )
    ax.set_ylabel("Number of Features", fontsize=11)
    ax.set_title("Feature Count by Method", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Plot 2: RMSE comparison
    ax = axes[1]
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width / 2, train_rmse, width, label="Training RMSE", color="skyblue")
    ax.bar(x + width / 2, val_rmse, width, label="Validation RMSE", color="orange")
    ax.set_ylabel("RMSE ($)", fontsize=11)
    ax.set_title("RMSE Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Plot 3: R² comparison
    ax = axes[2]
    bars = ax.bar(methods, val_r2, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_ylabel("Validation R²", fontsize=11)
    ax.set_title("R² Score Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([min(val_r2) * 0.95, 1.0])
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "dimensionality_reduction_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved comparison plot: {plot_path}")
    plt.close()


def save_results(results_list, output_dir="results"):
    """Saves dimensionality reduction results."""
    os.makedirs(output_dir, exist_ok=True)

    # Create DataFrame
    comparison_data = []
    for r in results_list:
        comparison_data.append(
            {
                "Method": r["method"],
                "N_Features": r["n_features"],
                "Train_RMSE": r["train_metrics"]["RMSE"],
                "Train_MAE": r["train_metrics"]["MAE"],
                "Train_R2": r["train_metrics"]["R2"],
                "Val_RMSE": r["val_metrics"]["RMSE"],
                "Val_MAE": r["val_metrics"]["MAE"],
                "Val_R2": r["val_metrics"]["R2"],
            }
        )

    df = pd.DataFrame(comparison_data)
    df = df.sort_values("Val_RMSE")

    # Save CSV
    csv_path = os.path.join(output_dir, "dimensionality_reduction_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved results table: {csv_path}")

    # Save models
    models_dir = "models"
    for r in results_list:
        method_name = r["method"].replace(" ", "_").lower()
        model_path = os.path.join(models_dir, f"{method_name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(r, f)
        print(f"✓ Saved {r['method']} model: {model_path}")


def main():
    """
    Main execution function for dimensionality reduction analysis.

    Compares:
    1. Original features (baseline) - 24 features
    2. PCA (95% variance) - typically 3-8 components
    3. CFS (forward selection) - selects best subset
    4. Sequential Forward Selection - model-based selection
    """
    print("=" * 80)
    print("NFLX STOCK PRICE PREDICTION - DIMENSIONALITY REDUCTION (TASK C)")
    print("=" * 80)
    print()

    # Load best baseline configuration
    features_path = "features/features_sigma3_12lags.npz"
    print("Loading data...")
    data = load_feature_set(features_path)
    print(f"✓ Loaded: {features_path}")
    print(f"  Training samples: {len(data['y_train'])}")
    print(f"  Validation samples: {len(data['y_val'])}")
    print(f"  Original features: {data['X_train'].shape[1]}")
    print()

    results_list = []

    # 1. Baseline (Original Features)
    print("=" * 80)
    print("1. BASELINE (Original Features)")
    print("=" * 80)
    result_baseline = train_and_evaluate(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"], "Baseline"
    )
    results_list.append(result_baseline)

    # 2. PCA
    print("\n" + "=" * 80)
    print("2. PCA (Principal Component Analysis)")
    print("=" * 80)
    print("Applying PCA with 95% variance threshold...")
    X_train_pca, X_val_pca, pca_model, n_components = apply_pca(
        data["X_train"], data["X_val"], variance_threshold=0.95
    )
    result_pca = train_and_evaluate(
        X_train_pca, data["y_train"], X_val_pca, data["y_val"], "PCA"
    )
    results_list.append(result_pca)

    # 3. CFS
    print("\n" + "=" * 80)
    print("3. CFS (Correlation-based Feature Selection)")
    print("=" * 80)
    X_train_cfs, X_val_cfs, cfs_indices, cfs_names = apply_cfs(
        data["X_train"], data["y_train"], data["X_val"], data["feature_names"]
    )
    result_cfs = train_and_evaluate(
        X_train_cfs, data["y_train"], X_val_cfs, data["y_val"], "CFS"
    )
    results_list.append(result_cfs)

    # 4. Sequential Forward Selection
    print("\n" + "=" * 80)
    print("4. SEQUENTIAL FORWARD SELECTION (Wrapper Method)")
    print("=" * 80)
    # Select 50% of features as target
    n_target_features = max(3, data["X_train"].shape[1] // 2)
    X_train_sfs, X_val_sfs, sfs_indices, sfs_names = apply_forward_selection(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["feature_names"],
        n_target_features,
    )
    result_sfs = train_and_evaluate(
        X_train_sfs, data["y_train"], X_val_sfs, data["y_val"], "Forward Selection"
    )
    results_list.append(result_sfs)

    # Comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    comparison_df = pd.DataFrame(
        [
            {
                "Method": r["method"],
                "N_Features": r["n_features"],
                "Val_RMSE": r["val_metrics"]["RMSE"],
                "Val_R2": r["val_metrics"]["R2"],
            }
            for r in results_list
        ]
    )
    comparison_df = comparison_df.sort_values("Val_RMSE")
    print("\n", comparison_df.to_string(index=False))

    # Visualizations
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    create_comparison_plot(results_list)

    # Save results
    print()
    save_results(results_list)

    # Summary
    best = comparison_df.iloc[0]
    print(f"\n{'='*80}")
    print("✓ DIMENSIONALITY REDUCTION ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print(f"\nBest Method: {best['Method']}")
    print(f"  Features: {int(best['N_Features'])}")
    print(f"  Validation RMSE: ${best['Val_RMSE']:.2f}")
    print(f"  Validation R²: {best['Val_R2']:.4f}")
    print()
    print("Next Step: Run step6_future_predictions.py for Task D")


if __name__ == "__main__":
    main()
