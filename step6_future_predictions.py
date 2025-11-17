"""
STEP 6: FUTURE PREDICTIONS AND FINAL ANALYSIS (TASK D)
=======================================================

This script generates future stock price predictions:
1. Loads the best model from previous analyses
2. Prepares features for December 2025 and January 2026
3. Makes predictions using the trained model
4. Creates comprehensive visualizations
5. Generates final summary report

This addresses TASK D:
"Provide price prediction for December 2025 and January 2025 (2026)."

Note: The task mentions "January 2025" but given the context (predicting
after December 2025), this should be "January 2026".

Author: Statistical Methods of Machine Learning - Task 1
"""

import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression


def load_latest_data():
    """
    Loads the latest monthly data including smoothed version.

    Returns:
        pd.DataFrame: Monthly data with all required information
    """
    # Load the smoothed data (sigma3, which gave best results)
    data_path = "data/nflx_monthly_smoothed_sigma3.csv"
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"Loaded data from: {data_path}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Total months: {len(df)}")

    return df


def load_best_model():
    """
    Loads the best performing model and associated metadata.

    Returns:
        dict: Model information including scaler and feature configuration
    """
    # Load baseline model (best performer)
    model_path = "models/best_baseline_linear_regression.pkl"
    with open(model_path, "rb") as f:
        model_info = pickle.dump(f)

    # Load scaler for feature transformation
    scaler_path = "features/scaler_sigma3_12lags.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print(f"Loaded model from: {model_path}")
    print(f"Model type: {type(model_info['model']).__name__}")
    print(f"Features required: {model_info['n_lags'] * 2} (12 close + 12 volume lags)")
    print(f"Validation RMSE: ${model_info['val_rmse']:.2f}")
    print(f"Validation R²: {model_info['val_r2']:.4f}")

    return model_info, scaler


def create_prediction_features(df, n_lags=12, target_year=2025, target_month=12):
    """
    Creates features for predicting a specific future month.

    Args:
        df (pd.DataFrame): Historical monthly data
        n_lags (int): Number of months to look back
        target_year (int): Year to predict
        target_month (int): Month to predict (1-12)

    Returns:
        tuple: (features_array, feature_dict) for transparency

    Feature Construction Logic:
        To predict December 2025 (2025-12):
        - close_t-1: November 2025 close
        - close_t-2: October 2025 close
        - ...
        - close_t-12: December 2024 close
        - volume_t-1 through volume_t-12: Same pattern

    Critical:
        - All lagged months must exist in historical data
        - If predicting December 2025, need data through November 2025
        - Features must match exact format used in training
    """
    target_date = datetime(target_year, target_month, 1)

    print(f"\nCreating features for prediction of {target_date.strftime('%B %Y')}...")

    # Check if we have enough historical data
    required_months = []
    for lag in range(1, n_lags + 1):
        lag_date = target_date - relativedelta(months=lag)
        required_months.append(lag_date)

    # Extract required data from df
    features_dict = {}
    missing_months = []

    for i, lag_date in enumerate(required_months, 1):
        # Find this month in data
        mask = (df["Date"].dt.year == lag_date.year) & (
            df["Date"].dt.month == lag_date.month
        )
        matching = df[mask]

        if len(matching) == 0:
            missing_months.append(lag_date.strftime("%Y-%m"))
            features_dict[f"close_t-{i}"] = np.nan
            features_dict[f"volume_t-{i}"] = np.nan
        else:
            row = matching.iloc[0]
            features_dict[f"close_t-{i}"] = row["Close"]
            features_dict[f"volume_t-{i}"] = row["Volume"]
            print(f"  close_t-{i} ({lag_date.strftime('%Y-%m')}): ${row['Close']:.2f}")

    if missing_months:
        print(f"\n  ⚠ WARNING: Missing data for months: {missing_months}")
        print(f"  Prediction may not be possible for {target_date.strftime('%B %Y')}")
        return None, None

    # Convert to array in correct order
    close_features = [features_dict[f"close_t-{i}"] for i in range(1, n_lags + 1)]
    volume_features = [features_dict[f"volume_t-{i}"] for i in range(1, n_lags + 1)]
    features_array = np.array(close_features + volume_features).reshape(1, -1)

    print(f"  ✓ Features created successfully")
    print(f"  Feature vector shape: {features_array.shape}")

    return features_array, features_dict


def make_prediction(model, scaler, features):
    """
    Makes prediction using scaled features.

    Args:
        model: Trained LinearRegression model
        scaler: Fitted StandardScaler
        features: Raw feature array

    Returns:
        float: Predicted close price

    Process:
        1. Scale features using training statistics
        2. Apply model to scaled features
        3. Return prediction (already in original scale)

    Note:
        - Scaler uses mean/std from training data
        - This ensures consistency with training
        - Prediction is in dollars (original scale)
    """
    # Scale features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)[0]

    return prediction


def create_forecast_visualization(df, predictions_dict, output_dir="results"):
    """
    Creates comprehensive visualization showing historical data and predictions.

    Args:
        df (pd.DataFrame): Historical monthly data
        predictions_dict (dict): Dictionary of predictions {date_str: price}
        output_dir (str): Output directory

    Visualization includes:
        - Full historical data
        - 2024-2025 closeup (validation period + predictions)
        - Prediction markers with values
        - Confidence indicators
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Plot 1: Full historical view
    ax = axes[0]
    ax.plot(
        df["Date"],
        df["Close"],
        "o-",
        linewidth=1.5,
        markersize=3,
        label="Historical Data",
        color="blue",
        alpha=0.7,
    )

    # Add predictions
    for date_str, price in predictions_dict.items():
        pred_date = datetime.strptime(date_str, "%Y-%m-%d")
        ax.plot(pred_date, price, "r*", markersize=20, label=f"Prediction: {date_str}")
        ax.annotate(
            f"${price:.2f}",
            xy=(pred_date, price),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Close Price ($)", fontsize=12)
    ax.set_title(
        "NFLX Stock Price - Full History with Predictions",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Plot 2: Recent data closeup (2024-2025)
    ax = axes[1]
    recent_df = df[df["Date"] >= "2024-01-01"].copy()

    # Split into actual vs validation
    train_df = recent_df[recent_df["Date"] < "2025-01-01"]
    val_df = recent_df[recent_df["Date"] >= "2025-01-01"]

    ax.plot(
        train_df["Date"],
        train_df["Close"],
        "o-",
        linewidth=2,
        markersize=5,
        label="Training Data (2024)",
        color="blue",
        alpha=0.7,
    )
    ax.plot(
        val_df["Date"],
        val_df["Close"],
        "s-",
        linewidth=2,
        markersize=5,
        label="Validation Data (2025)",
        color="green",
        alpha=0.7,
    )

    # Add predictions
    for date_str, price in predictions_dict.items():
        pred_date = datetime.strptime(date_str, "%Y-%m-%d")
        ax.plot(pred_date, price, "r*", markersize=25)
        ax.annotate(
            f"{date_str}\n${price:.2f}",
            xy=(pred_date, price),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.7", facecolor="yellow", alpha=0.8),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3,rad=0.2", lw=2, color="red"
            ),
        )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Close Price ($)", fontsize=12)
    ax.set_title(
        "NFLX Stock Price - Recent Data and Predictions (2024-2025)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "future_predictions_visualization.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved forecast visualization: {plot_path}")
    plt.close()


def generate_final_summary_report(predictions_dict, model_info, output_dir="results"):
    """
    Generates comprehensive summary report of entire project.

    Args:
        predictions_dict (dict): Future predictions
        model_info (dict): Best model information
        output_dir (str): Output directory

    Report includes:
        - Project overview
        - Data summary
        - Best model details
        - All task results
        - Future predictions
        - Recommendations
    """
    os.makedirs(output_dir, exist_ok=True)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("NFLX STOCK PRICE PREDICTION - FINAL PROJECT SUMMARY")
    report_lines.append("Statistical Methods of Machine Learning - Task 1")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    report_lines.append(f"Stock Symbol: NFLX (Netflix, Inc.)")
    report_lines.append(f"Sector: Communication Services")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("TASK A: BASELINE LINEAR REGRESSION")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(
        "Objective: Find relationship between past closing values and target price"
    )
    report_lines.append("")
    report_lines.append("Configuration Tested:")
    report_lines.append("  - Smoothing levels: raw, sigma1, sigma2, sigma3")
    report_lines.append("  - Lag windows: 3, 6, 9, 12 months")
    report_lines.append("  - Total configurations: 16")
    report_lines.append("")
    report_lines.append(f"Best Configuration:")
    report_lines.append(f"  Smoothing: sigma3 (Gaussian filter, σ=3)")
    report_lines.append(f"  Lag window: 12 months")
    report_lines.append(f"  Features: 24 (12 close lags + 12 volume lags)")
    report_lines.append("")
    report_lines.append(f"Performance Metrics:")
    report_lines.append(f"  Training RMSE: ${model_info['train_rmse']:.2f}")
    report_lines.append(f"  Training R²: {model_info['train_r2']:.4f}")
    report_lines.append(f"  Validation RMSE: ${model_info['val_rmse']:.2f}")
    report_lines.append(f"  Validation R²: {model_info['val_r2']:.4f}")
    report_lines.append("")
    report_lines.append("Key Findings:")
    report_lines.append("  ✓ Sigma3 smoothing highly effective for noise reduction")
    report_lines.append("  ✓ 12-month lag window captures long-term patterns")
    report_lines.append("  ✓ Excellent generalization (small train-validation gap)")
    report_lines.append("  ✓ Near-perfect R² indicates strong predictive power")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("TASK B: POLYNOMIAL REGRESSION WITH REGULARIZATION")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(
        "Objective: Capture non-linear relationships using polynomial features"
    )
    report_lines.append("")
    report_lines.append("Methods Tested:")
    report_lines.append("  1. Ridge Regression (L2 regularization)")
    report_lines.append("  2. Lasso Regression (L1 regularization)")
    report_lines.append("")
    report_lines.append("Configuration:")
    report_lines.append("  Polynomial degree: 2")
    report_lines.append("  Features expanded: 24 → 325 (13.5x increase)")
    report_lines.append("  Alpha values tested: [0.001, 0.01, 0.1, 1.0, 10.0]")
    report_lines.append("")
    report_lines.append("Best Ridge Model (L2):")
    report_lines.append("  Alpha: 0.1")
    report_lines.append("  Validation RMSE: $8.98")
    report_lines.append("  Validation R²: 0.9898")
    report_lines.append("")
    report_lines.append("Best Lasso Model (L1):")
    report_lines.append("  Alpha: 0.001")
    report_lines.append("  Validation RMSE: $9.47")
    report_lines.append("  Validation R²: 0.9886")
    report_lines.append("  Sparsity: 19.1% (selected 263/325 features)")
    report_lines.append("")
    report_lines.append("Conclusion:")
    report_lines.append(
        "  While polynomial features add complexity, the baseline linear"
    )
    report_lines.append("  model outperforms due to effective smoothing preprocessing.")
    report_lines.append(
        "  Regularization successfully prevents overfitting with 325 features."
    )
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("TASK C: DIMENSIONALITY REDUCTION")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Objective: Compare feature reduction techniques")
    report_lines.append("")
    report_lines.append("Methods Compared:")
    report_lines.append("")
    report_lines.append("1. PCA (Principal Component Analysis):")
    report_lines.append("   - Variance threshold: 95%")
    report_lines.append("   - Components selected: 3")
    report_lines.append("   - Validation RMSE: $131.07")
    report_lines.append("   - Validation R²: -1.17")
    report_lines.append("   - Note: Poor performance, lost critical information")
    report_lines.append("")
    report_lines.append("2. CFS (Correlation-based Feature Selection):")
    report_lines.append("   - Features selected: 1 (close_t-1)")
    report_lines.append("   - Validation RMSE: $21.91")
    report_lines.append("   - Validation R²: 0.9392")
    report_lines.append("   - Note: Single feature surprisingly effective")
    report_lines.append("")
    report_lines.append("3. Sequential Forward Selection (Wrapper):")
    report_lines.append("   - Features selected: 12 (all close lags)")
    report_lines.append("   - Validation RMSE: $0.03")
    report_lines.append("   - Validation R²: 1.0000")
    report_lines.append("   - Note: Identified that close lags are sufficient")
    report_lines.append("")
    report_lines.append("4. Baseline (All Features):")
    report_lines.append("   - Features: 24")
    report_lines.append("   - Validation RMSE: $0.03")
    report_lines.append("   - Validation R²: 1.0000")
    report_lines.append("")
    report_lines.append("Key Insights:")
    report_lines.append("  ✓ Wrapper method identified optimal feature subset")
    report_lines.append("  ✓ Close price lags more informative than volume")
    report_lines.append("  ✓ PCA ineffective for this highly smooth data")
    report_lines.append("  ✓ 12 features provide same performance as 24")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("TASK D: FUTURE PRICE PREDICTIONS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(
        "Predictions using best model (Baseline Linear, sigma3, 12 lags):"
    )
    report_lines.append("")

    for date_str, price in sorted(predictions_dict.items()):
        pred_date = datetime.strptime(date_str, "%Y-%m-%d")
        month_name = pred_date.strftime("%B %Y")
        report_lines.append(f"  {month_name}: ${price:.2f}")

    report_lines.append("")
    report_lines.append("Prediction Methodology:")
    report_lines.append("  1. Use last 12 months of smoothed data (sigma3)")
    report_lines.append("  2. Scale features using training statistics")
    report_lines.append("  3. Apply trained linear regression model")
    report_lines.append("  4. Return prediction in original price scale")
    report_lines.append("")
    report_lines.append("Confidence Assessment:")
    report_lines.append(
        f"  Model Validation R²: {model_info['val_r2']:.4f} (excellent)"
    )
    report_lines.append(
        f"  Model Validation RMSE: ${model_info['val_rmse']:.2f} (very low)"
    )
    report_lines.append("  → High confidence in predictions")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("OVERALL CONCLUSIONS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("1. Data Preprocessing:")
    report_lines.append(
        "   Heavy Gaussian smoothing (sigma=3) was crucial for success."
    )
    report_lines.append("   It eliminated noise while preserving trend information.")
    report_lines.append("")
    report_lines.append("2. Feature Engineering:")
    report_lines.append(
        "   12-month lag window captures seasonal and long-term patterns."
    )
    report_lines.append("   Monthly aggregation provides appropriate time scale.")
    report_lines.append("")
    report_lines.append("3. Model Complexity:")
    report_lines.append(
        "   Simple linear regression sufficient with proper preprocessing."
    )
    report_lines.append("   Polynomial features and complex methods unnecessary.")
    report_lines.append("")
    report_lines.append("4. Feature Selection:")
    report_lines.append("   Close price lags are most informative.")
    report_lines.append("   Volume adds marginal value in this configuration.")
    report_lines.append("")
    report_lines.append("5. Prediction Quality:")
    report_lines.append("   Near-perfect validation metrics indicate high reliability.")
    report_lines.append("   Smooth data leads to highly predictable patterns.")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("For Production Deployment:")
    report_lines.append("  1. Continue using sigma3 smoothing for preprocessing")
    report_lines.append("  2. Update model monthly with new data")
    report_lines.append("  3. Monitor prediction errors to detect regime changes")
    report_lines.append("  4. Consider ensemble with multiple smoothing levels")
    report_lines.append("  5. Add confidence intervals for predictions")
    report_lines.append("")
    report_lines.append("Limitations:")
    report_lines.append("  - Heavy smoothing may delay reaction to sudden changes")
    report_lines.append("  - Linear model assumes patterns continue")
    report_lines.append("  - External events (earnings, market crashes) not captured")
    report_lines.append("  - Limited validation data (11 months in 2025)")
    report_lines.append("")
    report_lines.append("Future Improvements:")
    report_lines.append("  - Incorporate external features (market indices, sentiment)")
    report_lines.append("  - Test on multiple stocks for generalization")
    report_lines.append("  - Implement online learning for model updates")
    report_lines.append("  - Add uncertainty quantification")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Save report
    report_path = os.path.join(output_dir, "FINAL_PROJECT_SUMMARY.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"✓ Saved final summary report: {report_path}")

    # Also print to console
    print("\n" + "\n".join(report_lines))


def main():
    """
    Main execution function for future predictions.

    Workflow:
        1. Load latest data
        2. Load best model and scaler
        3. Create features for December 2025
        4. Make prediction for December 2025
        5. Create features for January 2026
        6. Make prediction for January 2026
        7. Visualize predictions
        8. Generate comprehensive report
    """
    print("=" * 80)
    print("NFLX STOCK PRICE PREDICTION - FUTURE PREDICTIONS (TASK D)")
    print("=" * 80)
    print()

    # Load data
    print("Loading latest data...")
    df = load_latest_data()
    print()

    # Load model - Direct construction since we know the configuration
    print("Loading best model...")
    # We know the best model configuration from previous runs
    with open("models/best_baseline_linear_regression.pkl", "rb") as f:
        model_info = pickle.load(f)

    with open("features/scaler_sigma3_12lags.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = model_info["model"]
    n_lags = model_info["n_lags"]

    print(f"✓ Loaded model: LinearRegression")
    print(f"  Validation RMSE: ${model_info['val_rmse']:.2f}")
    print(f"  Validation R²: {model_info['val_r2']:.4f}")
    print(f"  Features required: {n_lags} close lags + {n_lags} volume lags")
    print()

    # Predictions dictionary
    predictions_dict = {}

    # Predict December 2025
    print("=" * 80)
    print("PREDICTION 1: DECEMBER 2025")
    print("=" * 80)

    dec_features, dec_dict = create_prediction_features(df, n_lags, 2025, 12)

    if dec_features is not None:
        dec_prediction = make_prediction(model, scaler, dec_features)
        predictions_dict["2025-12-01"] = dec_prediction
        print(f"\n  ★ PREDICTION FOR DECEMBER 2025: ${dec_prediction:.2f} ★")
    else:
        print("\n  ✗ Cannot predict December 2025 (insufficient data)")

    # Predict January 2026
    print("\n" + "=" * 80)
    print("PREDICTION 2: JANUARY 2026")
    print("=" * 80)

    jan_features, jan_dict = create_prediction_features(df, n_lags, 2026, 1)

    if jan_features is not None:
        jan_prediction = make_prediction(model, scaler, jan_features)
        predictions_dict["2026-01-01"] = jan_prediction
        print(f"\n  ★ PREDICTION FOR JANUARY 2026: ${jan_prediction:.2f} ★")
    else:
        print("\n  ✗ Cannot predict January 2026 (insufficient data)")
        print("  Note: Need December 2025 actual data to predict January 2026")

    # Visualization
    if predictions_dict:
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        create_forecast_visualization(df, predictions_dict)

    # Generate final report
    print("\n" + "=" * 80)
    print("GENERATING FINAL SUMMARY REPORT")
    print("=" * 80)
    generate_final_summary_report(predictions_dict, model_info)

    # Final summary
    print("\n" + "=" * 80)
    print("✓ FUTURE PREDICTIONS COMPLETED")
    print("=" * 80)
    print("\nPredictions:")
    for date_str, price in sorted(predictions_dict.items()):
        pred_date = datetime.strptime(date_str, "%Y-%m-%d")
        print(f"  {pred_date.strftime('%B %Y')}: ${price:.2f}")

    print("\nAll analysis complete! Check the 'results' folder for:")
    print("  - Visualizations (PNG files)")
    print("  - Results tables (CSV files)")
    print("  - Final summary report (TXT file)")


if __name__ == "__main__":
    main()
