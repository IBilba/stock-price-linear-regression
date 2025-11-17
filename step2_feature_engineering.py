"""
STEP 2: FEATURE ENGINEERING FOR TIME SERIES REGRESSION
=======================================================

This script creates lagged features for stock price prediction:
1. Loads preprocessed monthly data (with different smoothing levels)
2. Creates lagged features: close_t-1, close_t-2, ..., volume_t-1, volume_t-2, ...
3. Implements proper chronological train/validation split (pre-2025 vs 2025)
4. Applies StandardScaler fitted only on training data
5. Saves feature matrices for model training

Key Considerations:
- Time series requires chronological splitting (no random shuffling)
- Scaling must be fit on training data only to avoid data leakage
- Different lag windows (N=3,6,9,12) tested to find optimal lookback period
- Each lag configuration saved separately for comparison

Author: Statistical Methods of Machine Learning - Task 1
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_monthly_data(file_path):
    """
    Loads preprocessed monthly stock data from CSV.

    Args:
        file_path (str): Path to CSV file with monthly data

    Returns:
        pd.DataFrame: Monthly data with Date, Year, Month, Close, Volume columns

    Notes:
        - Date column is parsed as datetime
        - Ensures chronological ordering
    """
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"Loaded {len(df)} months of data from {file_path}")
    return df


def create_lagged_features(df, n_lags, target_col="Close"):
    """
    Creates lagged features for time series prediction.

    Args:
        df (pd.DataFrame): DataFrame with Close and Volume columns
        n_lags (int): Number of months to look back (e.g., 6 means use previous 6 months)
        target_col (str): Column to predict (default: 'Close')

    Returns:
        pd.DataFrame: DataFrame with lagged features and target

    Feature Creation Logic:
    - close_t-1: Average close price 1 month ago (most recent)
    - close_t-2: Average close price 2 months ago
    - ...
    - close_t-N: Average close price N months ago (oldest)
    - volume_t-1, volume_t-2, ..., volume_t-N: Same for volume
    - target: Close price we want to predict (current month)

    Example with n_lags=3:
        To predict Close for month t, we use:
        - close_t-1 (last month's close)
        - close_t-2 (2 months ago close)
        - close_t-3 (3 months ago close)
        - volume_t-1, volume_t-2, volume_t-3

    Notes:
        - First N rows are dropped because they don't have enough history
        - This is why we need long historical data (we have 283 months)
        - Larger N means fewer training samples but potentially more information
    """
    # Create a copy to avoid modifying original
    features_df = df[["Date", "Year", "Month", "Close", "Volume"]].copy()

    # Create lagged features for Close price
    for lag in range(1, n_lags + 1):
        features_df[f"close_t-{lag}"] = features_df["Close"].shift(lag)

    # Create lagged features for Volume
    for lag in range(1, n_lags + 1):
        features_df[f"volume_t-{lag}"] = features_df["Volume"].shift(lag)

    # The target is the current month's Close price
    features_df["target"] = features_df["Close"]

    # Drop rows with NaN values (first n_lags rows don't have enough history)
    features_df = features_df.dropna().reset_index(drop=True)

    print(f"  Created {n_lags} lagged features for Close and Volume")
    print(f"  Total features: {n_lags * 2} (Close + Volume)")
    print(f"  Usable samples after dropping NaN: {len(features_df)}")

    return features_df


def split_train_validation_chronological(df, validation_year=2025):
    """
    Splits data chronologically into training and validation sets.

    Args:
        df (pd.DataFrame): DataFrame with Year column and features
        validation_year (int): Year to use for validation (default: 2025)

    Returns:
        tuple: (train_df, val_df) - Training and validation DataFrames

    Critical for Time Series:
        - MUST use chronological split, never random split
        - Training data: All data before validation_year
        - Validation data: Data from validation_year onward
        - This simulates real-world scenario: train on past, predict future
        - Random split would leak future information into training (data leakage)

    Why This Matters:
        - In production, we only have past data to train on
        - We need to validate on truly unseen future data
        - Random splits make models appear better than they actually are
    """
    train_df = df[df["Year"] < validation_year].copy()
    val_df = df[df["Year"] >= validation_year].copy()

    print(f"\n  Chronological Split:")
    print(f"    Training: Years < {validation_year} → {len(train_df)} samples")
    print(f"    Validation: Years >= {validation_year} → {len(val_df)} samples")

    # Display date ranges
    if len(train_df) > 0:
        print(
            f"    Training range: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}"
        )
    if len(val_df) > 0:
        print(
            f"    Validation range: {val_df['Date'].min().date()} to {val_df['Date'].max().date()}"
        )

    return train_df, val_df


def prepare_X_y(df, n_lags):
    """
    Extracts feature matrix X and target vector y from DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with lagged features and target
        n_lags (int): Number of lags used (to identify feature columns)

    Returns:
        tuple: (X, y, feature_names, metadata_df)
            - X: Feature matrix (numpy array)
            - y: Target vector (numpy array)
            - feature_names: List of feature column names
            - metadata_df: DataFrame with Date, Year, Month for tracking

    Feature Ordering:
        [close_t-1, close_t-2, ..., close_t-N, volume_t-1, volume_t-2, ..., volume_t-N]
    """
    # Feature columns: all close_t-* and volume_t-* columns
    close_features = [f"close_t-{i}" for i in range(1, n_lags + 1)]
    volume_features = [f"volume_t-{i}" for i in range(1, n_lags + 1)]
    feature_names = close_features + volume_features

    # Extract features and target
    X = df[feature_names].values
    y = df["target"].values

    # Keep metadata for later analysis
    metadata_df = df[["Date", "Year", "Month"]].copy()

    return X, y, feature_names, metadata_df


def scale_features(X_train, X_val):
    """
    Applies standardization (z-score normalization) to features.

    Args:
        X_train (np.array): Training feature matrix
        X_val (np.array): Validation feature matrix

    Returns:
        tuple: (X_train_scaled, X_val_scaled, scaler)

    Standardization Formula:
        z = (x - μ) / σ
        where μ = mean, σ = standard deviation

    Critical Data Leakage Prevention:
        1. Fit scaler ONLY on training data (compute μ and σ from training)
        2. Transform training data using these statistics
        3. Transform validation data using SAME statistics (from training)
        4. Never fit scaler on validation or test data

    Why Scaling Matters:
        - Close prices (~$200-300) and Volume (~5M) have very different scales
        - Linear regression without scaling will be dominated by large-scale features
        - Standardization gives all features equal importance initially
        - Model coefficients become interpretable (1 unit = 1 std deviation)

    Example:
        If training close_t-1 has mean=$250, std=$50:
        - Training value $300 → scaled to (300-250)/50 = 1.0
        - Validation value $320 → scaled to (320-250)/50 = 1.4
        (Note: Uses training mean/std even for validation)
    """
    scaler = StandardScaler()

    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform validation data using training statistics
    X_val_scaled = scaler.transform(X_val)

    print(f"\n  Feature Scaling Applied:")
    print(f"    Method: StandardScaler (z-score normalization)")
    print(f"    Training shape: {X_train_scaled.shape}")
    print(f"    Validation shape: {X_val_scaled.shape}")
    print(f"    Features now have mean≈0, std≈1 (based on training data)")

    return X_train_scaled, X_val_scaled, scaler


def save_feature_set(
    output_dir,
    smoothing_type,
    n_lags,
    X_train,
    X_val,
    y_train,
    y_val,
    train_metadata,
    val_metadata,
    feature_names,
    scaler,
):
    """
    Saves all components of a feature set for later use.

    Args:
        output_dir (str): Directory to save files
        smoothing_type (str): Type of smoothing ('raw', 'sigma1', etc.)
        n_lags (int): Number of lags used
        X_train, X_val: Feature matrices (scaled)
        y_train, y_val: Target vectors
        train_metadata, val_metadata: Date/Year/Month information
        feature_names: List of feature names
        scaler: Fitted StandardScaler object

    Saves:
        - features_{smoothing}_{lags}lags.npz: All numpy arrays
        - scaler_{smoothing}_{lags}lags.pkl: Fitted scaler for future use
        - metadata_{smoothing}_{lags}lags.csv: Date information

    Why Save Everything:
        - Feature matrices for model training
        - Scaler for transforming future predictions
        - Metadata for plotting and analysis
        - Different configurations for comparison
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create filename prefix
    prefix = f"{smoothing_type}_{n_lags}lags"

    # Save numpy arrays
    npz_path = os.path.join(output_dir, f"features_{prefix}.npz")
    np.savez(
        npz_path,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_names=feature_names,
    )

    # Save scaler
    scaler_path = os.path.join(output_dir, f"scaler_{prefix}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save metadata
    metadata_path = os.path.join(output_dir, f"metadata_{prefix}.csv")
    metadata_df = pd.concat(
        [train_metadata.assign(split="train"), val_metadata.assign(split="validation")],
        ignore_index=True,
    )
    metadata_df.to_csv(metadata_path, index=False)

    print(f"  ✓ Saved feature set: {prefix}")
    print(f"    - Features: {npz_path}")
    print(f"    - Scaler: {scaler_path}")
    print(f"    - Metadata: {metadata_path}")


def visualize_train_val_split(df, n_lags, validation_year=2025, output_dir="features"):
    """
    Creates visualization showing the train/validation split.

    Args:
        df (pd.DataFrame): Full dataset with features
        n_lags (int): Number of lags (for title)
        validation_year (int): Year used for split
        output_dir (str): Directory to save plot

    Notes:
        - Shows closing price over time
        - Highlights train vs validation periods
        - Useful for understanding temporal distribution of data
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    train_df = df[df["Year"] < validation_year]
    val_df = df[df["Year"] >= validation_year]

    # Plot training data
    ax.plot(
        train_df["Date"],
        train_df["target"],
        "o-",
        label=f"Training ({len(train_df)} months)",
        alpha=0.7,
        markersize=3,
        linewidth=1.5,
        color="blue",
    )

    # Plot validation data
    ax.plot(
        val_df["Date"],
        val_df["target"],
        "o-",
        label=f"Validation ({len(val_df)} months)",
        alpha=0.7,
        markersize=3,
        linewidth=1.5,
        color="orange",
    )

    # Add vertical line at split point
    if len(val_df) > 0:
        split_date = val_df["Date"].min()
        ax.axvline(
            x=split_date,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Split Point ({split_date.date()})",
        )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Close Price ($)", fontsize=12)
    ax.set_title(
        f"NFLX Stock Price - Train/Validation Split (n_lags={n_lags})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"train_val_split_{n_lags}lags.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved split visualization: {plot_path}")
    plt.close()


def process_configuration(data_file, smoothing_type, n_lags, output_dir="features"):
    """
    Processes one complete configuration: specific smoothing level and lag count.

    Args:
        data_file (str): Path to monthly data CSV
        smoothing_type (str): Type of smoothing (for naming)
        n_lags (int): Number of months to look back
        output_dir (str): Output directory

    Pipeline:
        1. Load data
        2. Create lagged features
        3. Split train/validation chronologically
        4. Extract X, y matrices
        5. Scale features
        6. Save everything
    """
    print(f"\n{'='*80}")
    print(f"Processing: {smoothing_type}, n_lags={n_lags}")
    print(f"{'='*80}")

    # Load data
    df = load_monthly_data(data_file)

    # Create lagged features
    features_df = create_lagged_features(df, n_lags)

    # Split chronologically
    train_df, val_df = split_train_validation_chronological(features_df)

    # Check if we have validation data
    if len(val_df) == 0:
        print(f"  ⚠ WARNING: No validation data for year 2025 with {n_lags} lags")
        print(f"  This configuration will be skipped.")
        return

    # Extract X, y
    X_train, y_train, feature_names, train_metadata = prepare_X_y(train_df, n_lags)
    X_val, y_val, _, val_metadata = prepare_X_y(val_df, n_lags)

    # Scale features
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)

    # Save everything
    save_feature_set(
        output_dir,
        smoothing_type,
        n_lags,
        X_train_scaled,
        X_val_scaled,
        y_train,
        y_val,
        train_metadata,
        val_metadata,
        feature_names,
        scaler,
    )

    # Visualize split (only once per lag value, using raw data)
    if smoothing_type == "raw":
        visualize_train_val_split(features_df, n_lags, output_dir=output_dir)

    print(f"\n✓ Configuration complete: {smoothing_type}, {n_lags} lags")


def main():
    """
    Main execution function.

    Testing Strategy:
        - 4 smoothing levels: raw, sigma=1, sigma=2, sigma=3
        - 4 lag configurations: N=3, 6, 9, 12 months
        - Total: 16 feature configurations to compare

    This comprehensive approach allows finding optimal:
        1. Preprocessing (smoothing level)
        2. Lookback window (number of lags)

    Trade-offs:
        - More lags (N=12): More information but fewer samples, more parameters
        - Fewer lags (N=3): Less information but more samples, fewer parameters
        - More smoothing: Less noise but may remove important patterns
        - No smoothing: All patterns preserved but more noise
    """
    print("=" * 80)
    print("NFLX STOCK PRICE PREDICTION - FEATURE ENGINEERING")
    print("=" * 80)
    print()

    # Configuration
    DATA_DIR = "data"
    OUTPUT_DIR = "features"

    # Data files with different smoothing levels
    data_files = {
        "raw": os.path.join(DATA_DIR, "nflx_monthly_raw.csv"),
        "sigma1": os.path.join(DATA_DIR, "nflx_monthly_smoothed_sigma1.csv"),
        "sigma2": os.path.join(DATA_DIR, "nflx_monthly_smoothed_sigma2.csv"),
        "sigma3": os.path.join(DATA_DIR, "nflx_monthly_smoothed_sigma3.csv"),
    }

    # Lag configurations to test
    lag_configurations = [3, 6, 9, 12]

    print("Testing Configurations:")
    print(f"  Smoothing levels: {list(data_files.keys())}")
    print(f"  Lag configurations: {lag_configurations}")
    print(f"  Total configurations: {len(data_files) * len(lag_configurations)}")
    print()

    # Process all combinations
    for smoothing_type, data_file in data_files.items():
        for n_lags in lag_configurations:
            try:
                process_configuration(data_file, smoothing_type, n_lags, OUTPUT_DIR)
            except Exception as e:
                print(f"\n❌ Error processing {smoothing_type}, {n_lags} lags: {e}")
                continue

    print("\n" + "=" * 80)
    print("✓ FEATURE ENGINEERING COMPLETED")
    print("=" * 80)
    print(f"\nAll feature sets saved to: {OUTPUT_DIR}/")
    print("\nNext Step: Run step3_baseline_linear_regression.py to train models")


if __name__ == "__main__":
    main()
