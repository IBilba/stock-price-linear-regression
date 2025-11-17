"""
STEP 1: DATA ACQUISITION AND PREPROCESSING FOR NFLX STOCK PRICE PREDICTION
============================================================================

This script performs the following operations:
1. Fetches historical daily stock data from Alpha Vantage API for NFLX (Netflix)
2. Converts daily data to monthly averages (close price and volume)
3. Applies Gaussian smoothing with different sigma values to reduce noise
4. Saves the processed data to CSV files for further analysis

Author: Statistical Methods of Machine Learning - Task 1
Stock Symbol: NFLX (Netflix, Inc.)
Sector: Communication Services
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.ndimage import gaussian_filter1d


# Load API key from .env file
def load_api_key():
    """
    Loads the Alpha Vantage API key from the .env file.

    Returns:
        str: API key for Alpha Vantage service
    """
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("api_key="):
                return line.strip().split("=")[1]
    raise ValueError("API key not found in .env file")


def fetch_stock_data(symbol, api_key):
    """
    Fetches daily historical stock data from Alpha Vantage API.

    Args:
        symbol (str): Stock ticker symbol (e.g., 'NFLX')
        api_key (str): Alpha Vantage API key

    Returns:
        dict: JSON response containing time series data

    Notes:
        - Uses TIME_SERIES_DAILY function to get daily OHLCV data
        - outputsize=full retrieves 20+ years of historical data
        - Free tier has rate limits (5 API calls per minute, 500 per day)
    """
    print(f"Fetching historical data for {symbol}...")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()

        # Check for API errors
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if "Note" in data:
            raise ValueError(f"API Rate Limit: {data['Note']}")

        print(f"Successfully fetched data for {symbol}")
        return data

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch data: {e}")


def convert_to_daily_dataframe(json_data):
    """
    Converts Alpha Vantage JSON response to a pandas DataFrame with daily data.

    Args:
        json_data (dict): JSON response from Alpha Vantage API

    Returns:
        pd.DataFrame: DataFrame with columns [Date, Open, High, Low, Close, Volume]
                      sorted chronologically (oldest to newest)

    Notes:
        - Converts string values to appropriate numeric types
        - Handles date parsing automatically
        - Sorts data chronologically for time series analysis
    """
    time_series = json_data.get("Time Series (Daily)", {})

    if not time_series:
        raise ValueError("No time series data found in API response")

    # Extract data into lists
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    for date, values in time_series.items():
        dates.append(date)
        opens.append(float(values["1. open"]))
        highs.append(float(values["2. high"]))
        lows.append(float(values["3. low"]))
        closes.append(float(values["4. close"]))
        volumes.append(int(values["5. volume"]))

    # Create DataFrame
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }
    )

    # Sort chronologically (oldest first)
    df = df.sort_values("Date").reset_index(drop=True)

    print(
        f"Converted to DataFrame: {len(df)} daily records from {df['Date'].min().date()} to {df['Date'].max().date()}"
    )

    return df


def convert_to_monthly_averages(daily_df):
    """
    Converts daily stock data to monthly averages.

    Args:
        daily_df (pd.DataFrame): DataFrame with daily stock data

    Returns:
        pd.DataFrame: DataFrame with monthly averages [Year, Month, Close, Volume]

    Notes:
        - Groups data by year and month
        - Calculates mean of close prices and volumes for each month
        - This reduces noise and creates appropriate time scale for prediction
        - Each month becomes one data point for the regression model
    """
    # Extract year and month
    daily_df["Year"] = daily_df["Date"].dt.year
    daily_df["Month"] = daily_df["Date"].dt.month

    # Group by year and month, calculate mean
    monthly_df = (
        daily_df.groupby(["Year", "Month"])
        .agg({"Close": "mean", "Volume": "mean"})
        .reset_index()
    )

    # Create a date column for visualization (first day of each month)
    monthly_df["Date"] = pd.to_datetime(monthly_df[["Year", "Month"]].assign(Day=1))

    print(
        f"Converted to monthly averages: {len(monthly_df)} months from {monthly_df['Year'].min()}-{monthly_df['Month'].min()} to {monthly_df['Year'].max()}-{monthly_df['Month'].max()}"
    )

    return monthly_df


def apply_gaussian_smoothing(data, sigma):
    """
    Applies Gaussian filter to smooth time series data and reduce noise.

    Args:
        data (np.array or pd.Series): Time series data to smooth
        sigma (float): Standard deviation of the Gaussian kernel
                      Higher sigma = more smoothing

    Returns:
        np.array: Smoothed data

    Notes:
        - Gaussian filter is a weighted moving average with Gaussian weights
        - Helps reduce high-frequency noise while preserving overall trends
        - sigma=1: light smoothing, sigma=2: moderate, sigma=3: heavy
        - Too much smoothing can remove important patterns

    Mathematical Background:
        The Gaussian kernel has weights: w(x) = exp(-x²/(2σ²)) / √(2πσ²)
        Each point is replaced by a weighted average of nearby points
    """
    return gaussian_filter1d(data, sigma=sigma)


def save_data_versions(monthly_df, output_dir="data"):
    """
    Saves multiple versions of the data with different preprocessing options.

    Args:
        monthly_df (pd.DataFrame): Monthly averaged data
        output_dir (str): Directory to save CSV files

    Returns:
        dict: Dictionary with paths to saved files

    Notes:
        Saves 4 versions to allow comparison:
        1. Raw monthly data (no smoothing)
        2. Smoothed with sigma=1 (light smoothing)
        3. Smoothed with sigma=2 (moderate smoothing)
        4. Smoothed with sigma=3 (heavy smoothing)

    This allows testing which preprocessing yields best model performance.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    saved_files = {}

    # Save raw monthly data
    raw_path = os.path.join(output_dir, "nflx_monthly_raw.csv")
    monthly_df.to_csv(raw_path, index=False)
    saved_files["raw"] = raw_path
    print(f"Saved raw monthly data to: {raw_path}")

    # Save versions with different Gaussian smoothing
    for sigma in [1, 2, 3]:
        # Create copy to avoid modifying original
        smoothed_df = monthly_df.copy()

        # Apply Gaussian filter to Close and Volume
        smoothed_df["Close"] = apply_gaussian_smoothing(
            monthly_df["Close"].values, sigma
        )
        smoothed_df["Volume"] = apply_gaussian_smoothing(
            monthly_df["Volume"].values, sigma
        )

        # Save to CSV
        smoothed_path = os.path.join(
            output_dir, f"nflx_monthly_smoothed_sigma{sigma}.csv"
        )
        smoothed_df.to_csv(smoothed_path, index=False)
        saved_files[f"sigma_{sigma}"] = smoothed_path
        print(f"Saved smoothed data (sigma={sigma}) to: {smoothed_path}")

    return saved_files


def visualize_smoothing_comparison(monthly_df, output_dir="data"):
    """
    Creates visualization comparing raw data with different smoothing levels.

    Args:
        monthly_df (pd.DataFrame): Monthly averaged data
        output_dir (str): Directory to save plot

    Notes:
        - Shows how different sigma values affect the data
        - Helps in selecting appropriate smoothing level
        - Too little smoothing: noisy data, harder to learn patterns
        - Too much smoothing: loss of important variations
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot Close prices
    axes[0].plot(
        monthly_df["Date"],
        monthly_df["Close"],
        "o-",
        label="Raw",
        alpha=0.6,
        linewidth=1,
    )
    for sigma in [1, 2, 3]:
        smoothed_close = apply_gaussian_smoothing(monthly_df["Close"].values, sigma)
        axes[0].plot(
            monthly_df["Date"], smoothed_close, "-", label=f"Sigma={sigma}", linewidth=2
        )

    axes[0].set_xlabel("Date", fontsize=12)
    axes[0].set_ylabel("Average Close Price ($)", fontsize=12)
    axes[0].set_title(
        "NFLX Monthly Average Close Price - Raw vs Gaussian Smoothing",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    # Plot Volume
    axes[1].plot(
        monthly_df["Date"],
        monthly_df["Volume"],
        "o-",
        label="Raw",
        alpha=0.6,
        linewidth=1,
    )
    for sigma in [1, 2, 3]:
        smoothed_volume = apply_gaussian_smoothing(monthly_df["Volume"].values, sigma)
        axes[1].plot(
            monthly_df["Date"],
            smoothed_volume,
            "-",
            label=f"Sigma={sigma}",
            linewidth=2,
        )

    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].set_ylabel("Average Volume", fontsize=12)
    axes[1].set_title(
        "NFLX Monthly Average Volume - Raw vs Gaussian Smoothing",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "smoothing_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved smoothing comparison plot to: {plot_path}")
    plt.close()


def main():
    """
    Main execution function that orchestrates the entire data acquisition pipeline.

    Pipeline Steps:
    1. Load API key from .env file
    2. Fetch daily NFLX data from Alpha Vantage
    3. Convert to pandas DataFrame
    4. Aggregate to monthly averages
    5. Apply Gaussian smoothing with multiple sigma values
    6. Save all versions to CSV files
    7. Create visualization comparing smoothing levels
    """
    print("=" * 80)
    print("NFLX STOCK PRICE PREDICTION - DATA ACQUISITION")
    print("=" * 80)
    print()

    # Configuration
    SYMBOL = "NFLX"
    OUTPUT_DIR = "data"

    try:
        # Step 1: Load API key
        api_key = load_api_key()
        print(f"✓ Loaded API key from .env file")
        print()

        # Step 2: Fetch data from API
        json_data = fetch_stock_data(SYMBOL, api_key)
        print()

        # Step 3: Convert to daily DataFrame
        daily_df = convert_to_daily_dataframe(json_data)
        print()

        # Step 4: Convert to monthly averages
        monthly_df = convert_to_monthly_averages(daily_df)
        print()

        # Step 5: Save different versions
        print("Saving data files...")
        saved_files = save_data_versions(monthly_df, OUTPUT_DIR)
        print()

        # Step 6: Create visualization
        print("Creating visualization...")
        visualize_smoothing_comparison(monthly_df, OUTPUT_DIR)
        print()

        # Display summary statistics
        print("=" * 80)
        print("DATA SUMMARY")
        print("=" * 80)
        print(f"Stock Symbol: {SYMBOL}")
        print(f"Total Months: {len(monthly_df)}")
        print(
            f"Date Range: {monthly_df['Date'].min().date()} to {monthly_df['Date'].max().date()}"
        )
        print(f"\nClose Price Statistics:")
        print(f"  Mean: ${monthly_df['Close'].mean():.2f}")
        print(f"  Std Dev: ${monthly_df['Close'].std():.2f}")
        print(f"  Min: ${monthly_df['Close'].min():.2f}")
        print(f"  Max: ${monthly_df['Close'].max():.2f}")
        print(f"\nVolume Statistics:")
        print(f"  Mean: {monthly_df['Volume'].mean():.0f}")
        print(f"  Std Dev: {monthly_df['Volume'].std():.0f}")
        print(f"  Min: {monthly_df['Volume'].min():.0f}")
        print(f"  Max: {monthly_df['Volume'].max():.0f}")
        print()

        print("=" * 80)
        print("✓ DATA ACQUISITION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nSaved Files:")
        for key, path in saved_files.items():
            print(f"  - {key}: {path}")
        print()
        print("Next Step: Run step2_feature_engineering.py to create lagged features")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
