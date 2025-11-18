"""
ΒΗΜΑ 1: ΣΥΛΛΟΓΗ ΔΕΔΟΜΕΝΩΝ ΚΑΙ ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ΓΙΑ ΠΡΟΒΛΕΨΗ ΤΙΜΩΝ ΜΕΤΟΧΩΝ NFLX
=============================================================================

STEP 1: DATA ACQUISITION AND PREPROCESSING FOR NFLX STOCK PRICE PREDICTION
===========================================================================

Αυτό το script εκτελεί τις ακόλουθες λειτουργίες:
This script performs the following operations:

1. Ανακτά ιστορικά ημερήσια δεδομένα μετοχών από Alpha Vantage API για NFLX (Netflix)
   Fetches historical daily stock data from Alpha Vantage API for NFLX (Netflix)

2. Μετατρέπει ημερήσια δεδομένα σε μηνιαίους μέσους όρους (τιμή κλεισίματος και όγκος)
   Converts daily data to monthly averages (close price and volume)

3. Εφαρμόζει Gaussian smoothing με διαφορετικές τιμές sigma για μείωση θορύβου
   Applies Gaussian smoothing with different sigma values to reduce noise

4. Αποθηκεύει τα επεξεργασμένα δεδομένα σε αρχεία CSV για περαιτέρω ανάλυση
   Saves the processed data to CSV files for further analysis

Συγγραφέας (Author): Statistical Methods of Machine Learning - Task 1
Σύμβολο Μετοχής (Stock Symbol): NFLX (Netflix, Inc.)
Τομέας (Sector): Communication Services
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
    Φορτώνει το Alpha Vantage API key από το αρχείο .env.
    Loads the Alpha Vantage API key from the .env file.

    Returns:
        str: API key για την υπηρεσία Alpha Vantage (API key for Alpha Vantage service)
    """
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("api_key="):
                return line.strip().split("=")[1]
    raise ValueError("API key not found in .env file")


def fetch_stock_data(symbol, api_key):
    """
    Ανακτά ημερήσια ιστορικά δεδομένα μετοχών από Alpha Vantage API.
    Fetches daily historical stock data from Alpha Vantage API.

    Args:
        symbol (str): Σύμβολο μετοχής (Stock ticker symbol) (e.g., 'NFLX')
        api_key (str): Alpha Vantage API key

    Returns:
        dict: JSON απόκριση που περιέχει χρονοσειρές δεδομένων
              JSON response containing time series data

    Σημειώσεις (Notes):
        - Χρησιμοποιεί TIME_SERIES_DAILY για ημερήσια δεδομένα OHLCV
          Uses TIME_SERIES_DAILY function to get daily OHLCV data
        - outputsize=full ανακτά 20+ χρόνια ιστορικών δεδομένων
          outputsize=full retrieves 20+ years of historical data
        - Δωρεάν tier έχει όρια (5 κλήσεις/λεπτό, 500/ημέρα)
          Free tier has rate limits (5 API calls per minute, 500 per day)
    """
    print(
        f"Ανακτώνται ιστορικά δεδομένα για (Fetching historical data for) {symbol}..."
    )
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
    Μετατρέπει την JSON απόκριση του Alpha Vantage σε pandas DataFrame με ημερήσια δεδομένα.
    Converts Alpha Vantage JSON response to a pandas DataFrame with daily data.

    Παράμετροι (Args):
        json_data (dict): JSON απόκριση από το Alpha Vantage API
                          (JSON response from Alpha Vantage API)

    Επιστρέφει (Returns):
        pd.DataFrame: DataFrame με στήλες [Date, Open, High, Low, Close, Volume]
                      ταξινομημένο χρονολογικά (από παλιότερο σε νεότερο)
                      (DataFrame with columns [Date, Open, High, Low, Close, Volume]
                      sorted chronologically (oldest to newest))

    Σημειώσεις (Notes):
        - Μετατρέπει string τιμές στους κατάλληλους αριθμητικούς τύπους
          (Converts string values to appropriate numeric types)
        - Διαχειρίζεται αυτόματα την ανάλυση ημερομηνιών
          (Handles date parsing automatically)
        - Ταξινομεί δεδομένα χρονολογικά για ανάλυση χρονοσειρών
          (Sorts data chronologically for time series analysis)
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
        f"Μετατράπηκε σε DataFrame (Converted to DataFrame): {len(df)} ημερήσιες εγγραφές (daily records) από (from) {df['Date'].min().date()} έως (to) {df['Date'].max().date()}"
    )

    return df


def convert_to_monthly_averages(daily_df):
    """
    Μετατρέπει ημερήσια δεδομένα μετοχών σε μηνιαίους μέσους όρους.
    Converts daily stock data to monthly averages.

    Παράμετροι (Args):
        daily_df (pd.DataFrame): DataFrame με ημερήσια δεδομένα μετοχών
                                  (DataFrame with daily stock data)

    Επιστρέφει (Returns):
        pd.DataFrame: DataFrame με μηνιαίους μέσους όρους [Year, Month, Close, Volume]
                      (DataFrame with monthly averages [Year, Month, Close, Volume])

    Σημειώσεις (Notes):
        - Ομαδοποιεί δεδομένα ανά έτος και μήνα
          (Groups data by year and month)
        - Υπολογίζει μέσο όρο τιμών κλεισίματος και όγκων για κάθε μήνα
          (Calculates mean of close prices and volumes for each month)
        - Αυτό μειώνει τον θόρυβο και δημιουργεί κατάλληλη χρονική κλίμακα για πρόβλεψη
          (This reduces noise and creates appropriate time scale for prediction)
        - Κάθε μήνας γίνεται ένα σημείο δεδομένων για το μοντέλο παλινδρόμησης
          (Each month becomes one data point for the regression model)
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
        f"Μετατράπηκε σε μηνιαίους μέσους όρους (Converted to monthly averages): {len(monthly_df)} μήνες (months) από (from) {monthly_df['Year'].min()}-{monthly_df['Month'].min()} έως (to) {monthly_df['Year'].max()}-{monthly_df['Month'].max()}"
    )

    return monthly_df


def apply_gaussian_smoothing(data, sigma):
    """
    Εφαρμόζει Gaussian φίλτρο για εξομάλυνση δεδομένων χρονοσειράς και μείωση θορύβου.
    Applies Gaussian filter to smooth time series data and reduce noise.

    Παράμετροι (Args):
        data (np.array or pd.Series): Δεδομένα χρονοσειράς προς εξομάλυνση
                                       (Time series data to smooth)
        sigma (float): Τυπική απόκλιση του Gaussian kernel
                       (Standard deviation of the Gaussian kernel)
                       Μεγαλύτερο sigma = περισσότερη εξομάλυνση
                       (Higher sigma = more smoothing)

    Επιστρέφει (Returns):
        np.array: Εξομαλυμένα δεδομένα (Smoothed data)

    Σημειώσεις (Notes):
        - Το Gaussian φίλτρο είναι ένας σταθμισμένος κινητός μέσος με Gaussian βάρη
          (Gaussian filter is a weighted moving average with Gaussian weights)
        - Βοηθά στη μείωση θορύβου υψηλής συχνότητας διατηρώντας τις γενικές τάσεις
          (Helps reduce high-frequency noise while preserving overall trends)
        - sigma=1: ελαφριά εξομάλυνση (light smoothing)
        - sigma=2: μέτρια (moderate)
        - sigma=3: έντονη (heavy)
        - Πολλή εξομάλυνση μπορεί να αφαιρέσει σημαντικά μοτίβα
          (Too much smoothing can remove important patterns)

    Μαθηματικό Υπόβαθρο (Mathematical Background):
        Το Gaussian kernel έχει βάρη (The Gaussian kernel has weights):
        w(x) = exp(-x²/(2σ²)) / √(2πσ²)
        Κάθε σημείο αντικαθίσταται από σταθμισμένο μέσο όρο γειτονικών σημείων
        (Each point is replaced by a weighted average of nearby points)
    """
    return gaussian_filter1d(data, sigma=sigma)


def save_data_versions(monthly_df, output_dir="data"):
    """
    Αποθηκεύει πολλαπλές εκδόσεις των δεδομένων με διαφορετικές επιλογές προεπεξεργασίας.
    Saves multiple versions of the data with different preprocessing options.

    Παράμετροι (Args):
        monthly_df (pd.DataFrame): Μηνιαία δεδομένα μέσων όρων
                                    (Monthly averaged data)
        output_dir (str): Κατάλογος αποθήκευσης αρχείων CSV
                          (Directory to save CSV files)

    Επιστρέφει (Returns):
        dict: Λεξικό με διαδρομές αποθηκευμένων αρχείων
              (Dictionary with paths to saved files)

    Σημειώσεις (Notes):
        Αποθηκεύει 4 εκδόσεις για σύγκριση:
        Saves 4 versions to allow comparison:
        1. Ακατέργαστα μηνιαία δεδομένα (χωρίς εξομάλυνση)
           Raw monthly data (no smoothing)
        2. Εξομαλυμένα με sigma=1 (ελαφριά εξομάλυνση)
           Smoothed with sigma=1 (light smoothing)
        3. Εξομαλυμένα με sigma=2 (μέτρια εξομάλυνση)
           Smoothed with sigma=2 (moderate smoothing)
        4. Εξομαλυμένα με sigma=3 (έντονη εξομάλυνση)
           Smoothed with sigma=3 (heavy smoothing)

        Αυτό επιτρέπει τη δοκιμή ποιας προεπεξεργασίας δίνει την καλύτερη απόδοση μοντέλου.
        This allows testing which preprocessing yields best model performance.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    saved_files = {}

    # Save raw monthly data
    raw_path = os.path.join(output_dir, "nflx_monthly_raw.csv")
    monthly_df.to_csv(raw_path, index=False)
    saved_files["raw"] = raw_path
    print(
        f"Αποθηκεύτηκαν ακατέργαστα μηνιαία δεδομένα (Saved raw monthly data) σε (to): {raw_path}"
    )

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
        print(
            f"Αποθηκεύτηκαν εξομαλυμένα δεδομένα (Saved smoothed data) (sigma={sigma}) σε (to): {smoothed_path}"
        )

    return saved_files


def visualize_smoothing_comparison(monthly_df, output_dir="data"):
    """
    Δημιουργεί οπτικοποίηση που συγκρίνει ακατέργαστα δεδομένα με διαφορετικά επίπεδα εξομάλυνσης.
    Creates visualization comparing raw data with different smoothing levels.

    Παράμετροι (Args):
        monthly_df (pd.DataFrame): Μηνιαία δεδομένα μέσων όρων
                                    (Monthly averaged data)
        output_dir (str): Κατάλογος αποθήκευσης γραφήματος
                          (Directory to save plot)

    Σημειώσεις (Notes):
        - Δείχνει πώς οι διαφορετικές τιμές sigma επηρεάζουν τα δεδομένα
          (Shows how different sigma values affect the data)
        - Βοηθά στην επιλογή κατάλληλου επιπέδου εξομάλυνσης
          (Helps in selecting appropriate smoothing level)
        - Πολύ λίγη εξομάλυνση: θορυβώδη δεδομένα, δυσκολότερη εκμάθηση μοτίβων
          (Too little smoothing: noisy data, harder to learn patterns)
        - Πολύ εξομάλυνση: απώλεια σημαντικών διακυμάνσεων
          (Too much smoothing: loss of important variations)
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
    print(
        f"Αποθηκεύτηκε γράφημα σύγκρισης εξομάλυνσης (Saved smoothing comparison plot) σε (to): {plot_path}"
    )
    plt.close()


def main():
    """
    Κύρια συνάρτηση εκτέλεσης που ενορχηστρώνει ολόκληρη τη διαδικασία απόκτησης δεδομένων.
    Main execution function that orchestrates the entire data acquisition pipeline.

    Βήματα Διαδικασίας (Pipeline Steps):
    1. Φόρτωση API key από αρχείο .env
       (Load API key from .env file)
    2. Ανάκτηση ημερήσιων δεδομένων NFLX από Alpha Vantage
       (Fetch daily NFLX data from Alpha Vantage)
    3. Μετατροπή σε pandas DataFrame
       (Convert to pandas DataFrame)
    4. Συγκέντρωση σε μηνιαίους μέσους όρους
       (Aggregate to monthly averages)
    5. Εφαρμογή Gaussian εξομάλυνσης με πολλαπλές τιμές sigma
       (Apply Gaussian smoothing with multiple sigma values)
    6. Αποθήκευση όλων των εκδόσεων σε αρχεία CSV
       (Save all versions to CSV files)
    7. Δημιουργία οπτικοποίησης σύγκρισης επιπέδων εξομάλυνσης
       (Create visualization comparing smoothing levels)
    """
    print("=" * 80)
    print("ΠΡΟΒΛΕΨΗ ΤΙΜΗΣ ΜΕΤΟΧΗΣ NFLX - ΑΠΟΚΤΗΣΗ ΔΕΔΟΜΕΝΩΝ")
    print("NFLX STOCK PRICE PREDICTION - DATA ACQUISITION")
    print("=" * 80)
    print()

    # Configuration
    SYMBOL = "NFLX"
    OUTPUT_DIR = "data"

    try:
        # Βήμα 1: Φόρτωση API key (Step 1: Load API key)
        api_key = load_api_key()
        print(f"✓ Φορτώθηκε API key από αρχείο .env (Loaded API key from .env file)")
        print()

        # Βήμα 2: Ανάκτηση δεδομένων από API (Step 2: Fetch data from API)
        json_data = fetch_stock_data(SYMBOL, api_key)
        print()

        # Βήμα 3: Μετατροπή σε ημερήσιο DataFrame (Step 3: Convert to daily DataFrame)
        daily_df = convert_to_daily_dataframe(json_data)
        print()

        # Βήμα 4: Μετατροπή σε μηνιαίους μέσους όρους (Step 4: Convert to monthly averages)
        monthly_df = convert_to_monthly_averages(daily_df)
        print()

        # Βήμα 5: Αποθήκευση διαφορετικών εκδόσεων (Step 5: Save different versions)
        print("Αποθήκευση αρχείων δεδομένων... (Saving data files...)")
        saved_files = save_data_versions(monthly_df, OUTPUT_DIR)
        print()

        # Βήμα 6: Δημιουργία οπτικοποίησης (Step 6: Create visualization)
        print("Δημιουργία οπτικοποίησης... (Creating visualization...)")
        visualize_smoothing_comparison(monthly_df, OUTPUT_DIR)
        print()

        # Εμφάνιση συνοπτικών στατιστικών (Display summary statistics)
        print("=" * 80)
        print("ΣΥΝΟΨΗ ΔΕΔΟΜΕΝΩΝ (DATA SUMMARY)")
        print("=" * 80)
        print(f"Σύμβολο Μετοχής (Stock Symbol): {SYMBOL}")
        print(f"Σύνολο Μηνών (Total Months): {len(monthly_df)}")
        print(
            f"Εύρος Ημερομηνιών (Date Range): {monthly_df['Date'].min().date()} έως (to) {monthly_df['Date'].max().date()}"
        )
        print(f"\nΣτατιστικά Τιμής Κλεισίματος (Close Price Statistics):")
        print(f"  Μέσος Όρος (Mean): ${monthly_df['Close'].mean():.2f}")
        print(f"  Τυπική Απόκλιση (Std Dev): ${monthly_df['Close'].std():.2f}")
        print(f"  Ελάχιστο (Min): ${monthly_df['Close'].min():.2f}")
        print(f"  Μέγιστο (Max): ${monthly_df['Close'].max():.2f}")
        print(f"\nΣτατιστικά Όγκου (Volume Statistics):")
        print(f"  Μέσος Όρος (Mean): {monthly_df['Volume'].mean():.0f}")
        print(f"  Τυπική Απόκλιση (Std Dev): {monthly_df['Volume'].std():.0f}")
        print(f"  Ελάχιστο (Min): {monthly_df['Volume'].min():.0f}")
        print(f"  Μέγιστο (Max): {monthly_df['Volume'].max():.0f}")
        print()

        print("=" * 80)
        print("✓ Η ΑΠΟΚΤΗΣΗ ΔΕΔΟΜΕΝΩΝ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ")
        print("✓ DATA ACQUISITION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nΑποθηκευμένα Αρχεία (Saved Files):")
        for key, path in saved_files.items():
            print(f"  - {key}: {path}")
        print()
        print(
            "Επόμενο Βήμα (Next Step): Εκτέλεση (Run) step2_feature_engineering.py για δημιουργία χαρακτηριστικών υστέρησης (to create lagged features)"
        )

    except Exception as e:
        print(f"\n❌ ΣΦΑΛΜΑ (ERROR): {e}")
        raise


if __name__ == "__main__":
    main()
