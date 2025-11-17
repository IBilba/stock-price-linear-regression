"""
ΒΗΜΑ 2: ΔΗΜΙΟΥΡΓΙΑ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ ΓΙΑ ΧΡΟΝΟΣΕΙΡΕΣ ΠΑΛΙΝΔΡΟΜΗΣΗ
STEP 2: FEATURE ENGINEERING FOR TIME SERIES REGRESSION
======================================================

Αυτό το script δημιουργεί χαρακτηριστικά με υστέρηση για πρόβλεψη τιμών μετοχών:
This script creates lagged features for stock price prediction:

1. Φορτώνει προεπεξεργασμένα μηνιαία δεδομένα (με διαφορετικά επίπεδα εξομάλυνσης)
   Loads preprocessed monthly data (with different smoothing levels)

2. Δημιουργεί χαρακτηριστικά με υστέρηση: close_t-1, close_t-2, ..., volume_t-1, volume_t-2, ...
   Creates lagged features: close_t-1, close_t-2, ..., volume_t-1, volume_t-2, ...

3. Εφαρμόζει σωστή χρονολογική διαίρεση εκπαίδευσης/επικύρωσης (προ-2025 vs 2025)
   Implements proper chronological train/validation split (pre-2025 vs 2025)

4. Εφαρμόζει StandardScaler προσαρμοσμένο μόνο στα δεδομένα εκπαίδευσης
   Applies StandardScaler fitted only on training data

5. Αποθηκεύει πίνακες χαρακτηριστικών για εκπαίδευση μοντέλου
   Saves feature matrices for model training

Σημαντικές Σκέψεις / Key Considerations:
- Χρονοσειρές απαιτούν χρονολογική διαίρεση (όχι τυχαία ανακάτεμα)
  Time series requires chronological splitting (no random shuffling)
- Κλιμάκωση πρέπει να εφαρμόζεται μόνο στα δεδομένα εκπαίδευσης
  Scaling must be fit on training data only to avoid data leakage

Συγγραφέας / Author: Statistical Methods of Machine Learning - Task 1
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_monthly_data(file_path):
    """
    Φορτώνει προεπεξεργασμένα μηνιαία δεδομένα μετοχών από CSV.
    Loads preprocessed monthly stock data from CSV.

    Παράμετροι / Args:
        file_path (str): Διαδρομή προς το αρχείο CSV με μηνιαία δεδομένα
                          Path to CSV file with monthly data

    Επιστρέφει / Returns:
        pd.DataFrame: Μηνιαία δεδομένα με στήλες Date, Year, Month, Close, Volume
                      Monthly data with Date, Year, Month, Close, Volume columns

    Σημειώσεις / Notes:
        - Η στήλη Date αναλύεται ως datetime
          Date column is parsed as datetime
        - Διασφαλίζει χρονολογική ταξινόμηση
          Ensures chronological ordering
    """
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print(
        f"Φορτώθηκαν / Loaded {len(df)} μήνες δεδομένων από / months of data from {file_path}"
    )
    return df


def create_lagged_features(df, n_lags, target_col="Close"):
    """
    Δημιουργεί χαρακτηριστικά με υστέρηση για πρόβλεψη χρονοσειρών.
    Creates lagged features for time series prediction.

    Παράμετροι / Args:
        df (pd.DataFrame): DataFrame με στήλες Close και Volume
                            DataFrame with Close and Volume columns
        n_lags (int): Αριθμός μηνών να κοιτάξουμε πίσω (π.χ., 6 σημαίνει χρήση προηγούμενων 6 μηνών)
                      Number of months to look back (e.g., 6 means use previous 6 months)
        target_col (str): Στήλη προς πρόβλεψη (προεπιλογή: 'Close')
                          Column to predict (default: 'Close')

    Επιστρέφει / Returns:
        pd.DataFrame: DataFrame με χαρακτηριστικά υστέρησης και στόχο
                      DataFrame with lagged features and target

    Λογική Δημιουργίας Χαρακτηριστικών / Feature Creation Logic:
    - close_t-1: Μέση τιμή κλεισίματος 1 μήνα πριν (πιο πρόσφατη)
                 Average close price 1 month ago (most recent)
    - close_t-2: Μέση τιμή κλεισίματος 2 μήνες πριν
                 Average close price 2 months ago
    - ...
    - close_t-N: Μέση τιμή κλεισίματος N μήνες πριν (παλαιότερη)
                 Average close price N months ago (oldest)
    - volume_t-1, volume_t-2, ..., volume_t-N: Το ίδιο για τον όγκο
                                                Same for volume
    - target: Τιμή κλεισίματος που θέλουμε να προβλέψουμε (τρέχων μήνας)
              Close price we want to predict (current month)

    Παράδειγμα με n_lags=3 / Example with n_lags=3:
        Για να προβλέψουμε Close για μήνα t, χρησιμοποιούμε:
        To predict Close for month t, we use:
        - close_t-1 (κλείσιμο προηγούμενου μήνα / last month's close)
        - close_t-2 (κλείσιμο 2 μήνες πριν / 2 months ago close)
        - close_t-3 (κλείσιμο 3 μήνες πριν / 3 months ago close)
        - volume_t-1, volume_t-2, volume_t-3

    Σημειώσεις / Notes:
        - Οι πρώτες N γραμμές διαγράφονται γιατί δεν έχουν αρκετό ιστορικό
          First N rows are dropped because they don't have enough history
        - Γι' αυτό χρειαζόμαστε μεγάλο ιστορικό δεδομένων (έχουμε 283 μήνες)
          This is why we need long historical data (we have 283 months)
        - Μεγαλύτερο N σημαίνει λιγότερα δείγματα εκπαίδευσης αλλά πιθανώς περισσότερη πληροφορία
          Larger N means fewer training samples but potentially more information
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

    print(
        f"  Δημιουργήθηκαν / Created {n_lags} χαρακτηριστικά υστέρησης για / lagged features for Close και / and Volume"
    )
    print(f"  Σύνολο χαρακτηριστικών (Total features): {n_lags * 2} (Close + Volume)")
    print(
        f"  Χρήσιμα δείγματα μετά από διαγραφή NaN / Usable samples after dropping NaN: {len(features_df)}"
    )

    return features_df


def split_train_validation_chronological(df, validation_year=2025):
    """
    Διαχωρίζει χρονολογικά τα δεδομένα σε σύνολα εκπαίδευσης και επικύρωσης.
    Splits data chronologically into training and validation sets.

    Παράμετροι / Args:
        df (pd.DataFrame): DataFrame με στήλη Year και χαρακτηριστικά
                            DataFrame with Year column and features
        validation_year (int): Έτος προς χρήση για επικύρωση (προεπιλογή: 2025)
                               Year to use for validation (default: 2025)

    Επιστρέφει / Returns:
        tuple: (train_df, val_df) - DataFrames εκπαίδευσης και επικύρωσης
                                    Training and validation DataFrames

    Κρίσιμο για Χρονοσειρές / Critical for Time Series:
        - ΠΡΕΠΕΙ να χρησιμοποιούμε χρονολογική διαίρεση, ποτέ τυχαία
          MUST use chronological split, never random split
        - Δεδομένα εκπαίδευσης / Training data: Όλα τα δεδομένα πριν το validation_year
                                                   All data before validation_year
        - Δεδομένα επικύρωσης / Validation data: Δεδομένα από validation_year και μετά
                                                      Data from validation_year onward
        - Αυτό προσομοιώνει πραγματικό σενάριο: εκπαίδευση στο παρελθόν, πρόβλεψη στο μέλλον
          This simulates real-world scenario: train on past, predict future
        - Τυχαία διαίρεση θα διαρρέουσε μελλοντική πληροφορία στην εκπαίδευση (διαρροή δεδομένων)
          Random split would leak future information into training (data leakage)

    Γιατί Αυτό Έχει Σημασία / Why This Matters:
        - Στην παραγωγή, έχουμε μόνο παρελθοντικά δεδομένα για εκπαίδευση
          In production, we only have past data to train on
        - Πρέπει να επικυρώσουμε σε πραγματικά μη επιθεωρημένα μελλοντικά δεδομένα
          We need to validate on truly unseen future data
        - Τυχαίες διαιρέσεις κάνουν τα μοντέλα να φαίνονται καλύτερα απ' ό,τι είναι πραγματικά
          Random splits make models appear better than they actually are
    """
    train_df = df[df["Year"] < validation_year].copy()
    val_df = df[df["Year"] >= validation_year].copy()

    print(f"\n  Χρονολογική Διαίρεση (Chronological Split):")
    print(
        f"    Εκπαίδευση / Training: Έτη / Years < {validation_year} → {len(train_df)} δείγματα / samples"
    )
    print(
        f"    Επικύρωση / Validation: Έτη / Years >= {validation_year} → {len(val_df)} δείγματα / samples"
    )

    # Display date ranges
    if len(train_df) > 0:
        print(
            f"    Εύρος εκπαίδευσης / Training range: {train_df['Date'].min().date()} έως / to {train_df['Date'].max().date()}"
        )
    if len(val_df) > 0:
        print(
            f"    Εύρος επικύρωσης / Validation range: {val_df['Date'].min().date()} έως / to {val_df['Date'].max().date()}"
        )

    return train_df, val_df


def prepare_X_y(df, n_lags):
    """
    Εξάγει πίνακα χαρακτηριστικών X και διάνυσμα στόχων y από DataFrame.
    Extracts feature matrix X and target vector y from DataFrame.

    Παράμετροι / Args:
        df (pd.DataFrame): DataFrame με χαρακτηριστικά υστέρησης και στόχο
                            DataFrame with lagged features and target
        n_lags (int): Αριθμός υστερήσεων που χρησιμοποιήθηκαν (για αναγνώριση στηλών χαρακτηριστικών)
                      Number of lags used (to identify feature columns)

    Επιστρέφει / Returns:
        tuple: (X, y, feature_names, metadata_df)
            - X: Πίνακας χαρακτηριστικών (numpy array) / Feature matrix (numpy array)
            - y: Διάνυσμα στόχων (numpy array) / Target vector (numpy array)
            - feature_names: Λίστα ονομάτων στηλών χαρακτηριστικών
                                         List of feature column names
            - metadata_df: DataFrame με Date, Year, Month για παρακολούθηση
                           DataFrame with Date, Year, Month for tracking

    Σειρά Χαρακτηριστικών / Feature Ordering:
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
    Εφαρμόζει τυποποίηση (z-score κανονικοποίηση) στα χαρακτηριστικά.
    Applies standardization (z-score normalization) to features.

    Παράμετροι / Args:
        X_train (np.array): Πίνακας χαρακτηριστικών εκπαίδευσης / Training feature matrix
        X_val (np.array): Πίνακας χαρακτηριστικών επικύρωσης / Validation feature matrix

    Επιστρέφει / Returns:
        tuple: (X_train_scaled, X_val_scaled, scaler)

    Τύπος Τυποποίησης / Standardization Formula:
        z = (x - μ) / σ
        όπου / where μ = μέσος όρος / mean, σ = τυπική απόκλιση / standard deviation

    Κρίσιμη Πρόληψη Διαρροής Δεδομένων / Critical Data Leakage Prevention:
        1. Προσαρμογή scaler ΜΟΝΟ στα δεδομένα εκπαίδευσης (υπολογισμός μ και σ από εκπαίδευση)
           Fit scaler ONLY on training data (compute μ and σ from training)
        2. Μετασχηματισμός δεδομένων εκπαίδευσης με αυτά τα στατιστικά
           Transform training data using these statistics
        3. Μετασχηματισμός δεδομένων επικύρωσης με τα ΙΔΙΑ στατιστικά (από εκπαίδευση)
           Transform validation data using SAME statistics (from training)
        4. Ποτέ μην προσαρμόζετε scaler σε δεδομένα επικύρωσης ή δοκιμής
           Never fit scaler on validation or test data

    Γιατί Η Κλιμάκωση Έχει Σημασία / Why Scaling Matters:
        - Οι τιμές κλεισίματος (~$200-300) και όγκος (~5M) έχουν πολύ διαφορετικές κλίμακες
          Close prices (~$200-300) and Volume (~5M) have very different scales
        - Γραμμική παλινδρόμηση χωρίς κλιμάκωση θα κυριαρχείται από χαρακτηριστικά μεγάλης κλίμακας
          Linear regression without scaling will be dominated by large-scale features
        - Η τυποποίηση δίνει σε όλα τα χαρακτηριστικά ίση σημασία αρχικά
          Standardization gives all features equal importance initially
        - Οι συντελεστές του μοντέλου γίνονται ερμηνεύσιμες (1 μονάδα = 1 τυπική απόκλιση)
          Model coefficients become interpretable (1 unit = 1 std deviation)

    Παράδειγμα / Example:
        Αν το close_t-1 εκπαίδευσης έχει μέσο=$250, std=$50:
        If training close_t-1 has mean=$250, std=$50:
        - Τιμή εκπαίδευσης / Training value $300 → κλιμακωμένη σε / scaled to (300-250)/50 = 1.0
        - Τιμή επικύρωσης / Validation value $320 → κλιμακωμένη σε / scaled to (320-250)/50 = 1.4
        (Σημείωση / Note: Χρησιμοποιεί mean/std εκπαίδευσης ακόμα και για επικύρωση / Uses training mean/std even for validation)
    """
    scaler = StandardScaler()

    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform validation data using training statistics
    X_val_scaled = scaler.transform(X_val)

    print(f"\n  Κλιμάκωση Χαρακτηριστικών Εφαρμόστηκε / Feature Scaling Applied:")
    print(
        f"    Μέθοδος / Method: StandardScaler (z-score κανονικοποίηση / normalization)"
    )
    print(f"    Μέγεθος εκπαίδευσης / Training shape: {X_train_scaled.shape}")
    print(f"    Μέγεθος επικύρωσης / Validation shape: {X_val_scaled.shape}")
    print(
        f"    Τα χαρακτηριστικά έχουν τώρα mean≈0, std≈1 (βασισμένα σε δεδομένα εκπαίδευσης)"
    )
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
    Αποθηκεύει όλα τα στοιχεία ενός συνόλου χαρακτηριστικών για μεταγενέστερη χρήση.
    Saves all components of a feature set for later use.

    Παράμετροι / Args:
        output_dir (str): Κατάλογος αποθήκευσης αρχείων / Directory to save files
        smoothing_type (str): Τύπος εξομάλυνσης / Type of smoothing ('raw', 'sigma1', etc.)
        n_lags (int): Αριθμός υστερήσεων που χρησιμοποιήθηκαν / Number of lags used
        X_train, X_val: Πίνακες χαρακτηριστικών (κλιμακωμένοι) / Feature matrices (scaled)
        y_train, y_val: Διανύσματα στόχων / Target vectors
        train_metadata, val_metadata: Πληροφορίες Date/Year/Month
                                        Date/Year/Month information
        feature_names: Λίστα ονομάτων χαρακτηριστικών / List of feature names
        scaler: Προσαρμοσμένο StandardScaler αντικείμενο / Fitted StandardScaler object

    Αποθηκεύει / Saves:
        - features_{smoothing}_{lags}lags.npz: Όλοι οι numpy πίνακες / All numpy arrays
        - scaler_{smoothing}_{lags}lags.pkl: Προσαρμοσμένο scaler για μελλοντικές προβλέψεις
                                              Fitted scaler for future use
        - metadata_{smoothing}_{lags}lags.csv: Πληροφορίες ημερομηνιών / Date information

    Γιατί Να Αποθηκεύσουμε Τα Πάντα / Why Save Everything:
        - Πίνακες χαρακτηριστικών για εκπαίδευση μοντέλου / Feature matrices for model training
        - Scaler για μετασχηματισμό μελλοντικών προβλέψεων / Scaler for transforming future predictions
        - Metadata για γραφήματα και ανάλυση / Metadata for plotting and analysis
        - Διαφορετικές ρυθμίσεις για σύγκριση / Different configurations for comparison
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

    print(f"  ✓ Αποθηκεύτηκε σύνολο χαρακτηριστικών (Saved feature set): {prefix}")
    print(f"    - Χαρακτηριστικά (Features): {npz_path}")
    print(f"    - Scaler: {scaler_path}")
    print(f"    - Metadata: {metadata_path}")


def visualize_train_val_split(df, n_lags, validation_year=2025, output_dir="features"):
    """
    Δημιουργεί οπτικοποίηση που δείχνει τη διαίρεση εκπαίδευσης/επικύρωσης.
    Creates visualization showing the train/validation split.

    Παράμετροι / Args:
        df (pd.DataFrame): Πλήρες σύνολο δεδομένων με χαρακτηριστικά / Full dataset with features
        n_lags (int): Αριθμός υστερήσεων (για τίτλο) / Number of lags (for title)
        validation_year (int): Έτος που χρησιμοποιήθηκε για διαίρεση / Year used for split
        output_dir (str): Κατάλογος αποθήκευσης γραφήματος / Directory to save plot

    Σημειώσεις / Notes:
        - Δείχνει την τιμή κλεισίματος στο χρόνο / Shows closing price over time
        - Επισημαίνει περιόδους εκπαίδευσης vs επικύρωσης / Highlights train vs validation periods
        - Χρήσιμο για κατανόηση χρονικής κατανομής δεδομένων
          Useful for understanding temporal distribution of data
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
    print(
        f"  ✓ Αποθηκεύτηκε οπτικοποίηση διαίρεσης / Saved split visualization: {plot_path}"
    )
    plt.close()


def process_configuration(data_file, smoothing_type, n_lags, output_dir="features"):
    """
    Επεξεργάζεται μία ολοκληρωμένη ρύθμιση: συγκεκριμένο επίπεδο εξομάλυνσης και αριθμό υστερήσεων.
    Processes one complete configuration: specific smoothing level and lag count.

    Παράμετροι / Args:
        data_file (str): Διαδρομή προς CSV μηνιαίων δεδομένων / Path to monthly data CSV
        smoothing_type (str): Τύπος εξομάλυνσης (για ονοματοδοσία) / Type of smoothing (for naming)
        n_lags (int): Αριθμός μηνών να κοιτάξουμε πίσω / Number of months to look back
        output_dir (str): Κατάλογος εξόδου / Output directory

    Διαδικασία / Pipeline:
        1. Φόρτωση δεδομένων / Load data
        2. Δημιουργία χαρακτηριστικών υστέρησης / Create lagged features
        3. Διαίρεση εκπαίδευσης/επικύρωσης χρονολογικά / Split train/validation chronologically
        4. Εξαγωγή πινάκων X, y / Extract X, y matrices
        5. Κλιμάκωση χαρακτηριστικών / Scale features
        6. Αποθήκευση όλων / Save everything
    """
    print(f"\n{'='*80}")
    print(f"Επεξεργασία (Processing): {smoothing_type}, n_lags={n_lags}")
    print(f"{'='*80}")

    # Load data
    df = load_monthly_data(data_file)

    # Create lagged features
    features_df = create_lagged_features(df, n_lags)

    # Split chronologically
    train_df, val_df = split_train_validation_chronological(features_df)

    # Check if we have validation data
    if len(val_df) == 0:
        print(
            f"  ⚠ ΠΡΟΕΙΔΟΠΟΙΗΣΗ / WARNING: Δεν υπάρχουν δεδομένα επικύρωσης για το έτος 2025 με / No validation data for year 2025 with {n_lags} υστερήσεις / lags"
        )
        print(f"  Αυτή η ρύθμιση θα παραλειφθεί (This configuration will be skipped).")
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

    print(
        f"\n✓ Ολοκληρώθηκε η ρύθμιση / Configuration complete: {smoothing_type}, {n_lags} υστερήσεις / lags"
    )


def main():
    """
    Κύρια συνάρτηση εκτέλεσης.
    Main execution function.

    Στρατηγική Δοκιμών / Testing Strategy:
        - 4 επίπεδα εξομάλυνσης / smoothing levels: raw, sigma=1, sigma=2, sigma=3
        - 4 ρυθμίσεις υστέρησης / lag configurations: N=3, 6, 9, 12 μήνες / months
        - Σύνολο / Total: 16 ρυθμίσεις χαρακτηριστικών για σύγκριση / feature configurations to compare

    Αυτή η ολοκληρωμένη προσέγγιση επιτρέπει την εύρεση βέλτιστων:
    This comprehensive approach allows finding optimal:
        1. Προεπεξεργασία (επίπεδο εξομάλυνσης) / Preprocessing (smoothing level)
        2. Παράθυρο αναδρομής (αριθμός υστερήσεων) / Lookback window (number of lags)

    Αντισταθμίσεις / Trade-offs:
        - Περισσότερες υστερήσεις (N=12) / More lags (N=12): Περισσότερη πληροφορία αλλά λιγότερα δείγματα, περισσότερες παράμετροι
                                                         More information but fewer samples, more parameters
        - Λιγότερες υστερήσεις (N=3) / Fewer lags (N=3): Λιγότερη πληροφορία αλλά περισσότερα δείγματα, λιγότερες παράμετροι
                                                         Less information but more samples, fewer parameters
        - Περισσότερη εξομάλυνση / More smoothing: Λιγότερος θόρυβος αλλά μπορεί να αφαιρέσει σημαντικά μοτίβα
                                                   Less noise but may remove important patterns
        - Χωρίς εξομάλυνση / No smoothing: Όλα τα μοτίβα διατηρούνται αλλά περισσότερος θόρυβος
                                              All patterns preserved but more noise
    """
    print("=" * 80)
    print("ΠΡΟΒΛΕΨΗ ΤΙΜΗΣ ΜΕΤΟΧΗΣ NFLX - ΔΗΜΙΟΥΡΓΙΑ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ")
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

    print("Δοκιμή Ρυθμίσεων (Testing Configurations):")
    print(f"  Επίπεδα εξομάλυνσης (Smoothing levels): {list(data_files.keys())}")
    print(f"  Ρυθμίσεις υστέρησης (Lag configurations): {lag_configurations}")
    print(
        f"  Συνολικές ρυθμίσεις / Total configurations: {len(data_files) * len(lag_configurations)}"
    )
    print()

    # Process all combinations
    for smoothing_type, data_file in data_files.items():
        for n_lags in lag_configurations:
            try:
                process_configuration(data_file, smoothing_type, n_lags, OUTPUT_DIR)
            except Exception as e:
                print(
                    f"\n❌ Σφάλμα επεξεργασίας / Error processing {smoothing_type}, {n_lags} υστερήσεις / lags: {e}"
                )
                continue

    print("\n" + "=" * 80)
    print("✓ Η ΔΗΜΙΟΥΡΓΙΑ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ ΟΛΟΚΛΗΡΩΘΗΚΕ")
    print("✓ FEATURE ENGINEERING COMPLETED")
    print("=" * 80)
    print(
        f"\nΌλα τα σύνολα χαρακτηριστικών αποθηκεύτηκαν / All feature sets saved στο / to: {OUTPUT_DIR}/"
    )
    print(
        "\nΕπόμενο Βήμα / Next Step: Εκτέλεση / Run step3_baseline_linear_regression.py για εκπαίδευση μοντέλων / to train models"
    )


if __name__ == "__main__":
    main()
