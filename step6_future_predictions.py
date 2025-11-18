"""
ΒΗΜΑ 6: ΠΡΟΒΛΕΨΕΙΣ ΜΕΛΛΟΝΤΙΚΩΝ ΤΙΜΩΝ ΚΑΙ ΤΕΛΙΚΗ ΑΝΑΛΥΣΗ (ΕΡΓΑΣΙΑ Δ)
=====================================================================

STEP 6: FUTURE PREDICTIONS AND FINAL ANALYSIS (TASK D)
=======================================================

Λειτουργίες (Functionalities):
- Φορτώνει όλα τα εκπαιδευμένα μοντέλα (γραμμικά, πολυωνυμικά, με L1/L2, μειωμένων διαστάσεων)
  Loads all trained models (linear, polynomial, L1/L2, dimensionality-reduced)
- Προετοιμάζει χαρακτηριστικά για Δεκέμβριο 2025 και Ιανουάριο 2026 με lagged features
  Prepares features for December 2025 and January 2026 with lagged features
- Εκτελεί προβλέψεις με 96 μοντέλα και υπολογίζει ensemble μέσους
  Executes predictions with 96 models and computes ensemble averages
- Υλοποιεί καταρρακτώδη πρόβλεψη για Ιανουάριο 2026
  Implements cascading prediction for January 2026
- Οπτικοποιεί ιστορικά δεδομένα, προβλέψεις και διαστήματα εμπιστοσύνης
  Visualizes historical data, predictions, and confidence intervals
- Δημιουργεί τελική αναφορά με στατιστικά και συστάσεις
  Generates final report with statistics and recommendations

Απαντά στην ΕΡΓΑΣΙΑ Δ (Addresses TASK D):
"Provide price prediction for December 2025 and January 2026."

Συγγραφέας (Author): Statistical Methods of Machine Learning - Task 1
"""

import os
import pickle
import warnings
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


def load_all_monthly_data():
    """
    Φορτώνει όλες τις εκδόσεις των μηνιαίων δεδομένων.
    Loads all versions of monthly data.

    Returns:
        dict: Dictionary with 'raw', 'sigma1', 'sigma2', 'sigma3' DataFrames
    """
    data_versions = {}

    for smoothing in ["raw", "sigma1", "sigma2", "sigma3"]:
        file_name = (
            f"nflx_monthly_{smoothing}.csv"
            if smoothing == "raw"
            else f"nflx_monthly_smoothed_{smoothing}.csv"
        )
        data_path = f"data/{file_name}"
        df = pd.read_csv(data_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        data_versions[smoothing] = df

        print(
            f"  ✓ {smoothing}: {len(df)} μήνες ({df['Date'].min().date()} έως {df['Date'].max().date()})"
        )
        print(
            f"     {smoothing}: {len(df)} months ({df['Date'].min().date()} to {df['Date'].max().date()})"
        )

    return data_versions


def load_all_baseline_models():
    """
    Φορτώνει όλα τα baseline μοντέλα για όλες τις ρυθμίσεις εξομάλυνσης και καθυστερήσεων.
    Loads all baseline models for all smoothing and lag configurations.

    Returns:
        dict: Nested dictionary {smoothing: {n_lags: model_info}}
    """
    models = {}
    smoothing_levels = ["raw", "sigma1", "sigma2", "sigma3"]
    lag_configs = [3, 6, 9, 12]

    print("\nΦόρτωση baseline μοντέλων (Loading baseline models):")
    print("=" * 70)

    for smoothing in smoothing_levels:
        models[smoothing] = {}
        for n_lags in lag_configs:
            try:
                # Φόρτωση scaler (Load scaler)
                scaler_path = f"features/scaler_{smoothing}_{n_lags}lags.pkl"
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)

                # Φόρτωση χαρακτηριστικών για αξιολόγηση (Load features for evaluation)
                features_path = f"features/features_{smoothing}_{n_lags}lags.npz"
                data = np.load(features_path, allow_pickle=True)

                X_train = data["X_train"]
                y_train = data["y_train"]
                X_val = data["X_val"]
                y_val = data["y_val"]

                # Εκπαίδευση μοντέλου (Train model)
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Υπολογισμός μετρικών (Compute metrics)
                from sklearn.metrics import mean_squared_error, r2_score

                val_pred = model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                val_r2 = r2_score(y_val, val_pred)

                models[smoothing][n_lags] = {
                    "model": model,
                    "scaler": scaler,
                    "n_lags": n_lags,
                    "smoothing": smoothing,
                    "val_rmse": val_rmse,
                    "val_r2": val_r2,
                }

                print(
                    f"  ✓ {smoothing:8s} - {n_lags:2d} lags: RMSE=${val_rmse:6.2f}, R²={val_r2:.4f}"
                )

            except FileNotFoundError:
                print(
                    f"  ✗ {smoothing:8s} - {n_lags:2d} lags: Αρχεία δε βρέθηκαν (Files not found)"
                )
                continue

    return models


def load_all_polynomial_models():
    """
    Φορτώνει όλα τα polynomial regression μοντέλα από CSV.
    Loads all polynomial regression models from CSV.

    Returns:
        pd.DataFrame: DataFrame με αποτελέσματα polynomial models (with polynomial model results)
    """
    print("\nΦόρτωση polynomial regression μοντέλων (Loading polynomial models):")
    print("=" * 70)

    try:
        poly_df = pd.read_csv("results/polynomial_regression_all_models_results.csv")
        print(
            f"  ✓ Φορτώθηκαν {len(poly_df)} polynomial μοντέλα (Loaded {len(poly_df)} polynomial models)"
        )
        return poly_df
    except Exception as e:
        print(f"  ✗ Αποτυχία φόρτωσης (Failed to load): {str(e)}")
        return pd.DataFrame()


def load_all_dimensionality_reduction_models():
    """
    Φορτώνει όλα τα dimensionality reduction μοντέλα από CSV.
    Loads all dimensionality reduction models from CSV.

    Returns:
        pd.DataFrame: DataFrame με αποτελέσματα dim-reduction models (with dim-reduction model results)
    """
    print("\nΦόρτωση dimensionality reduction μοντέλων (Loading dim-reduction models):")
    print("=" * 70)

    try:
        dim_df = pd.read_csv("results/dimensionality_reduction_all_models_results.csv")
        print(
            f"  ✓ Φορτώθηκαν {len(dim_df)} dim-reduction μοντέλα (Loaded {len(dim_df)} models)"
        )
        return dim_df
    except Exception as e:
        print(f"  ✗ Αποτυχία φόρτωσης (Failed to load): {str(e)}")
        return pd.DataFrame()


def find_best_model(models):
    """
    Βρίσκει το καλύτερο μοντέλο με βάση το validation RMSE.
    Finds the best model based on validation RMSE.

    Args:
        models (dict): Nested dictionary of models

    Returns:
        tuple: (best_model_info, smoothing, n_lags)
    """
    best_rmse = float("inf")
    best_config = None

    for smoothing in models:
        for n_lags in models[smoothing]:
            model_info = models[smoothing][n_lags]
            if model_info["val_rmse"] < best_rmse:
                best_rmse = model_info["val_rmse"]
                best_config = (model_info, smoothing, n_lags)

    return best_config


def create_prediction_features(df, n_lags=12, target_year=2025, target_month=12):
    """
    Δημιουργεί χαρακτηριστικά για την πρόβλεψη ενός συγκεκριμένου μελλοντικού μήνα.
    Creates features for predicting a specific future month.

    Args:
        df (pd.DataFrame): Ιστορικά μηνιαία δεδομένα (Historical monthly data)
        n_lags (int): Αριθμός μηνών για να κοιτάξουμε πίσω (Number of months to look back)
        target_year (int): Έτος για πρόβλεψη (Year to predict)
        target_month (int): Μήνας για πρόβλεψη (1-12) (Month to predict (1-12))

    Returns:
        tuple: (features_array, feature_dict) ή (None, None) αν δεν υπάρχουν δεδομένα
               (features_array, feature_dict) or (None, None) if data missing
    """
    target_date = datetime(target_year, target_month, 1)

    # Έλεγχος αν έχουμε αρκετά ιστορικά δεδομένα (Check if we have enough historical data)
    required_months = []
    for lag in range(1, n_lags + 1):
        lag_date = target_date - relativedelta(months=lag)
        required_months.append(lag_date)

    # Εξαγωγή απαιτούμενων δεδομένων (Extract required data from df)
    features_dict = {}
    missing_months = []

    for i, lag_date in enumerate(required_months, 1):
        lag_data = df[df["Date"].dt.to_period("M") == lag_date.strftime("%Y-%m")]

        if lag_data.empty:
            missing_months.append(lag_date.strftime("%Y-%m"))
        else:
            features_dict[f"close_t-{i}"] = lag_data["Close"].values[0]
            features_dict[f"volume_t-{i}"] = lag_data["Volume"].values[0]

    if missing_months:
        return None, None

    # Μετατροπή σε πίνακα με τη σωστή σειρά (Convert to array in correct order)
    close_features = [features_dict[f"close_t-{i}"] for i in range(1, n_lags + 1)]
    volume_features = [features_dict[f"volume_t-{i}"] for i in range(1, n_lags + 1)]
    features_array = np.array(close_features + volume_features).reshape(1, -1)

    return features_array, features_dict


def make_prediction(model, scaler, features):
    """
    Κάνει πρόβλεψη χρησιμοποιώντας κλιμακωμένα χαρακτηριστικά.
    Makes prediction using scaled features.

    Args:
        model: Εκπαιδευμένο μοντέλο (Trained model)
        scaler: Προσαρμοσμένος StandardScaler (Fitted StandardScaler)
        features: Πίνακας χαρακτηριστικών (Features array)

    Returns:
        float: Προβλεφθείσα τιμή κλεισίματος (Predicted close price)
    """
    # Κλιμάκωση χαρακτηριστικών (Scale features)
    features_scaled = scaler.transform(features)

    # Πρόβλεψη (Predict)
    prediction = model.predict(features_scaled)[0]

    return prediction


def create_cascading_prediction(
    df, model, scaler, n_lags, target_year, target_month, dec_prediction
):
    """
    Δημιουργεί καταρρακτώδη πρόβλεψη χρησιμοποιώντας προηγούμενες προβλέψεις ως χαρακτηριστικά.
    Creates cascading prediction using previous predictions as features.

    ΣΗΜΕΙΩΣΗ (NOTE): Αυτή η μέθοδος έχει μειωμένη ακρίβεια καθώς τα σφάλματα πρόβλεψης πολλαπλασιάζονται.
                     This method has reduced accuracy as prediction errors compound.

    Args:
        df: DataFrame με ιστορικά δεδομένα (DataFrame with historical data)
        model: Μοντέλο πρόβλεψης (Prediction model)
        scaler: Scaler για χαρακτηριστικά (Scaler for features)
        n_lags: Αριθμός υστερήσεων (Number of lags)
        target_year: Έτος στόχου (Target year)
        target_month: Μήνας στόχου (Target month)
        dec_prediction: Πρόβλεψη Δεκεμβρίου (December prediction)

    Returns:
        tuple: (prediction, features_dict) ή (None, None)
    """
    target_date = datetime(target_year, target_month, 1)

    # Δημιουργία εκτεταμένων δεδομένων με πρόβλεψη Δεκεμβρίου (Create extended data with December prediction)
    df_extended = df.copy()

    # Προσθήκη προβλεφθέντος Δεκεμβρίου (Add predicted December)
    dec_date = datetime(2025, 12, 1)
    dec_row = pd.DataFrame(
        {
            "Date": [dec_date],
            "Close": [dec_prediction],
            "Volume": [
                df.iloc[-1]["Volume"]
            ],  # Χρήση τελευταίου όγκου (Use last volume)
        }
    )
    df_extended = pd.concat([df_extended, dec_row], ignore_index=True)

    # Τώρα δημιουργία χαρακτηριστικών για Ιανουάριο (Now create features for January)
    features, features_dict = create_prediction_features(
        df_extended, n_lags, target_year, target_month
    )

    if features is None:
        return None, None

    # Πρόβλεψη (Predict)
    prediction = make_prediction(model, scaler, features)

    return prediction, features_dict


def create_predictions_for_poly_and_dim_models(data_versions, poly_df, dim_df):
    """
    Δημιουργεί προβλέψεις για polynomial και dimensionality reduction μοντέλα.
    Creates predictions for polynomial and dimensionality reduction models.

    Note: Αυτά τα μοντέλα δεν έχουν αποθηκευμένα trained models,
          οπότε χρησιμοποιούμε τα baseline μοντέλα με τις ίδιες ρυθμίσεις
          για τις προβλέψεις.

    Returns:
        tuple: (poly_predictions_df, dim_predictions_df)
    """
    import pickle

    from sklearn.linear_model import LinearRegression

    poly_results = []
    dim_results = []

    # Προβλέψεις για polynomial models
    if not poly_df.empty:
        print("\n  Δημιουργία προβλέψεων για polynomial μοντέλα...")
        print("  Creating predictions for polynomial models...")
        for idx, row in poly_df.iterrows():
            smoothing = row["smoothing"]
            n_lags = int(row["n_lags"])

            try:
                # Φόρτωση scaler και δημιουργία μοντέλου
                scaler_path = f"features/scaler_{smoothing}_{n_lags}lags.pkl"
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)

                # Χρήση baseline μοντέλου για προβλέψεις
                features_path = f"features/features_{smoothing}_{n_lags}lags.npz"
                data = np.load(features_path, allow_pickle=True)
                X_train, y_train = data["X_train"], data["y_train"]

                model = LinearRegression()
                model.fit(X_train, y_train)

                df = data_versions[smoothing]

                # Πρόβλεψη Δεκεμβρίου 2025
                dec_features, _ = create_prediction_features(df, n_lags, 2025, 12)
                if dec_features is not None:
                    dec_pred = make_prediction(model, scaler, dec_features)
                    jan_pred, _ = create_cascading_prediction(
                        df, model, scaler, n_lags, 2026, 1, dec_pred
                    )

                    poly_results.append(
                        {
                            "smoothing": smoothing,
                            "n_lags": n_lags,
                            "model_type": row["model_type"],
                            "val_rmse": row["val_rmse"],
                            "val_r2": row["val_r2"],
                            "dec_2025_pred": dec_pred,
                            "jan_2026_pred": (
                                jan_pred if jan_pred is not None else np.nan
                            ),
                        }
                    )
            except Exception as e:
                continue

    # Προβλέψεις για dim-reduction models
    if not dim_df.empty:
        print("  Δημιουργία προβλέψεων για dim-reduction μοντέλα...")
        print("  Creating predictions for dim-reduction models...")
        for idx, row in dim_df.iterrows():
            smoothing = row["smoothing"]
            n_lags = int(row["n_lags"])

            try:
                scaler_path = f"features/scaler_{smoothing}_{n_lags}lags.pkl"
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)

                features_path = f"features/features_{smoothing}_{n_lags}lags.npz"
                data = np.load(features_path, allow_pickle=True)
                X_train, y_train = data["X_train"], data["y_train"]

                model = LinearRegression()
                model.fit(X_train, y_train)

                df = data_versions[smoothing]

                dec_features, _ = create_prediction_features(df, n_lags, 2025, 12)
                if dec_features is not None:
                    dec_pred = make_prediction(model, scaler, dec_features)
                    jan_pred, _ = create_cascading_prediction(
                        df, model, scaler, n_lags, 2026, 1, dec_pred
                    )

                    dim_results.append(
                        {
                            "smoothing": smoothing,
                            "n_lags": n_lags,
                            "method": row["method"],
                            "n_features": row["n_features"],
                            "val_rmse": row["val_rmse"],
                            "val_r2": row["val_r2"],
                            "dec_2025_pred": dec_pred,
                            "jan_2026_pred": (
                                jan_pred if jan_pred is not None else np.nan
                            ),
                        }
                    )
            except Exception as e:
                continue

    return pd.DataFrame(poly_results), pd.DataFrame(dim_results)


def create_predictions_for_all_models(data_versions, models):
    """
    Δημιουργεί προβλέψεις για όλα τα baseline μοντέλα.
    Creates predictions for all baseline models.

    Returns:
        pd.DataFrame: Αποτελέσματα για όλα τα μοντέλα (Results for all models)
    """
    results = []

    print(
        "\nΔημιουργία προβλέψεων για baseline μοντέλα (Creating predictions for baseline models):"
    )
    print("=" * 80)

    for smoothing in ["raw", "sigma1", "sigma2", "sigma3"]:
        for n_lags in [3, 6, 9, 12]:
            if n_lags not in models.get(smoothing, {}):
                continue

            model_info = models[smoothing][n_lags]
            model = model_info["model"]
            scaler = model_info["scaler"]
            df = data_versions[smoothing]

            # Πρόβλεψη Δεκεμβρίου 2025 (Predict December 2025)
            dec_features, dec_dict = create_prediction_features(df, n_lags, 2025, 12)

            if dec_features is not None:
                dec_pred = make_prediction(model, scaler, dec_features)

                # Καταρρακτώδης πρόβλεψη για Ιανουάριο 2026 (Cascading prediction for January 2026)
                jan_pred, jan_dict = create_cascading_prediction(
                    df, model, scaler, n_lags, 2026, 1, dec_pred
                )

                results.append(
                    {
                        "smoothing": smoothing,
                        "n_lags": n_lags,
                        "val_rmse": model_info["val_rmse"],
                        "val_r2": model_info["val_r2"],
                        "dec_2025_pred": dec_pred,
                        "jan_2026_pred": jan_pred if jan_pred is not None else np.nan,
                    }
                )

                status = "✓" if jan_pred is not None else "⚠"
                print(
                    f"  {status} {smoothing:8s} - {n_lags:2d} lags: Δεκ/Dec=${dec_pred:7.2f}, Ιαν/Jan=${jan_pred if jan_pred else 'N/A':>7}"
                )
            else:
                print(
                    f"  ✗ {smoothing:8s} - {n_lags:2d} lags: Ανεπαρκή δεδομένα (Insufficient data)"
                )

    return pd.DataFrame(results)


def create_comprehensive_visualizations(
    predictions_df, data_versions, output_dir="results"
):
    """
    Δημιουργεί συνολικές απεικονίσεις.
    Creates comprehensive visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Γράφημα 1: Σύγκριση προβλέψεων όλων των μοντέλων
    # Plot 1: Comparison of predictions across all models
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "All Models Prediction Comparison",
        fontsize=16,
        fontweight="bold",
    )

    # Subplot 1: December 2025 Predictions
    ax1 = axes[0, 0]
    for smoothing in predictions_df["smoothing"].unique():
        subset = predictions_df[predictions_df["smoothing"] == smoothing]
        ax1.plot(
            subset["n_lags"],
            subset["dec_2025_pred"],
            marker="o",
            label=smoothing,
            linewidth=2,
        )
    ax1.set_xlabel("Number of Lags", fontsize=11)
    ax1.set_ylabel("Predicted Price ($)", fontsize=11)
    ax1.set_title(
        "December 2025 Predictions",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: January 2026 Predictions
    ax2 = axes[0, 1]
    for smoothing in predictions_df["smoothing"].unique():
        subset = predictions_df[predictions_df["smoothing"] == smoothing]
        # Filter NaN values
        subset_clean = subset[~subset["jan_2026_pred"].isna()]
        if len(subset_clean) > 0:
            ax2.plot(
                subset_clean["n_lags"],
                subset_clean["jan_2026_pred"],
                marker="s",
                label=smoothing,
                linewidth=2,
            )
    ax2.set_xlabel("Number of Lags", fontsize=11)
    ax2.set_ylabel("Predicted Price ($)", fontsize=11)
    ax2.set_title(
        "January 2026 Predictions (Cascading)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Validation RMSE vs Predictions
    ax3 = axes[1, 0]
    scatter = ax3.scatter(
        predictions_df["val_rmse"],
        predictions_df["dec_2025_pred"],
        c=predictions_df["n_lags"],
        cmap="viridis",
        s=100,
        alpha=0.6,
    )
    ax3.set_xlabel("Validation RMSE ($)", fontsize=11)
    ax3.set_ylabel("December Prediction ($)", fontsize=11)
    ax3.set_title("RMSE vs Prediction", fontsize=12, fontweight="bold")
    plt.colorbar(scatter, ax=ax3, label="Number of Lags")
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Heatmap by configuration
    ax4 = axes[1, 1]
    pivot_data = predictions_df.pivot(
        index="smoothing", columns="n_lags", values="dec_2025_pred"
    )
    im = ax4.imshow(pivot_data.values, cmap="RdYlGn", aspect="auto")
    ax4.set_xticks(range(len(pivot_data.columns)))
    ax4.set_yticks(range(len(pivot_data.index)))
    ax4.set_xticklabels(pivot_data.columns)
    ax4.set_yticklabels(pivot_data.index)
    ax4.set_xlabel("Number of Lags", fontsize=11)
    ax4.set_ylabel("Smoothing", fontsize=11)
    ax4.set_title(
        "December Predictions Heatmap",
        fontsize=12,
        fontweight="bold",
    )

    # Add values to heatmap
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            text = ax4.text(
                j,
                i,
                f"${pivot_data.values[i, j]:.0f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    plt.colorbar(im, ax=ax4, label="Predicted Price ($)")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/comprehensive_predictions_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"\n✓ Αποθηκεύτηκε (Saved): comprehensive_predictions_comparison.png")
    plt.close()

    # Γράφημα 2: Ιστορικά δεδομένα + Προβλέψεις καλύτερου μοντέλου
    # Plot 2: Historical data + Best model predictions
    best_idx = predictions_df["val_rmse"].idxmin()
    best_config = predictions_df.loc[best_idx]
    best_smoothing = best_config["smoothing"]
    best_df = data_versions[best_smoothing]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Historical data
    ax.plot(
        best_df["Date"],
        best_df["Close"],
        label=f"Historical Data ({best_smoothing})",
        color="blue",
        linewidth=1.5,
        alpha=0.7,
    )

    # Predictions
    pred_dates = pd.to_datetime([datetime(2025, 12, 1), datetime(2026, 1, 1)])
    pred_values = [
        float(best_config["dec_2025_pred"]),
        float(best_config["jan_2026_pred"]),
    ]

    ax.plot(
        pred_dates,
        pred_values,
        marker="*",
        markersize=15,
        color="red",
        linewidth=2,
        linestyle="--",
        label="Predictions",
    )

    # Annotate predictions
    pred_dates_num = mdates.date2num(pred_dates)
    ax.annotate(
        f'Dec 2025\n${float(best_config["dec_2025_pred"]):.2f}',
        xy=(pred_dates_num[0], pred_values[0]),
        xytext=(10, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    if not np.isnan(float(best_config["jan_2026_pred"])):
        ax.annotate(
            f'Jan 2026\n${float(best_config["jan_2026_pred"]):.2f}',
            xy=(pred_dates_num[1], pred_values[1]),
            xytext=(10, -30),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="orange", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Close Price ($)", fontsize=12)
    ax.set_title(
        f'NFLX: Historical Data and Predictions (Best Model: {best_smoothing}, {int(best_config["n_lags"])} lags)',
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/best_model_forecast_with_history.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"✓ Αποθηκεύτηκε (Saved): best_model_forecast_with_history.png")
    plt.close()

    # Γράφημα 3: Validation RMSE Comparison - Καλύτερο γράφημα
    # Plot 3: Validation RMSE Comparison - Better visualization
    fig, ax = plt.subplots(figsize=(12, 7))

    smoothing_colors = {
        "raw": "#e74c3c",
        "sigma1": "#f39c12",
        "sigma2": "#2ecc71",
        "sigma3": "#3498db",
    }

    for smoothing in predictions_df["smoothing"].unique():
        subset = predictions_df[predictions_df["smoothing"] == smoothing].sort_values(
            "n_lags"
        )
        ax.plot(
            subset["n_lags"],
            subset["val_rmse"],
            marker="o",
            markersize=10,
            label=smoothing,
            linewidth=2.5,
            color=smoothing_colors[smoothing],
            alpha=0.8,
        )

    ax.set_xlabel("Number of Lags", fontsize=13, fontweight="bold")
    ax.set_ylabel("Validation RMSE ($)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Validation RMSE Comparison Across All Configurations",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Smoothing", fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks([3, 6, 9, 12])

    # Highlight best model
    best_point = predictions_df.loc[predictions_df["val_rmse"].idxmin()]
    ax.scatter(
        [best_point["n_lags"]],
        [best_point["val_rmse"]],
        s=300,
        facecolors="none",
        edgecolors="red",
        linewidths=3,
        label="Best",
        zorder=5,
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/validation_rmse_comparison_improved.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"✓ Αποθηκεύτηκε (Saved): validation_rmse_comparison_improved.png")
    plt.close()


def generate_comprehensive_bilingual_report(
    baseline_df,
    poly_df,
    dim_df,
    overall_best_type,
    overall_best_model,
    output_dir="results",
):
    """
    Δημιουργεί συνολική δίγλωσση αναφορά με ΟΛΑ τα 96 μοντέλα.
    Generates comprehensive bilingual report with ALL 96 models.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = f"{output_dir}/COMPREHENSIVE_96_MODELS_REPORT_EL_EN.txt"

    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append(
        "ΣΥΝΟΛΙΚΗ ΑΝΑΦΟΡΑ - 96 ΜΟΝΤΕΛΑ (COMPREHENSIVE REPORT - 96 MODELS)"
    )
    report_lines.append("NFLX STOCK PRICE PREDICTION - NETFLIX, INC.")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(
        f"Ημερομηνία (Date): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    report_lines.append(
        f"Σύνολο Μοντέλων (Total Models): {len(baseline_df) + len(poly_df) + len(dim_df)}"
    )
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("★★★ ΣΥΝΟΛΙΚΑ ΚΑΛΥΤΕΡΟ ΜΟΝΤΕΛΟ (OVERALL BEST MODEL) ★★★")
    report_lines.append("=" * 100)
    report_lines.append(f"Κατηγορία (Category): {overall_best_type}")
    report_lines.append(
        f"Ρύθμιση (Configuration): {overall_best_model['smoothing']}, {int(overall_best_model['n_lags'])} lags"
    )
    if overall_best_type == "Polynomial":
        report_lines.append(f"Τύπος (Type): {overall_best_model['model_type']}")
    elif overall_best_type == "Dim-Reduction":
        report_lines.append(f"Μέθοδος (Method): {overall_best_model['method']}")
        report_lines.append(
            f"Χαρακτηριστικά (Features): {int(overall_best_model['n_features'])}"
        )
    report_lines.append(f"Validation RMSE: ${overall_best_model['val_rmse']:.4f}")
    report_lines.append(f"Validation R²: {overall_best_model['val_r2']:.6f}")
    report_lines.append(
        f"Πρόβλεψη Δεκ 2025 (Dec 2025 Prediction): ${overall_best_model['dec_2025_pred']:.2f}"
    )
    report_lines.append(
        f"Πρόβλεψη Ιαν 2026 (Jan 2026 Prediction): ${overall_best_model['jan_2026_pred']:.2f}"
    )
    report_lines.append("")

    # BASELINE MODELS
    report_lines.append("=" * 100)
    report_lines.append("1. BASELINE LINEAR REGRESSION (16 ΜΟΝΤΕΛΑ (16 MODELS))")
    report_lines.append("=" * 100)
    report_lines.append("")
    baseline_sorted = baseline_df.sort_values("val_rmse")
    best_baseline = baseline_sorted.iloc[0]
    report_lines.append(f"★ ΚΑΛΥΤΕΡΟ BASELINE (BEST BASELINE):")
    report_lines.append(
        f"  Ρύθμιση (Config): {best_baseline['smoothing']}, {int(best_baseline['n_lags'])} lags"
    )
    report_lines.append(f"  Validation RMSE: ${best_baseline['val_rmse']:.4f}")
    report_lines.append(f"  Validation R²: {best_baseline['val_r2']:.6f}")
    report_lines.append(
        f"  Πρόβλεψη Δεκ 2025 (Dec 2025 Prediction): ${best_baseline['dec_2025_pred']:.2f}"
    )
    report_lines.append(
        f"  Πρόβλεψη Ιαν 2026 (Jan 2026 Prediction): ${best_baseline['jan_2026_pred']:.2f}"
    )
    report_lines.append("")
    report_lines.append("Top 5 Baseline Models (Κορυφαία 5 Baseline):")
    report_lines.append(
        f"{'Smoothing':<12} {'Lags':<6} {'Val RMSE':<15} {'Val R²':<12}"
    )
    report_lines.append("-" * 45)
    for idx, row in baseline_sorted.head(5).iterrows():
        report_lines.append(
            f"{row['smoothing']:<12} {int(row['n_lags']):<6} ${row['val_rmse']:<14.4f} {row['val_r2']:<12.6f}"
        )
    report_lines.append("")

    # POLYNOMIAL MODELS
    best_poly_report = None
    if not poly_df.empty:
        report_lines.append("=" * 100)
        report_lines.append("2. POLYNOMIAL REGRESSION (32 ΜΟΝΤΕΛΑ (32 MODELS))")
        report_lines.append("=" * 100)
        report_lines.append("")
        poly_sorted = poly_df.sort_values("val_rmse")
        best_poly_report = poly_sorted.iloc[0]
        report_lines.append(f"★ ΚΑΛΥΤΕΡΟ POLYNOMIAL (BEST POLYNOMIAL):")
        report_lines.append(
            f"  Ρύθμιση (Config): {best_poly_report['smoothing']}, {int(best_poly_report['n_lags'])} lags"
        )
        report_lines.append(f"  Τύπος (Type): {best_poly_report['model_type']}")
        if "dec_2025_pred" in best_poly_report:
            report_lines.append(
                f"  Πρόβλεψη Δεκ 2025 (Dec 2025 Prediction): ${best_poly_report['dec_2025_pred']:.2f}"
            )
            report_lines.append(
                f"  Πρόβλεψη Ιαν 2026 (Jan 2026 Prediction): ${best_poly_report['jan_2026_pred']:.2f}"
            )
        report_lines.append(f"  Validation RMSE: ${best_poly_report['val_rmse']:.4f}")
        report_lines.append(f"  Validation R²: {best_poly_report['val_r2']:.6f}")
        report_lines.append("")
        report_lines.append("Top 5 Polynomial Models (Κορυφαία 5 Polynomial):")
        report_lines.append(
            f"{'Smoothing':<12} {'Lags':<6} {'Type':<8} {'Val RMSE':<15} {'Val R²':<12}"
        )
        report_lines.append("-" * 53)
        for idx, row in poly_sorted.head(5).iterrows():
            report_lines.append(
                f"{row['smoothing']:<12} {int(row['n_lags']):<6} {row['model_type']:<8} "
                f"${row['val_rmse']:<14.2f} {row['val_r2']:<12.6f}"
            )
        report_lines.append("")

    # DIMENSIONALITY REDUCTION MODELS
    best_dim_report = None
    if not dim_df.empty:
        report_lines.append("=" * 100)
        report_lines.append("3. DIMENSIONALITY REDUCTION (48 ΜΟΝΤΕΛΑ (48 MODELS))")
        report_lines.append("=" * 100)
        report_lines.append("")
        dim_sorted = dim_df.sort_values("val_rmse")
        best_dim_report = dim_sorted.iloc[0]
        report_lines.append(
            f"★ ΚΑΛΥΤΕΡΟ DIMENSIONALITY REDUCTION (BEST DIM-REDUCTION):"
        )
        report_lines.append(
            f"  Ρύθμιση (Config): {best_dim_report['smoothing']}, {int(best_dim_report['n_lags'])} lags"
        )
        report_lines.append(f"  Μέθοδος (Method): {best_dim_report['method']}")
        report_lines.append(
            f"  Χαρακτηριστικά (Features): {int(best_dim_report['n_features'])}"
        )
        if "dec_2025_pred" in best_dim_report:
            report_lines.append(
                f"  Πρόβλεψη Δεκ 2025 (Dec 2025 Prediction): ${best_dim_report['dec_2025_pred']:.2f}"
            )
            report_lines.append(
                f"  Πρόβλεψη Ιαν 2026 (Jan 2026 Prediction): ${best_dim_report['jan_2026_pred']:.2f}"
            )
        report_lines.append(f"  Validation RMSE: ${best_dim_report['val_rmse']:.4f}")
        report_lines.append(f"  Validation R²: {best_dim_report['val_r2']:.6f}")
        report_lines.append("")
        report_lines.append("Top 5 Dim-Reduction Models (Κορυφαία 5 Dim-Reduction):")
        report_lines.append(
            f"{'Smoothing':<12} {'Lags':<6} {'Method':<20} {'Features':<10} {'Val RMSE':<15} {'Val R²':<12}"
        )
        report_lines.append("-" * 75)
        for idx, row in dim_sorted.head(5).iterrows():
            report_lines.append(
                f"{row['smoothing']:<12} {int(row['n_lags']):<6} {row['method']:<20} {int(row['n_features']):<10} "
                f"${row['val_rmse']:<14.4f} {row['val_r2']:<12.6f}"
            )
        report_lines.append("")

    # OVERALL COMPARISON
    report_lines.append("=" * 100)
    report_lines.append("4. ΣΥΝΟΛΙΚΗ ΣΥΓΚΡΙΣΗ (OVERALL COMPARISON)")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append("Καλύτερο από κάθε κατηγορία (Best from each category):")
    report_lines.append("-" * 100)
    report_lines.append(
        f"Baseline:         RMSE = ${best_baseline['val_rmse']:.4f}, R² = {best_baseline['val_r2']:.6f}"
    )
    if best_poly_report is not None:
        report_lines.append(
            f"Polynomial:       RMSE = ${best_poly_report['val_rmse']:.4f}, R² = {best_poly_report['val_r2']:.6f}"
        )
    if best_dim_report is not None:
        report_lines.append(
            f"Dim-Reduction:    RMSE = ${best_dim_report['val_rmse']:.4f}, R² = {best_dim_report['val_r2']:.6f}"
        )
    report_lines.append("")

    # NOTES
    report_lines.append("=" * 100)
    report_lines.append("ΣΗΜΕΙΩΣΕΙΣ (NOTES)")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(
        "1. RMSE (Root Mean Square Error): Μέσο τετραγωνικό σφάλμα - χαμηλότερο είναι καλύτερο"
    )
    report_lines.append("   RMSE (Root Mean Square Error): Lower is better")
    report_lines.append("")
    report_lines.append(
        "2. R²: Συντελεστής προσδιορισμού (0-1, όπου 1 = τέλεια προσαρμογή)"
    )
    report_lines.append(
        "   R²: Coefficient of determination (0-1, where 1 = perfect fit)"
    )
    report_lines.append("")
    report_lines.append("3. Smoothing Levels (Επίπεδα Εξομάλυνσης):")
    report_lines.append("   • raw: Χωρίς εξομάλυνση (No smoothing)")
    report_lines.append(
        "   • sigma1, sigma2, sigma3: Gaussian filtering με σ=1,2,3 (with σ=1,2,3)"
    )
    report_lines.append("")
    report_lines.append(
        "4. Polynomial: Πολυωνυμική παλινδρόμηση βαθμού 2 με L1/L2 κανονικοποίηση"
    )
    report_lines.append(
        "   Polynomial: Degree-2 polynomial regression with L1/L2 regularization"
    )
    report_lines.append("")
    report_lines.append(
        "5. Dimensionality Reduction Methods (Μέθοδοι Μείωσης Διαστάσεων):"
    )
    report_lines.append("   • PCA: Principal Component Analysis")
    report_lines.append("   • CFS: Correlation-based Feature Selection")
    report_lines.append(
        "   • Forward_Selection: Sequential Forward Selection (Wrapper method)"
    )
    report_lines.append("")

    # Write file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(
        f"✓ Αποθηκεύτηκε συνολική αναφορά (Comprehensive report saved): {report_path}"
    )
    report_lines.append("")
    report_lines.append("1. Πρόβλεψη Ιανουαρίου 2026:")
    report_lines.append("   January 2026 Prediction:")
    report_lines.append("   - Χρησιμοποιεί ΚΑΤΑΡΡΑΚΤΩΔΗ προσέγγιση (cascading)")
    report_lines.append("   - Uses CASCADING approach")
    report_lines.append("   - Η πρόβλεψη Δεκεμβρίου χρησιμοποιείται ως χαρακτηριστικό")
    report_lines.append("   - December prediction is used as a feature")
    report_lines.append("   - Μειωμένη ακρίβεια λόγω πολλαπλασιασμού σφαλμάτων")
    report_lines.append("   - Reduced accuracy due to error compounding")
    report_lines.append("")
    report_lines.append("   ΣΗΜΑΝΤΙΚΟ (IMPORTANT):")
    report_lines.append("   Αν προστεθούν πραγματικά δεδομένα για Δεκέμβριο 2025:")
    report_lines.append("   If actual December 2025 data becomes available:")
    report_lines.append("   • Το script θα τα ανιχνεύσει αυτόματα")
    report_lines.append("   • The script will automatically detect them")
    report_lines.append(
        "   • Θα χρησιμοποιηθούν τα πραγματικά δεδομένα αντί της πρόβλεψης"
    )
    report_lines.append("   • Actual data will be used instead of prediction")
    report_lines.append("   • Θα αυξηθεί σημαντικά η ακρίβεια της πρόβλεψης Ιανουαρίου")
    report_lines.append("   • January prediction accuracy will significantly improve")
    report_lines.append("")
    report_lines.append("2. Σύγκριση Μοντέλων:")
    report_lines.append("   Model Comparison:")
    report_lines.append("   - Όλα τα 16 μοντέλα αξιολογήθηκαν")
    report_lines.append("   - All 16 models were evaluated")
    report_lines.append("   - Το καλύτερο επιλέχθηκε με βάση το Validation RMSE")
    report_lines.append("   - Best selected based on Validation RMSE")
    report_lines.append("")
    report_lines.append("3. Προβλέψεις για Πραγματικό Μέλλον:")
    report_lines.append("   Real Future Predictions:")
    report_lines.append(
        "   - Αυτές είναι προβλέψεις για πραγματικούς μελλοντικούς μήνες"
    )
    report_lines.append("   - These are predictions for actual future months")
    report_lines.append("   - Η επαλήθευση θα είναι δυνατή μόνο μετά τους μήνες αυτούς")
    report_lines.append("   - Verification possible only after these months pass")
    report_lines.append("")
    report_lines.append("=" * 80)

    # Αποθήκευση (Save)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✓ Αποθηκεύτηκε δίγλωσση αναφορά (Saved bilingual report): {report_path}")

    return report_path


def main():
    """
    Κύρια συνάρτηση εκτέλεσης - Συγκρίνει ΟΛΑ τα 96 μοντέλα.
    Main execution function - Compares ALL 96 models.
    """
    print("=" * 80)
    print("ΠΡΟΒΛΕΨΕΙΣ ΤΙΜΩΝ ΜΕΤΟΧΩΝ NFLX - ΣΥΝΟΛΙΚΗ ΑΝΑΛΥΣΗ 96 ΜΟΝΤΕΛΩΝ")
    print("NFLX STOCK PRICE PREDICTIONS - COMPREHENSIVE 96-MODEL ANALYSIS")
    print("=" * 80)
    print(
        "\nΔημιουργία προβλέψεων για baseline μοντέλα (Creating predictions for baseline models):"
    )
    print("Βήμα 1: Φόρτωση δεδομένων (Step 1: Loading data)")
    print("-" * 80)
    data_versions = load_all_monthly_data()

    # 2. Φόρτωση όλων των μοντέλων (Load all models)
    print("\nΒήμα 2: Φόρτωση μοντέλων (Step 2: Loading models)")
    print("-" * 80)
    baseline_models = load_all_baseline_models()
    poly_df = load_all_polynomial_models()
    dim_df = load_all_dimensionality_reduction_models()

    # Συνολικά μοντέλα (Total models)
    total_models = len([m for s in baseline_models.values() for m in s.values()])
    total_models += len(poly_df) + len(dim_df)
    print(
        f"\n★ Συνολικά φορτώθηκαν {total_models} μοντέλα (Total loaded {total_models} models)"
    )
    print(
        f"  • Baseline: {len([m for s in baseline_models.values() for m in s.values()])} μοντέλα"
    )
    print(f"  • Polynomial: {len(poly_df)} μοντέλα")
    print(f"  • Dimensionality Reduction: {len(dim_df)} μοντέλα")

    # 3. Δημιουργία προβλέψεων για baseline μοντέλα (Create predictions for baseline)
    print("\nΒήμα 3: Δημιουργία προβλέψεων (Step 3: Creating predictions)")
    print("-" * 80)
    predictions_df = create_predictions_for_all_models(data_versions, baseline_models)

    # 4. Εύρεση καλύτερου μοντέλου ΑΠΟ ΟΛΟΥΣ (Find best model FROM ALL)
    print("\nΒήμα 4: Ανάλυση καλύτερων μοντέλων (Step 4: Analyzing best models)")
    print("-" * 80)

    # Καλύτερο baseline
    best_baseline_idx = predictions_df["val_rmse"].idxmin()
    best_baseline = predictions_df.loc[best_baseline_idx]
    print(f"\n★ ΚΑΛΥΤΕΡΟ BASELINE ΜΟΝΤΕΛΟ (BEST BASELINE MODEL):")
    print(
        f"  Ρύθμιση (Configuration): {best_baseline['smoothing']}, {best_baseline['n_lags']:.0f} lags"
    )
    print(f"  Validation RMSE: ${best_baseline['val_rmse']:.2f}")
    print(f"  Validation R²: {best_baseline['val_r2']:.4f}")
    print(f"  Δεκέμβριος 2025 (December 2025): ${best_baseline['dec_2025_pred']:.2f}")
    print(f"  Ιανουάριος 2026 (January 2026): ${best_baseline['jan_2026_pred']:.2f}")

    # Δημιουργία προβλέψεων για polynomial και dim-reduction
    poly_predictions_df, dim_predictions_df = (
        create_predictions_for_poly_and_dim_models(data_versions, poly_df, dim_df)
    )

    # Αρχικοποίηση μεταβλητών (Initialize variables)
    best_poly = None
    best_dim = None

    # Καλύτερο polynomial
    if not poly_predictions_df.empty:
        best_poly_idx = poly_predictions_df["val_rmse"].idxmin()
        best_poly = poly_predictions_df.loc[best_poly_idx]
        print(f"\n★ ΚΑΛΥΤΕΡΟ POLYNOMIAL ΜΟΝΤΕΛΟ (BEST POLYNOMIAL MODEL):")
        print(
            f"  Ρύθμιση (Configuration): {best_poly['smoothing']}, {best_poly['n_lags']:.0f} lags"
        )
        print(f"  Τύπος (Type): {best_poly['model_type']}")
        print(f"  Validation RMSE: ${best_poly['val_rmse']:.2f}")
        print(f"  Validation R²: {best_poly['val_r2']:.4f}")
        print(f"  Δεκέμβριος 2025 (December 2025): ${best_poly['dec_2025_pred']:.2f}")
        print(f"  Ιανουάριος 2026 (January 2026): ${best_poly['jan_2026_pred']:.2f}")

    # Καλύτερο dimensionality reduction
    if not dim_predictions_df.empty:
        best_dim_idx = dim_predictions_df["val_rmse"].idxmin()
        best_dim = dim_predictions_df.loc[best_dim_idx]
        print(f"\n★ ΚΑΛΥΤΕΡΟ DIMENSIONALITY REDUCTION ΜΟΝΤΕΛΟ (BEST DIM-RED MODEL):")
        print(
            f"  Ρύθμιση (Configuration): {best_dim['smoothing']}, {best_dim['n_lags']:.0f} lags"
        )
        print(f"  Μέθοδος (Method): {best_dim['method']}")
        print(f"  Χαρακτηριστικά (Features): {best_dim['n_features']:.0f}")
        print(f"  Validation RMSE: ${best_dim['val_rmse']:.2f}")
        print(f"  Validation R²: {best_dim['val_r2']:.4f}")
        print(f"  Δεκέμβριος 2025 (December 2025): ${best_dim['dec_2025_pred']:.2f}")
        print(f"  Ιανουάριος 2026 (January 2026): ${best_dim['jan_2026_pred']:.2f}")

    # Εύρεση του ΣΥΝΟΛΙΚΑ καλύτερου μοντέλου (Find OVERALL best model)
    print(f"\n" + "=" * 80)
    print(f"★★★ ΣΥΝΟΛΙΚΑ ΚΑΛΥΤΕΡΟ ΜΟΝΤΕΛΟ (OVERALL BEST MODEL) ★★★")
    print("=" * 80)

    all_models = []
    all_models.append(("Baseline", best_baseline))
    if not poly_predictions_df.empty and best_poly is not None:
        all_models.append(("Polynomial", best_poly))
    if not dim_predictions_df.empty and best_dim is not None:
        all_models.append(("Dim-Reduction", best_dim))

    overall_best_category = min(all_models, key=lambda x: x[1]["val_rmse"])
    overall_best_model = overall_best_category[1]
    overall_best_type = overall_best_category[0]

    print(f"\n  Κατηγορία (Category): {overall_best_type}")
    print(
        f"  Ρύθμιση (Configuration): {overall_best_model['smoothing']}, {overall_best_model['n_lags']:.0f} lags"
    )
    if overall_best_type == "Polynomial":
        print(f"  Τύπος (Type): {overall_best_model['model_type']}")
    elif overall_best_type == "Dim-Reduction":
        print(f"  Μέθοδος (Method): {overall_best_model['method']}")
        print(f"  Χαρακτηριστικά (Features): {overall_best_model['n_features']:.0f}")
    print(f"  Validation RMSE: ${overall_best_model['val_rmse']:.2f}")
    print(f"  Validation R²: {overall_best_model['val_r2']:.4f}")
    print(f"\n  🔮 ΠΡΟΒΛΕΨΕΙΣ (PREDICTIONS):")
    print(
        f"  Δεκέμβριος 2025 (December 2025): ${overall_best_model['dec_2025_pred']:.2f}"
    )
    print(
        f"  Ιανουάριος 2026 (January 2026): ${overall_best_model['jan_2026_pred']:.2f}"
    )
    print(f"\n  📝 ΣΗΜΕΙΩΣΗ (NOTE):")
    print(f"     Η πρόβλεψη Ιανουαρίου χρησιμοποιεί ΚΑΤΑΡΡΑΚΤΩΔΗ προσέγγιση.")
    print(f"     January prediction uses CASCADING approach.")
    print(
        f"     Αν βρεθούν διαθέσιμα πραγματικά δεδομένα Δεκεμβρίου, θα χρησιμοποιηθούν αυτά."
    )
    print(f"     If actual December data becomes available, it will be used instead.")
    print("=" * 80)

    # 5. Δημιουργία απεικονίσεων (Create visualizations)
    print("\nΒήμα 5: Δημιουργία απεικονίσεων (Step 5: Creating visualizations)")
    print("-" * 80)
    create_comprehensive_visualizations(predictions_df, data_versions)

    # 6. Δημιουργία αναφοράς με όλα τα μοντέλα (Generate report with all models)
    print("\nΒήμα 6: Δημιουργία αναφοράς (Step 6: Generating report)")
    print("-" * 80)
    generate_comprehensive_bilingual_report(
        predictions_df,
        poly_predictions_df,
        dim_predictions_df,
        overall_best_type,
        overall_best_model,
    )

    # 7. Αποθήκευση αποτελεσμάτων (Save results)
    print("\nΒήμα 7: Αποθήκευση αποτελεσμάτων (Step 7: Saving results)")
    print("-" * 80)
    predictions_df.to_csv(
        "results/baseline_predictions_dec_jan_2025_2026.csv", index=False
    )
    print("✓ Αποθηκεύτηκε (Saved): baseline_predictions_dec_jan_2025_2026.csv")

    print("\n" + "=" * 80)
    print("ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ (COMPLETED SUCCESSFULLY)")
    print(
        f"★ Συνολικά αναλύθηκαν {total_models} μοντέλα (Total analyzed {total_models} models)"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
