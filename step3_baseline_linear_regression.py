"""
ΒΗΜΑ 3: BASELINE ΓΡΑΜΜΙΚΗ ΠΑΛΙΝΔΡΟΜΗΣΗ (ΕΡΓΑΣΙΑ Α)
STEP 3: BASELINE LINEAR REGRESSION MODELS (TASK A)
==================================================

Αυτό το script εκπαιδεύει απλά μοντέλα γραμμικής παλινδρόμησης:
This script trains simple linear regression models:

1. Φορτώνει όλα τα 16 σύνολα χαρακτηριστικών (4 εξομάλυνση × 4 υστερήσεις)
   Loads all 16 feature sets (4 smoothing levels × 4 lag configurations)

2. Εκπαιδεύει LinearRegression μοντέλο για κάθε ρύθμιση
   Trains LinearRegression model for each configuration

3. Υπολογίζει μετρικές: RMSE, MAE, R² για training και validation sets
   Computes metrics: RMSE, MAE, R² for training and validation sets

4. Εντοπίζει την καλύτερη ρύθμιση βάσει της απόδοσης επικύρωσης
   Identifies best configuration based on validation performance

Αυτό απαντά στην ΕΡΓΑΣΙΑ Α / This addresses TASK A:
"Χρησιμοποιήστε ένα μοντέλο γραμμικής παλινδρόμησης για να βρείτε τη σχέση
μεταξύ παρελθοντικών τιμών κλεισίματος και της επόμενης τιμής."
"Use a linear regression model to find the relationship between past
closing values and the target (next closing price)."

Συγγραφέας / Author: Statistical Methods of Machine Learning - Task 1
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
    Φορτώνει ένα αποθηκευμένο σύνολο χαρακτηριστικών από τον δίσκο.
    Loads a saved feature set from disk.

    Παράμετροι / Args:
        features_path (str): Διαδρομή προς αρχείο .npz που περιέχει χαρακτηριστικά
                              Path to .npz file containing features

    Επιστρέφει / Returns:
        dict: Λεξικό με X_train, X_val, y_train, y_val, feature_names
              Dictionary with X_train, X_val, y_train, y_val, feature_names

    Σημειώσεις / Notes:
        - Τα χαρακτηριστικά είναι ήδη κλιμακωμένα (StandardScaler εφαρμόστηκε)
          Features are already scaled (StandardScaler was applied)
        - Η διαίρεση εκπαίδευσης/επικύρωσης είναι χρονολογική (προ-2025 vs 2025)
          Train/validation split is chronological (pre-2025 vs 2025)
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
    Υπολογίζει μετρικές παλινδρόμησης.
    Computes regression metrics.

    Παράμετροι / Args:
        y_true (np.array): Πραγματικές τιμές στόχων / True target values
        y_pred (np.array): Προβλεπόμενες τιμές στόχων / Predicted target values

    Επιστρέφει / Returns:
        dict: Λεξικό με μετρικές RMSE, MAE, R² / Dictionary with RMSE, MAE, R² metrics

    Επεξηγήσεις Μετρικών / Metric Explanations:

    1. RMSE (Ρίζα Μέσου Τετραγωνικού Σφάλματος / Root Mean Squared Error):
       - Τετραγωνική ρίζα μέσου τετραγωνικών διαφορών / Square root of average squared differences
       - Τύπος / Formula: √(Σ(y_true - y_pred)² / n)
       - Μονάδες / Units: Όπως ο στόχος (δολάρια) / Same as target (dollars)
       - Τιμωρεί μεγάλα σφάλματα περισσότερο από μικρά / Penalizes large errors more than small ones
       - Χαμηλότερο είναι καλύτερο / Lower is better

    2. MAE (Μέσο Απόλυτο Σφάλμα / Mean Absolute Error):
       - Μέσος όρος απόλυτων διαφορών / Average absolute differences
       - Τύπος / Formula: Σ|y_true - y_pred| / n
       - Μονάδες / Units: Όπως ο στόχος (δολάρια) / Same as target (dollars)
       - Πιο ανθεκτικό σε outliers από RMSE / More robust to outliers than RMSE
       - Ευκολότερο να ερμηνευτεί: "κατά μέσο όρο, οι προβλέψεις απέχουν $X"
         Easier to interpret: "on average, predictions are off by $X"
       - Χαμηλότερο είναι καλύτερο / Lower is better

    3. R² (Συντελεστής Προσδιορισμού / Coefficient of Determination):
       - Αναλογία διακύμανσης που εξηγείται από το μοντέλο
         Proportion of variance explained by the model
       - Τύπος / Formula: 1 - (SS_res / SS_tot)
       - Εύρος / Range: (-∞, 1], συνήθως / typically [0, 1]
       - R² = 1: Τέλειες προβλέψεις / Perfect predictions
       - R² = 0: Μοντέλο όχι καλύτερο από πρόβλεψη μέσου όρου / Model no better than predicting mean
       - R² < 0: Μοντέλο χειρότερο από πρόβλεψη μέσου όρου / Model worse than predicting mean
       - Υψηλότερο είναι καλύτερο / Higher is better

    Γιατί Και οι Τρεις Μετρικές / Why All Three Metrics:
        - RMSE δείχνει συνολικό σφάλμα πρόβλεψης με έμφαση σε μεγάλα λάθη
          RMSE shows overall prediction error with emphasis on large mistakes
        - MAE δείχνει τυπικό σφάλμα πρόβλεψης / MAE shows typical prediction error
        - R² δείχνει πόσο καλά το μοντέλο συλλαμβάνει τη διακύμανση (ανεξάρτητο κλίμακας)
          R² shows how well model captures variance (scale-independent)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def train_linear_regression(X_train, y_train, X_val, y_val):
    """
    Εκπαιδεύει ένα απλό μοντέλο γραμμικής παλινδρόμησης.
    Trains a simple linear regression model.

    Παράμετροι / Args:
        X_train: Χαρακτηριστικά εκπαίδευσης (κλιμακωμένα) / Training features (scaled)
        y_train: Στόχοι εκπαίδευσης / Training targets
        X_val: Χαρακτηριστικά επικύρωσης (κλιμακωμένα) / Validation features (scaled)
        y_val: Στόχοι επικύρωσης / Validation targets

    Επιστρέφει / Returns:
        tuple: (model, train_metrics, val_metrics, predictions)

    Μοντέλο Γραμμικής Παλινδρόμησης / Linear Regression Model:
        y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

        όπου / where:
        - y: Προβλεπόμενη τιμή κλεισίματος / Predicted close price
        - β₀: Τομή (όρος πολωσης) / Intercept (bias term)
        - β₁, β₂, ..., βₙ: Συντελεστές για κάθε χαρακτηριστικό / Coefficients for each feature
        - x₁, x₂, ..., xₙ: Τιμές χαρακτηριστικών (τιμές κλεισίματος και όγκων με υστέρηση)
                            Feature values (lagged close prices and volumes)

    Βελτιστοποίηση / Optimization:
        Ελαχιστοποιεί / Minimizes: Σ(y_true - y_pred)²
        Αυτό ονομάζεται Ελάχιστα Τετράγωνα (Ordinary Least Squares - OLS)
        This is called Ordinary Least Squares (OLS)
        Έχει λύση κλειστής μορφής / Has closed-form solution: β = (X'X)⁻¹X'y

    Χωρίς Κανονικοποίηση / No Regularization:
        Η βασική γραμμική παλινδρόμηση δεν έχει όρο ποινής
        Basic linear regression has no penalty term
        Μπορεί να κάνει overfitting αν πολλά χαρακτηριστικά σε σχέση με δείγματα
        Can overfit if too many features relative to samples
        Οι συντελεστές μπορούν να γίνουν πολύ μεγάλοι / Coefficients can become very large
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
    Εξάγει τύπο εξομάλυνσης και n_lags από όνομα αρχείου χαρακτηριστικών.
    Extracts smoothing type and n_lags from feature filename.

    Παράμετροι / Args:
        filename (str): π.χ. / e.g., "features_raw_6lags.npz"

    Επιστρέφει / Returns:
        tuple: (smoothing_type, n_lags)

    Παραδείγματα / Examples:
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
    Εκπαιδεύει μοντέλα γραμμικής παλινδρόμησης σε όλα τα διαθέσιμα σύνολα χαρακτηριστικών.
    Trains linear regression models on all available feature sets.

    Παράμετροι / Args:
        features_dir (str): Κατάλογος που περιέχει αρχεία χαρακτηριστικών
                             Directory containing feature files

    Επιστρέφει / Returns:
        pd.DataFrame: Αποτελέσματα για όλες τις ρυθμίσεις / Results for all configurations

    Στρατηγική / Strategy:
        - Δοκιμή όλων των συνδυασμών προεπεξεργασίας και παραθύρων υστέρησης
          Test all combinations of preprocessing and lag windows
        - Παρακολούθηση απόδοσης σε εκπαίδευση και επικύρωση
          Track performance on both training and validation sets
        - Identify overfitting: large gap between train and validation error
        - Select best model based on validation performance (not training)
    """
    # Find all feature files
    feature_files = glob.glob(os.path.join(features_dir, "features_*.npz"))

    if len(feature_files) == 0:
        raise FileNotFoundError(f"No feature files found in {features_dir}/")

    print(
        f"Βρέθηκαν / Found {len(feature_files)} ρυθμίσεις χαρακτηριστικών / feature configurations"
    )
    print()

    results = []

    for features_path in sorted(feature_files):
        # Extract configuration
        smoothing_type, n_lags = extract_config_from_filename(features_path)

        if smoothing_type is None:
            continue

        print(
            f"Εκπαίδευση / Training: εξομάλυνση / smoothing={smoothing_type}, n_lags={n_lags}"
        )

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
            f"  Εκπαίδευση / Training   - RMSE: ${train_metrics['RMSE']:.2f}, MAE: ${train_metrics['MAE']:.2f}, R²: {train_metrics['R2']:.4f}"
        )
        print(
            f"  Επικύρωση / Validation - RMSE: ${val_metrics['RMSE']:.2f}, MAE: ${val_metrics['MAE']:.2f}, R²: {val_metrics['R2']:.4f}"
        )
        print()

    return pd.DataFrame(results)


def analyze_best_configuration(results_df):
    """
    Εντοπίζει και αναλύει την καλύτερα ρύθμιση.
    Identifies and analyzes the best performing configuration.

    Παράμετροι / Args:
        results_df (pd.DataFrame): Αποτελέσματα από όλες τις ρυθμίσεις / Results from all configurations

    Επιστρέφει / Returns:
        dict: Πληροφορίες καλύτερης ρύθμισης / Best configuration information

    Κριτήρια Επιλογής / Selection Criteria:
        - Κύριο / Primary: Χαμηλότερο validation RMSE / Lowest validation RMSE
        - Το validation set αντιπροσωπεύει μελλοντική απόδοση / Validation set represents future performance
        - Οι training μετρικές μπορεί να είναι παραπλανητικά καλές (overfitting)
          Training metrics can be misleadingly good (overfitting)
    """
    # Sort by validation RMSE (lower is better)
    results_sorted = results_df.sort_values("val_rmse")

    best_config = results_sorted.iloc[0]

    print("=" * 80)
    print("ΚΑΛΥΤΕΡΗ ΡΥΘΜΙΣΗ / BEST CONFIGURATION")
    print("=" * 80)
    print(f"Εξομάλυνση / Smoothing: {best_config['smoothing']}")
    print(f"Αριθμός Υστερήσεων / Number of Lags: {best_config['n_lags']}")
    print(f"Σύνολο Χαρακτηριστικών / Total Features: {best_config['n_features']}")
    print(f"Δείγματα Εκπαίδευσης / Training Samples: {best_config['n_train_samples']}")
    print(f"Δείγματα Επικύρωσης / Validation Samples: {best_config['n_val_samples']}")
    print()
    print("Μετρικές Εκπαίδευσης / Training Metrics:")
    print(f"  RMSE: ${best_config['train_rmse']:.2f}")
    print(f"  MAE:  ${best_config['train_mae']:.2f}")
    print(f"  R²:   {best_config['train_r2']:.4f}")
    print()
    print("Μετρικές Επικύρωσης / Validation Metrics:")
    print(f"  RMSE: ${best_config['val_rmse']:.2f}")
    print(f"  MAE:  ${best_config['val_mae']:.2f}")
    print(f"  R²:   {best_config['val_r2']:.4f}")
    print()

    # Analyze overfitting
    rmse_gap = best_config["val_rmse"] - best_config["train_rmse"]
    mae_gap = best_config["val_mae"] - best_config["train_mae"]
    r2_gap = best_config["train_r2"] - best_config["val_r2"]

    print("Ανάλυση Overfitting / Overfitting Analysis:")
    print(f"  Διαφορά RMSE (Val - Train) / RMSE Gap (Val - Train): ${rmse_gap:.2f}")
    print(f"  Διαφορά MAE (Val - Train) / MAE Gap (Val - Train):  ${mae_gap:.2f}")
    print(f"  Διαφορά R² (Train - Val) / R² Gap (Train - Val):   {r2_gap:.4f}")

    if rmse_gap > 50:
        print(
            "  ⚠ Προειδοποίηση / Warning: Σημαντικό overfitting (μεγάλη διαφορά RMSE) / Significant overfitting detected (large RMSE gap)"
        )
    elif rmse_gap > 20:
        print(
            "  ⚠ Μέτριο overfitting (μέτρια διαφορά RMSE) / Moderate overfitting (moderate RMSE gap)"
        )
    else:
        print(
            "  ✓ Καλή γενίκευση (μικρή διαφορά RMSE) / Good generalization (small RMSE gap)"
        )

    return best_config


def display_model_parameters(model, feature_names):
    """
    Εμφανίζει τις μαθημένες παραμέτρους του μοντέλου (συντελεστές).
    Displays the learned model parameters (coefficients).

    Παράμετροι / Args:
        model: Εκπαιδευμένο LinearRegression μοντέλο / Trained LinearRegression model
        feature_names: Ονόματα χαρακτηριστικών / Names of features

    Εξίσωση Μοντέλου / Model Equation:
        τιμή / price = τομή / intercept + Σ(συντελεστής_i / coefficient_i × χαρακτηριστικό_i / feature_i)

    Ερμηνεία (για κλιμακωμένα χαρακτηριστικά) / Interpretation (for scaled features):
        - Θετικός συντελεστής / Positive coefficient: αύξηση χαρακτηριστικού → αύξηση τιμής / feature increase → price increase
        - Αρνητικός συντελεστής / Negative coefficient: αύξηση χαρακτηριστικού → μείωση τιμής / feature increase → price decrease
        - Μεγαλύτερη απόλυτη τιμή / Larger absolute value: ισχυρότερη επιρροή / stronger influence
        - Τα χαρακτηριστικά είναι κλιμακωμένα, άρα οι συντελεστές είναι άμεσα συγκρίσιμοι
          Features are scaled, so coefficients are directly comparable
    """
    print("=" * 80)
    print("ΠΑΡΑΜΕΤΡΟΙ ΜΟΝΤΕΛΟΥ (ΓΡΑΜΜΙΚΗ ΠΑΛΙΝΔΡΟΜΗΣΗ)")
    print("MODEL PARAMETERS (LINEAR REGRESSION)")
    print("=" * 80)
    print()
    print(f"Τομή / Intercept (β₀): ${model.intercept_:.2f}")
    print()
    print("Συντελεστές Χαρακτηριστικών / Feature Coefficients:")
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
    print("Ερμηνεία / Interpretation:")
    print("  - Οι συντελεστές δείχνουν την αλλαγή στην προβλεπόμενη τιμή (σε $)")
    print("    Coefficients show the change in predicted price (in $)")
    print("    για αλλαγή 1 τυπικής απόκλισης στο χαρακτηριστικό")
    print("    for a 1 standard deviation change in the feature")
    print(
        "  - Θετικό / Positive: αύξηση χαρακτηριστικού → αύξηση τιμής / feature increase → price increase"
    )
    print(
        "  - Αρνητικό / Negative: αύξηση χαρακτηριστικού → μείωση τιμής / feature increase → price decrease"
    )
    print(
        "  - Μεγαλύτερη απόλυτη τιμή / Larger absolute value → ισχυρότερη επιρροή / stronger influence"
    )
    print()

    # Highlight most important features
    print("Τοπ 5 Πιο Επιδραστικά Χαρακτηριστικά / Top 5 Most Influential Features:")
    for rank, (_, row) in enumerate(coef_df.head(5).iterrows(), 1):
        direction = (
            "αυξάνει / increases" if row["coefficient"] > 0 else "μειώνει / decreases"
        )
        print(
            f"  {rank}. {row['feature']}: {direction} τιμή / price by ${abs(row['coefficient']):.2f} ανά std dev / per std dev"
        )


def create_performance_comparison_plots(results_df, output_dir="results"):
    """
    Δημιουργεί ολοκληρωμένες οπτικοποιήσεις που συγκρίνουν όλες τις ρυθμίσεις.
    Creates comprehensive visualizations comparing all configurations.

    Παράμετροι / Args:
        results_df (pd.DataFrame): Αποτελέσματα από όλες τις ρυθμίσεις / Results from all configurations
        output_dir (str): Κατάλογος αποθήκευσης γραφημάτων / Directory to save plots
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
    Δημιουργεί γράφημα διασποράς πραγματικών vs προβλεπόμενων τιμών για το καλύτερο μοντέλο.
    Creates scatter plot of actual vs predicted prices for best model.

    Παράμετροι / Args:
        best_config: Καλύτερη ρύθμιση από αποτελέσματα / Best configuration from results
        output_dir (str): Κατάλογος αποθήκευσης γραφήματος / Directory to save plot
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
    Αποθηκεύει λεπτομερή πίνακα αποτελεσμάτων σε CSV.
    Saves detailed results table to CSV.

    Παράμετροι / Args:
        results_df (pd.DataFrame): Αποτελέσματα από όλες τις ρυθμίσεις / Results from all configurations
        output_dir (str): Κατάλογος αποθήκευσης CSV / Directory to save CSV
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
    print(f"✓ Αποθηκεύτηκε πίνακας αποτελεσμάτων / Saved results table: {csv_path}")

    return csv_path


def save_best_model(best_config, output_dir="models"):
    """
    Αποθηκεύει το καλύτερο μοντέλο για μελλοντική χρήση.
    Saves the best model for later use.

    Παράμετροι / Args:
        best_config: Καλύτερη ρύθμιση dictionary / Best configuration dictionary
        output_dir (str): Κατάλογος αποθήκευσης μοντέλου / Directory to save model
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

    print(f"✓ Αποθηκεύτηκε το καλύτερο μοντέλο / Saved best model: {model_path}")

    return model_path


def main():
    """
    Κύρια συνάρτηση εκτέλεσης για ανάλυση baseline γραμμικής παλινδρόμησης.
    Main execution function for baseline linear regression analysis.

    Ροή Εργασιών / Workflow:
        1. Εκπαίδευση μοντέλων σε όλες τις ρυθμίσεις / Train models on all configurations
        2. Σύγκριση μετρικών απόδοσης / Compare performance metrics
        3. Εντοπισμός καλύτερης ρύθμισης / Identify best configuration
        4. Εμφάνιση παραμέτρων μοντέλου / Display model parameters
        5. Δημιουργία οπτικοποιήσεων / Create visualizations
        6. Αποθήκευση αποτελεσμάτων και καλύτερου μοντέλου / Save results and best model
    """
    print("=" * 80)
    print("ΠΡΟΒΛΕΨΗ ΤΙΜΗΣ ΜΕΤΟΧΗΣ NFLX - BASELINE ΓΡΑΜΜΙΚΗ ΠΑΛΙΝΔΡΟΜΗΣΗ (ΕΡΓΑΣΙΑ Α)")
    print("NFLX STOCK PRICE PREDICTION - BASELINE LINEAR REGRESSION (TASK A)")
    print("=" * 80)
    print()

    # Train all configurations
    print("Εκπαίδευση μοντέλων γραμμικής παλινδρόμησης σε όλες τις ρυθμίσεις...")
    print("Training linear regression models on all configurations...")
    print()
    results_df = train_all_configurations()

    print("=" * 80)
    print("Η ΕΚΠΑΙΔΕΥΣΗ ΟΛΟΚΛΗΡΩΘΗΚΕ / TRAINING COMPLETED")
    print("=" * 80)
    print(
        f"Σύνολο ρυθμίσεων που δοκιμάστηκαν / Total configurations tested: {len(results_df)}"
    )
    print()

    # Analyze best configuration
    best_config = analyze_best_configuration(results_df)
    print()

    # Display model parameters
    display_model_parameters(best_config["model"], best_config["feature_names"])
    print()

    # Create visualizations
    print("Δημιουργία οπτικοποιήσεων... / Creating visualizations...")
    create_performance_comparison_plots(results_df)
    create_actual_vs_predicted_plot(best_config)
    print()

    # Save results
    print("Αποθήκευση αποτελεσμάτων... / Saving results...")
    save_results_table(results_df)
    save_best_model(best_config)
    print()

    print("=" * 80)
    print("✓ Η ΑΝΑΛΥΣΗ BASELINE ΓΡΑΜΜΙΚΗΣ ΠΑΛΙΝΔΡΟΜΗΣΗΣ ΟΛΟΚΛΗΡΩΘΗΚΕ")
    print("✓ BASELINE LINEAR REGRESSION ANALYSIS COMPLETED")
    print("=" * 80)
    print()
    print("Κύρια Ευρήματα (Key Findings):")
    print(f"  Καλύτερη Εξομάλυνση (Best Smoothing): {best_config['smoothing']}")
    print(
        f"  Καλύτερο Παράθυρο Υστέρησης (Best Lag Window): {best_config['n_lags']} μήνες (months)"
    )
    print(f"  Validation RMSE: ${best_config['val_rmse']:.2f}")
    print(f"  Validation R²: {best_config['val_r2']:.4f}")
    print()
    print("Επόμενο Βήμα (Next Step):")
    print("  Εκτέλεση step4_polynomial_regression_regularization.py για Εργασία Β")
    print("  Run step4_polynomial_regression_regularization.py for Task B")


if __name__ == "__main__":
    main()
