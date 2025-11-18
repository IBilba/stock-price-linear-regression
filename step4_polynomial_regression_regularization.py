"""
ΒΗΜΑ 4: ΠΟΛΥΩΝΥΜΙΚΗ ΠΑΛΙΝΔΡΟΜΗΣΗ ΜΕ L1/L2 ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ (ΕΡΓΑΣΙΑ Β)
STEP 4: POLYNOMIAL REGRESSION WITH L1/L2 REGULARIZATION (TASK B)
================================================================

Αυτό το script υλοποιεί πολυωνυμική παλινδρόμηση με κανονικοποίηση:
This script implements polynomial regression with regularization:

1. Χρησιμοποιεί ΌΛΕΣ τις 16 ρυθμίσεις baseline
   Uses ALL 16 baseline configurations

2. Δημιουργεί πολυωνυμικά χαρακτηριστικά βαθμού 2
   Creates polynomial features (degree 2)

3. Εκπαιδεύει Ridge (L2) και Lasso (L1) μοντέλα
   Trains Ridge (L2) and Lasso (L1) regression models

4. Εκτελεί grid search για βέλτιστο alpha (ισχύς κανονικοποίησης)
   Performs grid search for optimal alpha (regularization strength)

Αυτό απαντά στην ΕΡΓΑΣΙΑ Β (This addresses TASK B):
"Χρησιμοποιήστε πολυωνυμική παλινδρόμηση με L1, L2 κανονικοποίηση."
"Use a polynomial regression model with L1, L2 normalization norms."

Βασικές Έννοιες (Key Concepts):
- Πολυωνυμικά χαρακτηριστικά συλλαμβάνουν μη-γραμμικές σχέσεις
  (Polynomial features capture non-linear relationships)
- L1 (Lasso): Αραιά μοντέλα, επιλογή χαρακτηριστικών (Sparse models, feature selection)
- L2 (Ridge): Διαχειρίζεται multicollinearity, αποτρέπει overfitting
  (Handles multicollinearity, prevents overfitting)

Συγγραφέας (Author): Statistical Methods of Machine Learning - Task 1
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
    Φορτώνει ένα αποθηκευμένο σύνολο χαρακτηριστικών.
    Loads a saved feature set.

    Παράμετροι (Args):
        features_path (str): Διαδρομή προς αρχείο .npz (Path to .npz file)

    Επιστρέφει (Returns):
        dict: Δεδομένα χαρακτηριστικών (Feature data)
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
    Υπολογίζει μετρικές RMSE, MAE, R².
    Computes RMSE, MAE, R² metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def create_polynomial_features(X_train, X_val, degree):
    """
    Δημιουργεί πολυωνυμικά χαρακτηριστικά από χαρακτηριστικά εισόδου.
    Creates polynomial features from input features.

    Παράμετροι (Args):
        X_train: Χαρακτηριστικά εκπαίδευσης (Training features)
        X_val: Χαρακτηριστικά επικύρωσης (Validation features)
        degree (int): Βαθμός πολυωνύμου (Polynomial degree)

    Επιστρέφει (Returns):
        tuple: (X_train_poly, X_val_poly, poly_transformer, new_feature_names)

    Μετασχηματισμός Πολυωνυμικών Χαρακτηριστικών (Polynomial Feature Transformation):

        Βαθμός 1 (Degree 1): Αρχικά χαρακτηριστικά (Original features) [x₁, x₂, x₃, ...]

        Βαθμός 2 (Degree 2): [1, x₁, x₂, x₃, x₁², x₁x₂, x₁x₃, x₂², x₂x₃, x₃², ...]
                  - Προσθέτει τετραγωνικούς όρους (Adds squared terms) (x²)
                  - Προσθέτει όρους αλληλεπίδρασης (Adds interaction terms) (x₁x₂)

        Βαθμός 3 (Degree 3): [1, x₁, x₂, ..., x₁², x₁x₂, ..., x₁³, x₁²x₂, ...]
                  - Προσθέτει κυβικούς όρους (Adds cubic terms) (x³)
                  - Προσθέτει αλληλεπιδράσεις υψηλότερης τάξης (Adds higher-order interactions)

    Πλεονεκτήματα (Benefits):
        - Συλλαμβάνει μη-γραμμικές σχέσεις (Captures non-linear relationships)
        - Οι τιμές μετοχών μπορεί να έχουν τετραγωνικές τάσεις ή αλληλεπιδράσεις
          (Stock prices may have quadratic trends or interactions)
        - Παράδειγμα (Example): close_t-1 * volume_t-1 (αλληλεπίδραση τιμής-όγκου (price-volume interaction))

    Κίνδυνοι (Risks):
        - Εκθετική αύξηση χαρακτηριστικών (κατάρα της διαστατικότητας)
          (Exponential growth in features (curse of dimensionality))
        - 24 χαρακτηριστικά (features) → βαθμός (degree) 2 → 300 χαρακτηριστικά (features)
        - 24 χαρακτηριστικά (features) → βαθμός (degree) 3 → 2,600 χαρακτηριστικά (features)
        - Υψηλός κίνδυνος overfitting χωρίς κανονικοποίηση
          (High risk of overfitting without regularization)

    Γιατί η Κανονικοποίηση είναι Κρίσιμη (Why Regularization is Critical):
        Με πολυωνυμικά χαρακτηριστικά, έχουμε πολύ περισσότερες παραμέτρους από δείγματα
        (With polynomial features, we have far more parameters than samples)
        Αυτό οδηγεί σε σοβαρό overfitting χωρίς L1/L2 ποινές
        (This leads to severe overfitting without L1/L2 penalties)
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True)

    # Fit on training data only
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Get feature names
    feature_names = poly.get_feature_names_out()

    print(f"  Πολυωνυμικός βαθμός (Polynomial degree) {degree}:")
    print(f"    Αρχικά χαρακτηριστικά (Original features): {X_train.shape[1]}")
    print(
        f"    Πολυωνυμικά χαρακτηριστικά (Polynomial features): {X_train_poly.shape[1]}"
    )
    expansion_ratio = X_train_poly.shape[1] / X_train.shape[1]
    print(f"    Αναλογία επέκτασης (Expansion ratio): {expansion_ratio:.1f}x")

    return X_train_poly, X_val_poly, poly, feature_names


def train_ridge_regression(X_train, y_train, X_val, y_val, alpha_values):
    """
    Εκπαιδεύει Ridge παλινδρόμηση με L2 κανονικοποίηση.
    Trains Ridge regression with L2 regularization.

    Παράμετροι (Args):
        X_train, y_train: Δεδομένα εκπαίδευσης (Training data)
        X_val, y_val: Δεδομένα επικύρωσης (Validation data)
        alpha_values: Λίστα ισχύος κανονικοποίησης προς δοκιμή (List of regularization strengths to test)

    Επιστρέφει (Returns):
        dict: Καλύτερο μοντέλο και αποτελέσματα (Best model and results)

    Ridge Παλινδρόμηση (L2 Κανονικοποίηση) (Ridge Regression (L2 Regularization)):

        Αντικείμενο (Objective): Minimize Σ(y - ŷ)² + α·Σ(β²)
                   ├──────┬──────┘   └───┬────┘
                   Συνάρτηση σφάλματος (Loss function)      L2 ποινή (penalty)

        Όπου (Where):
        - α (alpha): Ισχύς κανονικοποίησης (Regularization strength)
        - β: Συντελεστές μοντέλου (Model coefficients)
        - Σ(β²): Άθροισμα τετραγωνικών συντελεστών (Sum of squared coefficients)

    Πώς λειτουργεί το L2 (How L2 Works):
        - Ποινή σε μεγάλους συντελεστές (Penalizes large coefficients)
        - Ενθαρρύνει μικρούς αλλά μη μηδενικούς συντελεστές (Encourages small but non-zero coefficients)
        - Όλα τα χαρακτηριστικά παραμένουν στο μοντέλο (όχι αραίωση)
          All features remain in model (no sparsity)
        - Βοηθά με multicollinearity (συσχετισμένα χαρακτηριστικά)
          Helps with multicollinearity (correlated features)
        - Συρρικνώνει συντελεστές προς το μηδέν αλλά ποτέ ακριβώς στο μηδέν
          Shrinks coefficients toward zero but never exactly zero

    Παράμετρος Alpha:
        - α = 0: Χωρίς κανονικοποίηση (στανταρδ γραμμική παλινδρόμηση) (No regularization (standard linear regression))
        - Μικρό α (0.001, 0.01): Ελαφριά κανονικοποίηση (Small α (0.001, 0.01): Light regularization)
        - Μέτριο α (0.1, 1.0): Μέτρια κανονικοποίηση (Medium α (0.1, 1.0): Moderate regularization)
        - Μεγάλο α (10, 100): Ισχυρή κανονικοποίηση (συντελεστές → 0) (Large α (10, 100): Heavy regularization (coefficients → 0))

    Πλεονεκτήματα (Benefits):
        - Αποτρέπει το overfitting με πολλά χαρακτηριστικά (Prevents overfitting with many features)
        - Σταθερό όταν τα χαρακτηριστικά είναι συσχετισμένα (Stable when features are correlated)
        - Πάντα έχει μοναδική λύση (Always has a unique solution)
        - Υπολογιστικά αποδοτικό (Computationally efficient)

    Περίπτωση Χρήσης (Use Case):
        - Όταν όλα τα χαρακτηριστικά μπορεί να είναι σχετικά (When all features might be relevant)
        - Όταν τα χαρακτηριστικά είναι πολύ συσχετισμένα (π.χ. τιμές με υστέρηση)
          (When features are highly correlated (e.g., lagged prices))
        - Όταν θέλετε σταθερές εκτιμήσεις συντελεστών (When you want stable coefficient estimates)
    """
    best_alpha = None
    best_val_rmse = float("inf")
    best_model = None
    results = []

    print("\n  Δοκιμή Ridge (L2) παλινδρόμησης (Testing Ridge (L2) regression):")
    print(f"    Τιμές Alpha (Alpha values): {alpha_values}")

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

    print(
        f"    Καλύτερο alpha (Best alpha): {best_alpha} (Επικύρωση RMSE (Validation RMSE): ${best_val_rmse:.2f})"
    )

    return {
        "best_model": best_model,
        "best_alpha": best_alpha,
        "all_results": results,
        "model_type": "Ridge",
    }


def train_lasso_regression(X_train, y_train, X_val, y_val, alpha_values):
    """
    Εκπαιδεύει Lasso παλινδρόμηση με L1 κανονικοποίηση.
    Trains Lasso regression with L1 regularization.

    Παράμετροι (Args):
        X_train, y_train: Δεδομένα εκπαίδευσης (Training data)
        X_val, y_val: Δεδομένα επικύρωσης (Validation data)
        alpha_values: Λίστα ισχύος κανονικοποίησης προς δοκιμή (List of regularization strengths to test)

    Επιστρέφει (Returns):
        dict: Καλύτερο μοντέλο και αποτελέσματα (Best model and results)

    Lasso Παλινδρόμηση (L1 Κανονικοποίηση) (Lasso Regression (L1 Regularization)):

        Αντικείμενο (Objective): Minimize Σ(y - ŷ)² + α·Σ|β|
                   ├──────┬──────┘   └──┬──┘
                   Συνάρτηση σφάλματος (Loss function)    L1 ποινή (penalty)

        Όπου (Where):
        - α (alpha): Ισχύς κανονικοποίησης (Regularization strength)
        - β: Συντελεστές μοντέλου (Model coefficients)
        - Σ|β|: Άθροισμα απόλυτων τιμών συντελεστών (Sum of absolute values of coefficients)

    Πώς λειτουργεί το L1 (How L1 Works):
        - Ποινή στην απόλυτη τιμή των συντελεστών (Penalizes absolute value of coefficients)
        - Εξαναγκάζει μερικούς συντελεστές να γίνουν ΑΚΡΙΒΩΣ μηδέν (Forces some coefficients to EXACTLY zero)
        - Εκτελεί αυτόματη επιλογή χαρακτηριστικών (Performs automatic feature selection)
        - Δημιουργεί αραιά μοντέλα (λίγοι μη μηδενικοί συντελεστές)
          (Creates sparse models (few non-zero coefficients))
        - Λιγότερο σταθερό από το L2 με συσχετισμένα χαρακτηριστικά (Less stable than L2 with correlated features)

    Παράμετρος Alpha:
        - α = 0: Χωρίς κανονικοποίηση (μπορεί να μη συγκλίνει) (No regularization (may not converge))
        - Μικρό α (0.001, 0.01): Ελαφριά επιλογή (Small α (0.001, 0.01): Light selection)
        - Μέτριο α (0.1, 1.0): Μέτρια επιλογή (Medium α (0.1, 1.0): Moderate selection)
        - Μεγάλο α (10, 100): Ισχυρή επιλογή (οι περισσότεροι συντελεστές → 0)
          (Large α (10, 100): Heavy selection (most coefficients → 0))

    Πλεονεκτήματα (Benefits):
        - Αυτόματη επιλογή χαρακτηριστικών (Automatic feature selection)
        - Ερμηνεύσιμα μοντέλα (μόνο σημαντικά χαρακτηριστικά μένουν)
          (Interpretable models (only important features remain))
        - Διαχειρίζεται δεδομένα υψηλής διάστασης (Handles high-dimensional data)
        - Καλό όταν πολλά χαρακτηριστικά είναι ασύμφορα (Good when many features are irrelevant)

    Μειονεκτήματα (Drawbacks):
        - Μπορεί να επιλέξει τυχαία ένα από συσχετισμένα χαρακτηριστικά
          (Can arbitrarily select one among correlated features)
        - Λιγότερο σταθερό από το Ridge (Less stable than Ridge)
        - Μπορεί να απαιτεί περισσότερες επαναλήψεις για να συγκλίνει (May require more iterations to converge)

    Περίπτωση Χρήσης (Use Case):
        - Όταν υποπτεύεστε ότι πολλά χαρακτηριστικά είναι ασύμφορα (When you suspect many features are irrelevant)
        - Όταν θέλετε ένα ερμηνεύσιμο μοντέλο (When you want an interpretable model)
        - Όταν χρειάζεστε επιλογή χαρακτηριστικών (When you need feature selection)
        - Με πολυωνυμικά χαρακτηριστικά (επιλέγει σημαντικές αλληλεπιδράσεις)
          (With polynomial features (selects important interactions))

    Σύγκριση με Ridge (Comparison with Ridge):
        - Lasso: Αραιό, επιλογή χαρακτηριστικών, λιγότερο σταθερό (Sparse, feature selection, less stable)
        - Ridge: Πυκνό, όχι επιλογή, πιο σταθερό (Dense, no selection, more stable)
        - Lasso καλύτερο για επιλογή χαρακτηριστικών (Lasso better for feature selection)
        - Ridge καλύτερο όταν όλα τα χαρακτηριστικά μετράνε (Ridge better when all features matter)
    """
    best_alpha = None
    best_val_rmse = float("inf")
    best_model = None
    results = []

    print("\n  Δοκιμή Lasso (L1) παλινδρόμησης (Testing Lasso (L1) regression):")
    print(f"    Τιμές Alpha (Alpha values): {alpha_values}")

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

    print(
        f"    Καλύτερο alpha (Best alpha): {best_alpha} (Επικύρωση RMSE (Validation RMSE): ${best_val_rmse:.2f})"
    )
    if best_model is not None:
        print(
            f"    Μη μηδενικοί συντελεστές (Non-zero coefficients): {np.sum(np.abs(best_model.coef_) > 1e-6)} out of {len(best_model.coef_)}"
        )

    return {
        "best_model": best_model,
        "best_alpha": best_alpha,
        "all_results": results,
        "model_type": "Lasso",
    }


def compare_models(baseline_results, poly_results, degree):
    """
    Δημιουργεί πίνακα σύγκρισης όλων των μοντέλων.
    Creates comparison table of all models.

    Παράμετροι (Args):
        baseline_results: Αποτελέσματα baseline γραμμικής παλινδρόμησης (Baseline linear regression results)
        poly_results: Λίστα αποτελεσμάτων πολυωνυμικής παλινδρόμησης (List of polynomial regression results)
        degree: Βαθμός πολυωνύμου που χρησιμοποιήθηκε (Polynomial degree used)

    Επιστρέφει (Returns):
        pd.DataFrame: Πίνακας σύγκρισης (Comparison table)
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
    Εμφανίζει παραμέτρους και ανάλυση του καλύτερου πολυωνυμικού μοντέλου.
    Displays parameters and analysis of best polynomial model.

    Παράμετροι (Args):
        result: Λεξικό αποτελεσμάτων μοντέλου (Model results dictionary)
        feature_names: Ονόματα πολυωνυμικών χαρακτηριστικών (Names of polynomial features)
    """
    model = result["best_model"]
    model_type = result["model_type"]
    alpha = result["best_alpha"]

    print(f"\n{'='*80}")
    print(
        f"ΚΑΛΥΤΕΡΕΣ ΠΑΡΑΜΕΤΡΟΙ {model_type.upper()} ΜΟΝΤΕΛΟΥ (BEST {model_type.upper()} MODEL PARAMETERS)"
    )
    print(f"{'='*80}")
    print(
        f"Κανονικοποίηση (Regularization): {model_type} ({'L2' if model_type == 'Ridge' else 'L1'})"
    )
    print(f"Alpha (λ): {alpha}")
    print(f"Τομή (Intercept): ${model.intercept_:.2f}")
    print(f"Συνολικά Χαρακτηριστικά (Total Features): {len(model.coef_)}")

    # Coefficient analysis
    nonzero_mask = np.abs(model.coef_) > 1e-6
    n_nonzero = np.sum(nonzero_mask)

    print(
        f"Μη μηδενικοί Συντελεστές (Non-zero Coefficients): {n_nonzero} ({100*n_nonzero/len(model.coef_):.1f}%)"
    )

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

    print(f"\nΤοπ 15 Πιο Σημαντικά Χαρακτηριστικά (Top 15 Most Important Features):")
    print("-" * 80)
    print(f"{'Χαρακτηριστικό (Feature)':<40} {'Συντελεστής (Coefficient)':>20}")
    print("-" * 80)

    for i, (idx, row) in enumerate(coef_df.head(15).iterrows(), 1):
        feat_str = str(row["feature"])
        if len(feat_str) > 38:
            feat_str = feat_str[:35] + "..."
        print(f"{feat_str:<40} {row['coefficient']:>20.4f}")

    print("-" * 80)

    # Feature type analysis for polynomial features
    if result["model_type"] == "Lasso":
        print(
            f"\nΑνάλυση Επιλογής Χαρακτηριστικών (Lasso) (Feature Selection Analysis (Lasso)):"
        )
        print(
            f"  Επιλεγμένα χαρακτηριστικά (Selected features): {n_nonzero} από (out of) {len(model.coef_)}"
        )
        print(f"  Αραίωση (Sparsity): {100*(1 - n_nonzero/len(model.coef_)):.1f}%")
        print(
            f"  → Το μοντέλο χρησιμοποιεί μόνο (Model uses only) {n_nonzero} χαρακτηριστικά για πρόβλεψη (features for prediction)"
        )


def create_regularization_path_plot(poly_results, output_dir="results"):
    """
    Οπτικοποιεί πώς η απόδοση αλλάζει με την ισχύ κανονικοποίησης.
    Visualizes how performance changes with regularization strength.

    Παράμετροι (Args):
        poly_results: Λίστα αποτελεσμάτων πολυωνυμικής παλινδρόμησης (List of polynomial regression results)
        output_dir: Κατάλογος εξόδου (Output directory)

    Εμφανίζει (Shows):
        - Επικύρωση RMSE vs alpha για Ridge και Lasso (Validation RMSE vs alpha for Ridge and Lasso)
        - Βοηθά να κατανοήσουμε την επίδραση της κανονικοποίησης (Helps understand regularization impact)
        - Προσδιορίζει βέλτιστες τιμές alpha (Identifies optimal alpha values)
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
            alphas,
            train_rmse,
            "o-",
            label="Training RMSE",
            linewidth=2,
            markersize=8,
        )
        ax.semilogx(
            alphas,
            val_rmse,
            "s-",
            label="Validation RMSE",
            linewidth=2,
            markersize=8,
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
            f"{model_type} Regularization Path",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "polynomial_regularization_paths.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(
        f"✓ Αποθηκεύτηκε γράφημα διαδρομής κανονικοποίησης (Saved regularization path plot): {plot_path}"
    )
    plt.close()


def save_results(comparison_df, poly_results, output_dir="results"):
    """
    Αποθηκεύει αποτελέσματα πολυωνυμικής παλινδρόμησης.
    Saves polynomial regression results.

    Παράμετροι (Args):
        comparison_df: DataFrame σύγκρισης (Comparison DataFrame)
        poly_results: Αποτελέσματα πολυωνυμικού μοντέλου (Polynomial model results)
        output_dir: Κατάλογος εξόδου (Output directory)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save comparison table
    csv_path = os.path.join(output_dir, "polynomial_regression_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"✓ Αποθηκεύτηκε πίνακας σύγκρισης (Saved comparison table): {csv_path}")

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
        print(
            f"✓ Αποθηκεύτηκε {model_type} μοντέλο (Saved {model_type} model): {model_path}"
        )


def main():
    """
    Κύρια συνάρτηση εκτέλεσης για πολυωνυμική παλινδρόμηση με κανονικοποίηση.
    Main execution function for polynomial regression with regularization.

    Στρατηγική (Strategy):
        1. Δοκιμή ΌΛΩΝ των 16 baseline ρυθμίσεων (όχι μόνο την καλύτερη)
           (Test ALL 16 baseline configurations (not just best))
        2. Εφαρμόζει πολυωνυμικά χαρακτηριστικά βαθμού 2 σε κάθε ρύθμιση
           (Apply polynomial degree 2 features to each)
        3. Grid search Ridge και Lasso για κάθε ρύθμιση
           (Grid search Ridge and Lasso for each configuration)
        4. Αποθηκεύει αποτελέσματα για όλες τις ρυθμίσεις
           (Save results for all configurations)
        5. Συγκρίνει και προσδιορίζει το καλύτερο συνολικά
           (Compare and identify best overall)
    """
    print("=" * 80)
    print("ΠΡΟΒΛΕΨΗ ΤΙΜΗΣ ΜΕΤΟΧΗΣ NFLX - ΠΟΛΥΩΝΥΜΙΚΗ ΠΑΛΙΝΔΡΟΜΗΣΗ (ΕΡΓΑΣΙΑ Β)")
    print("(NFLX STOCK PRICE PREDICTION - POLYNOMIAL REGRESSION (TASK B))")
    print("=" * 80)
    print()

    # Configuration
    alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    polynomial_degree = 2  # Degree 2 to avoid feature explosion

    # Find all feature configurations
    features_dir = "features"
    feature_files = [
        f
        for f in os.listdir(features_dir)
        if f.startswith("features_") and f.endswith(".npz")
    ]

    print(
        f"Βρέθηκαν (Found) {len(feature_files)} ρυθμίσεις χαρακτηριστικών (feature configurations)"
    )
    print()

    # Results storage
    all_configs_results = []
    all_poly_models = {}

    # Process each configuration
    for feature_file in feature_files:
        # Extract config name (e.g., "sigma3_12lags")
        config_name = feature_file.replace("features_", "").replace(".npz", "")
        parts = config_name.split("_")
        smoothing = parts[0]
        n_lags = int(parts[1].replace("lags", ""))

        print("=" * 80)
        print(f"Επεξεργασία (Processing): {smoothing}, {n_lags} υστερήσεις (lags)")
        print("=" * 80)

        # Load feature set
        features_path = os.path.join(features_dir, feature_file)
        data = load_feature_set(features_path)
        print(f"✓ Φορτώθηκε (Loaded): {features_path}")
        print(f"  Δείγματα εκπαίδευσης (Training samples): {len(data['y_train'])}")
        print(f"  Δείγματα επικύρωσης (Validation samples): {len(data['y_val'])}")
        print(
            f"  Αρχικά χαρακτηριστικά (Original features): {data['X_train'].shape[1]}"
        )

        # Create polynomial features
        print(
            f"  Δημιουργία πολυωνυμικών χαρακτηριστικών (Creating polynomial features) (βαθμός (degree)={polynomial_degree})..."
        )
        X_train_poly, X_val_poly, poly_transformer, poly_feature_names = (
            create_polynomial_features(
                data["X_train"], data["X_val"], polynomial_degree
            )
        )

        # Train Ridge regression
        print(
            "  Εκπαίδευση Ridge παλινδρόμησης (L2) (Training Ridge regression (L2))..."
        )
        ridge_results = train_ridge_regression(
            X_train_poly, data["y_train"], X_val_poly, data["y_val"], alpha_values
        )

        # Train Lasso regression
        print(
            "  Εκπαίδευση Lasso παλινδρόμησης (L1) (Training Lasso regression (L1))..."
        )
        lasso_results = train_lasso_regression(
            X_train_poly, data["y_train"], X_val_poly, data["y_val"], alpha_values
        )

        # Get best from Ridge and Lasso
        ridge_best = [
            r
            for r in ridge_results["all_results"]
            if r["alpha"] == ridge_results["best_alpha"]
        ][0]
        lasso_best = [
            r
            for r in lasso_results["all_results"]
            if r["alpha"] == lasso_results["best_alpha"]
        ][0]

        # Store results for this configuration
        for model_type, results, best_result in [
            ("Ridge", ridge_results, ridge_best),
            ("Lasso", lasso_results, lasso_best),
        ]:
            all_configs_results.append(
                {
                    "smoothing": smoothing,
                    "n_lags": n_lags,
                    "model_type": model_type,
                    "degree": polynomial_degree,
                    "best_alpha": results["best_alpha"],
                    "n_features": X_train_poly.shape[1],
                    "train_rmse": best_result["train_rmse"],
                    "train_mae": best_result["train_mae"],
                    "train_r2": best_result["train_r2"],
                    "val_rmse": best_result["val_rmse"],
                    "val_mae": best_result["val_mae"],
                    "val_r2": best_result["val_r2"],
                }
            )

        # Store models
        all_poly_models[config_name] = {
            "ridge": ridge_results,
            "lasso": lasso_results,
            "poly_transformer": poly_transformer,
        }

        print(
            f"✓ Η ρύθμιση ολοκληρώθηκε (Configuration complete): {smoothing}, {n_lags} υστερήσεις (lags)"
        )
        print(
            f"  Καλύτερο Ridge alpha (Best Ridge alpha): {ridge_results['best_alpha']} (Επικύρωση RMSE (Val RMSE): ${ridge_best['val_rmse']:.2f})"
        )
        print(
            f"  Καλύτερο Lasso alpha (Best Lasso alpha): {lasso_results['best_alpha']} (Επικύρωση RMSE (Val RMSE): ${lasso_best['val_rmse']:.2f})"
        )
        print()

    # Create results DataFrame
    results_df = pd.DataFrame(all_configs_results)
    results_df = results_df.sort_values("val_rmse")

    # Display summary
    print("=" * 80)
    print("ΣΥΝΟΨΗ ΟΛΩΝ ΤΩΝ ΡΥΘΜΙΣΕΩΝ - ΠΟΛΥΩΝΥΜΙΚΗ ΠΑΛΙΝΔΡΟΜΗΣΗ")
    print("(ALL CONFIGURATIONS SUMMARY - POLYNOMIAL REGRESSION)")
    print("=" * 80)
    print("\nΤοπ 10 Μοντέλα βάσει Επικύρωσης RMSE (Top 10 Models by Validation RMSE):")
    print(results_df.head(10).to_string(index=False))
    print()

    # Save results
    print("=" * 80)
    print("ΑΠΟΘΗΚΕΥΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ (SAVING RESULTS)")
    print("=" * 80)

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save all results table
    csv_path = "results/polynomial_regression_all_models_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Αποθηκεύτηκε πίνακας αποτελεσμάτων (Saved results table): {csv_path}")

    # Save all models
    models_path = "models/all_polynomial_models.pkl"
    with open(models_path, "wb") as f:
        pickle.dump(all_poly_models, f)
    print(f"✓ Αποθηκεύτηκαν όλα τα μοντέλα (Saved all models): {models_path}")

    # Final summary
    best = results_df.iloc[0]
    print()
    print("=" * 80)
    print(
        "✓ Η ΑΝΑΛΥΣΗ ΠΟΛΥΩΝΥΜΙΚΗΣ ΠΑΛΙΝΔΡΟΜΗΣΗΣ ΟΛΟΚΛΗΡΩΘΗΚΕ (POLYNOMIAL REGRESSION ANALYSIS COMPLETED)"
    )
    print("=" * 80)
    print(
        f"\nΔοκιμάστηκαν (Tested) {len(feature_files)} ρυθμίσεις (configurations) × 2 μέθοδοι (methods) = {len(results_df)} μοντέλα (models)"
    )
    print(f"\nΚαλύτερο Συνολικό Μοντέλο (Best Overall Model):")
    print(
        f"  Ρύθμιση (Configuration): {best['smoothing']}, {best['n_lags']} υστερήσεις (lags)"
    )
    print(f"  Τύπος Μοντέλου (Model Type): {best['model_type']}")
    print(f"  Βαθμός (Degree): {best['degree']}")
    print(f"  Alpha: {best['best_alpha']}")
    print(f"  Επικύρωση RMSE (Validation RMSE): ${best['val_rmse']:.2f}")
    print(f"  Επικύρωση R² (Validation R²): {best['val_r2']:.4f}")
    print()
    print(
        "Επόμενο Βήμα (Next Step): Εκτέλεση (Run) step5_dimensionality_reduction.py για την Εργασία Γ (for Task C)"
    )


if __name__ == "__main__":
    main()
