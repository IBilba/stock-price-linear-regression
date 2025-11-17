"""
ΒΗΜΑ 5: ΜΕΙΩΣΗ ΔΙΑΣΤΑΣΕΩΝ ΚΑΙ ΕΠΙΛΟΓΗ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ (ΕΡΓΑΣΙΑ Γ)
STEP 5: DIMENSIONALITY REDUCTION AND FEATURE SELECTION (TASK C)
===============================================================

Αυτό το script υλοποιεί τρεις προσεγγίσεις μείωσης διαστάσεων:
This script implements three dimensionality reduction approaches:

1. PCA (Principal Component Analysis) - Μη εποπτευόμενος μετασχηματισμός
   Unsupervised transformation

2. CFS (Correlation-based Feature Selection) - Μέθοδος φίλτρου
   Filter method

3. Sequential Forward Selection - Wrapper μέθοδος
   Wrapper method

Αυτό απαντά στην ΕΡΓΑΣΙΑ Γ / This addresses TASK C:
"Μειώστε τη διάσταση με PCA, CFA και wrapper μέθοδο. Συγκρίνετε τα αποτελέσματα."
"Reduce the dimension by following PCA, CFA and a wrapper method of
your choice. Compare the results."

Κάθε μέθοδος μειώνει τα χαρακτηριστικά διαφορετικά:
Each method reduces features differently:
- PCA: Δημιουργεί νέα μη-συσχετιζόμενα χαρακτηριστικά (γραμμικοί συνδυασμοί)
  Creates new uncorrelated features (linear combinations)
- CFS: Επιλέγει χαρακτηριστικά με υψηλή συσχέτιση στόχου, χαμηλή δια-συσχέτιση
  Selects features with high target correlation, low inter-correlation
- Wrapper: Χρησιμοποιεί απόδοση μοντέλου για επιλογή χαρακτηριστικών
  Uses model performance to select features

Συγγραφέας / Author: Statistical Methods of Machine Learning - Task 1
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
    """
    Φορτώνει αποθηκευμένο σύνολο χαρακτηριστικών.
    Loads saved feature set.
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
    Υπολογίζει RMSE, MAE, R².
    Computes RMSE, MAE, R².
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def apply_pca(X_train, X_val, variance_threshold=0.95):
    """
    Εφαρμόζει PCA (Ανάλυση Κύριων Συνιστωσών) για μείωση διαστάσεων.
    Applies PCA (Principal Component Analysis) for dimensionality reduction.

    Παράμετροι / Args:
        X_train: Χαρακτηριστικά εκπαίδευσης (ήδη κλιμακωμένα) / Training features (already scaled)
        X_val: Χαρακτηριστικά επικύρωσης (ήδη κλιμακωμένα) / Validation features (already scaled)
        variance_threshold: Ελάχιστη αθροιστική διακύμανση να διατηρηθεί (προεπιλογή: 95%)
                            Minimum cumulative variance to retain (default: 95%)

    Επιστρέφει / Returns:
        tuple: (X_train_pca, X_val_pca, pca_model, n_components)

    Ανάλυση Κύριων Συνιστωσών (PCA) / Principal Component Analysis (PCA):

        Στόχος / Goal: Μετασχηματισμός χαρακτηριστικών σε μη-συσχετιζόμενες κύριες συνιστώσες
                        Transform features into uncorrelated principal components

        Πώς λειτουργεί / How it works:
        1. Βρίσκει κατευθύνσεις μέγιστης διακύμανσης στα δεδομένα
           Finds directions of maximum variance in data
        2. Προβάλλει δεδομένα σε αυτές τις κατευθύνσεις
           Projects data onto these directions
        3. Πρώτη PC συλλαμβάνει περισσότερη διακύμανση, δεύτερη PC δεύτερη περισσότερη, κλπ.
           First PC captures most variance, second PC captures second-most, etc.
        4. Οι PCs είναι ορθογώνιες (μη-συσχετιζόμενες) μεταξύ τους
           PCs are orthogonal (uncorrelated) to each other

        Μαθηματική Βάση / Mathematical Foundation:
        - Ανάλυση ιδιοτιμών πίνακα συνδιακύμανσης / Eigenvalue decomposition of covariance matrix
        - PCs είναι ιδιοδιανύσματα, επεξηγούμενη διακύμανση είναι ιδιοτιμές
          PCs are eigenvectors, variance explained is eigenvalues
        - PC₁ = w₁₁·x₁ + w₁₂·x₂ + ... + w₁ₙ·xₙ (γραμμικός συνδυασμός / linear combination)

        Πλεονεκτήματα / Benefits:
        - Αφαιρεί multicollinearity (καθυστερημένες τιμές είναι πολύ συσχετισμένες)
          Removes multicollinearity (lagged prices are highly correlated)
        - Μειώνει διαστάσεις διατηρώντας διακύμανση / Reduces dimensionality while preserving variance
        - Μείωση θορύβου (μικρές συνιστώσες συχνά αντιπροσωπεύουν θόρυβο)
          Noise reduction (minor components often represent noise)
        - Υπολογιστική αποδοτικότητα με λιγότερα χαρακτηριστικά
          Computational efficiency with fewer features

        Μειονεκτήματα / Drawbacks:
        - Απώλεια ερμηνευσιμότητας (PCs είναι συνδυασμοί, όχι αρχικά χαρακτηριστικά)
          Loss of interpretability (PCs are combinations, not original features)
        - Υποθέτει γραμμικές σχέσεις / Assumes linear relationships
        - Ευαίσθητο σε κλιμάκωση χαρακτηριστικών (ήδη χειρισμένο στο pipeline μας)
          Sensitive to feature scaling (already handled in our pipeline)

        Για Πρόβλεψη Μετοχών / For Stock Prediction:
        - Καθυστερημένες τιμές (close_t-1, close_t-2, ...) είναι πολύ συσχετισμένες
          Lagged prices (close_t-1, close_t-2, ...) are highly correlated
        - PCA μπορεί να συλλάβει την "τάση" σε μία συνιστώσα
          PCA can capture the "trend" in one component
        - Χαρακτηριστικά όγκου μπορεί να σχηματίσουν ξεχωριστές συνιστώσες
          Volume features may form separate components
        - Συνήθως 3-5 συνιστώσες εξηγούν >95% διακύμανση
          Typically 3-5 components explain >95% variance

    Όριο Διακύμανσης / Variance Threshold:
        - 0.95 (95%): Διατηρεί περισσότερες πληροφορίες, μέτρια μείωση
                      Retains most information, moderate reduction
        - 0.90 (90%): Πιο επιθετική μείωση / More aggressive reduction
        - 0.99 (99%): Συντηρητικό, ελάχιστη μείωση / Conservative, minimal reduction
    """
    # Apply PCA
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    n_components = X_train_pca.shape[1]
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(f"  Αρχικά χαρακτηριστικά / Original features: {X_train.shape[1]}")
    print(f"  Επιλεγμένες συνιστώσες / Selected components: {n_components}")
    print(
        f"  Επεξηγούμενη διακύμανση / Variance explained: {cumulative_variance[-1]:.4f}"
    )
    print(
        f"  Μείωση / Reduction: {X_train.shape[1] - n_components} χαρακτηριστικά αφαιρέθηκαν / features removed"
    )
    print(f"\n  Διακύμανση ανά συνιστώσα / Variance per component:")
    for i, var in enumerate(explained_variance[:10], 1):  # Show first 10
        print(
            f"    PC{i}: {var:.4f} ({cumulative_variance[i-1]:.4f} αθροιστική / cumulative)"
        )

    return X_train_pca, X_val_pca, pca, n_components


def compute_cfs_merit(X, y, feature_indices):
    """
    Υπολογίζει βαθμό CFS (Επιλογή Χαρακτηριστικών Βασισμένη σε Συσχέτιση).
    Computes CFS (Correlation-based Feature Selection) merit score.

    Παράμετροι / Args:
        X: Πίνακας χαρακτηριστικών / Feature matrix
        y: Διάνυσμα στόχων / Target vector
        feature_indices: Δείκτες χαρακτηριστικών προς αξιολόγηση / Indices of features to evaluate

    Επιστρέφει / Returns:
        float: Βαθμός αξίας / Merit score

    Τύπος Αξίας CFS / CFS Merit Formula:
        Merit(S) = (k · r̄cf) / √(k + k(k-1) · r̄ff)

        όπου / where:
        - k: Αριθμός χαρακτηριστικών στο υποσύνολο S / Number of features in subset S
        - r̄cf: Μέση συσχέτιση μεταξύ χαρακτηριστικών και στόχου (κλάσης)
              Average correlation between features and target (class)
        - r̄ff: Μέση συσχέτιση μεταξύ χαρακτηριστικών (χαρακτηριστικό-χαρακτηριστικό)
              Average correlation between features (feature-feature)

    Ερμηνεία / Interpretation:
        - Αριθμητής: Ανταμείβει υψηλή συσχέτιση χαρακτηριστικών-στόχου
          Numerator: Rewards high feature-target correlation
        - Παρονομαστής: Ποινή σε υψηλή συσχέτιση χαρακτηριστικών-χαρακτηριστικών (πλεονασμός)
          Denominator: Penalizes high feature-feature correlation (redundancy)
        - Υψηλότερη αξία = καλύτερο υποσύνολο χαρακτηριστικών
          Higher merit = better feature subset
        - Ισορροπεί σχετικότητα (προς στόχο) με πλεονασμό (μεταξύ χαρακτηριστικών)
          Balances relevance (to target) with redundancy (among features)

    Γιατί το CFS είναι Αποτελεσματικό / Why CFS is Effective:
        - Επιλέγει χαρακτηριστικά υψηλής συσχέτισης με στόχο
          Selects features highly correlated with target
        - Αποφεύγει πλεονάζοντα χαρακτηριστικά (π.χ. close_t-1 και close_t-2 είναι όμοια)
          Avoids redundant features (e.g., close_t-1 and close_t-2 are similar)
        - Γρήγορος υπολογισμός (βασισμένος σε συσχέτιση, όχι εκπαίδευση μοντέλου)
          Fast computation (correlation-based, no model training)
        - Λειτουργεί καλά για γραμμικές σχέσεις / Works well for linear relationships

    Για Πρόβλεψη Μετοχών / For Stock Prediction:
        - Πρόσφατες υστερήσεις (close_t-1, close_t-2) υψηλής συσχέτισης με στόχο
          Recent lags (close_t-1, close_t-2) highly correlated with target
        - Αυτές οι υστερήσεις επίσης υψηλής συσχέτισης μεταξύ τους
          These lags also highly correlated with each other
        - CFS μπορεί να επιλέξει close_t-1 και να παραλείψει close_t-2 (πλεονάζον)
          CFS might select close_t-1 and skip close_t-2 (redundant)
        - Χαρακτηριστικά όγκου λιγότερο συσχετισμένα → μπορεί να συμπεριληφθούν εάν προσθέτουν νέες πληροφορίες
          Volume features less correlated → may be included if adding new info
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
    Εφαρμόζει CFS χρησιμοποιώντας προς-τα-εμπρός επιλογή βασισμένη σε αξία.
    Applies CFS using forward selection based on merit.

    Παράμετροι / Args:
        X_train: Χαρακτηριστικά εκπαίδευσης / Training features
        y_train: Στόχοι εκπαίδευσης / Training targets
        X_val: Χαρακτηριστικά επικύρωσης / Validation features
        feature_names: Ονόματα χαρακτηριστικών / Names of features
        max_features: Μέγιστα χαρακτηριστικά προς επιλογή (None = αυτόματο)
                      Maximum features to select (None = automatic)

    Επιστρέφει / Returns:
        tuple: (X_train_cfs, X_val_cfs, selected_indices, selected_names)

    Αλγόριθμος CFS (Προς-τα-Εμπρός Επιλογή) / CFS Algorithm (Forward Selection):
        1. Ξεκινήστε με κενό σύνολο χαρακτηριστικών / Start with empty feature set
        2. Για κάθε υπόλοιπο χαρακτηριστικό / For each remaining feature:
           - Δοκιμή προσθήκης του στο τρέχον σύνολο / Try adding it to current set
           - Υπολογισμός αξίας με αυτό το χαρακτηριστικό / Compute merit with this feature
        3. Προσθήκη χαρακτηριστικού που αυξάνει περισσότερο την αξία
           Add feature that most increases merit
        4. Επανάληψη μέχρι η αξία να σταματήσει να βελτιώνεται / Repeat until merit stops improving
        5. Επιστροφή καλύτερου υποσυνόλου χαρακτηριστικών / Return best feature subset

    Κριτήριο Διακοπής / Stopping Criterion:
        - Η αξία σταματά να αυξάνεται / Merit stops increasing
        - Ή έφτασε max_features / Or max_features reached

    Πολυπλοκότητα Χρόνου / Time Complexity:
        - O(n² · m) όπου / where n = χαρακτηριστικά / features, m = δείγματα / samples
        - Πολύ πιο γρήγορο από wrapper μεθόδους (όχι εκπαίδευση μοντέλου)
          Much faster than wrapper methods (no model training)
    """
    n_features = X_train.shape[1]

    # Initialize
    selected_indices = []
    remaining_indices = list(range(n_features))
    best_merit = 0.0

    print(f"  Έναρξη προς-τα-εμπρός επιλογής CFS / Starting CFS forward selection...")
    print(f"  Σύνολο χαρακτηριστικών / Total features: {n_features}")

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
                f"    Επανάληψη / Iteration {iteration}: Προστέθηκε χαρακτηριστικό / Added feature {best_idx} "
                f"({feature_names[best_idx]}), Αξία / Merit: {best_merit:.4f}"
            )

            # Check max_features
            if max_features and len(selected_indices) >= max_features:
                break
        else:
            # Merit didn't improve, stop
            print(
                f"    Επανάληψη / Iteration {iteration}: Καμία βελτίωση, διακοπή / No improvement, stopping"
            )
            break

    # Extract selected features
    selected_indices = sorted(selected_indices)
    selected_names = [feature_names[i] for i in selected_indices]

    X_train_cfs = X_train[:, selected_indices]
    X_val_cfs = X_val[:, selected_indices]

    print(f"\n  Αποτελέσματα CFS / CFS Results:")
    print(
        f"    Επιλέχθηκαν / Selected {len(selected_indices)} χαρακτηριστικά / features"
    )
    print(f"    Τελική αξία / Final merit: {best_merit:.4f}")
    print(f"    Επιλεγμένα χαρακτηριστικά / Selected features: {selected_names}")

    return X_train_cfs, X_val_cfs, selected_indices, selected_names


def apply_forward_selection(
    X_train, y_train, X_val, feature_names, n_features_to_select
):
    """
    Εφαρμόζει Διαδοχική Προς-τα-Εμπρός Επιλογή (wrapper μέθοδος).
    Applies Sequential Forward Selection (wrapper method).

    Παράμετροι / Args:
        X_train, y_train: Δεδομένα εκπαίδευσης / Training data
        X_val: Δεδομένα επικύρωσης / Validation data
        feature_names: Ονόματα χαρακτηριστικών / Feature names
        n_features_to_select: Αριθμός χαρακτηριστικών προς επιλογή / Number of features to select

    Επιστρέφει / Returns:
        tuple: (X_train_selected, X_val_selected, selected_indices, selected_names)

    Διαδοχική Προς-τα-Εμπρός Επιλογή (Wrapper Μέθοδος) / Sequential Forward Selection (Wrapper Method):

        Αλγόριθμος / Algorithm:
        1. Έναρξη με κενό σύνολο χαρακτηριστικών / Start with empty feature set
        2. Για κάθε υπόλοιπο χαρακτηριστικό / For each remaining feature:
           - Εκπαίδευση μοντέλου με τρέχοντα + αυτό το χαρακτηριστικό
             Train model with current features + this feature
           - Αξιολόγηση απόδοσης (cross-validation) / Evaluate performance (cross-validation)
        3. Προσθήκη χαρακτηριστικού που δίνει καλύτερη απόδοση
           Add feature that gives best performance
        4. Επανάληψη μέχρι επιλογή επιθυμητού αριθμού χαρακτηριστικών
           Repeat until desired number of features selected

        Αξιολόγηση / Evaluation:
        - Χρησιμοποιεί πραγματική απόδοση μοντέλου (όχι συσχέτιση)
          Uses actual model performance (not correlation)
        - Cross-validation για αποφυγή overfitting / to avoid overfitting
        - Για παλινδρόμηση: χρησιμοποιεί negative MSE / For regression: uses negative MSE as scoring metric

        Πλεονεκτήματα / Benefits:
        - Λαμβάνει υπόψη αλληλεπιδράσεις χαρακτηριστικών / Considers feature interactions
        - Βελτιστοποιεί για συγκεκριμένο μοντέλο (LinearRegression)
          Optimizes for specific model (LinearRegression)
        - Συνήθως δίνει καλύτερη προβλεπτική απόδοση / Usually gives best predictive performance

        Μειονεκτήματα / Drawbacks:
        - Υπολογιστικά επιβαρυντικό (εκπαιδεύει πολλά μοντέλα)
          Computationally expensive (trains many models)
        - Μπορεί να κάνει overfit στα δεδομένα εκπαίδευσης / Can overfit to training data
        - Ευαίσθητο σε cross-validation splits / Sensitive to cross-validation splits
        - Άπληστος αλγόριθμος (μπορεί να μη βρει καθολικό βέλτιστο)
          Greedy algorithm (may not find global optimum)

    Σύγκριση με CFS / Comparison with CFS:
        - CFS: Γρήγορο, βασισμένο σε συσχέτιση, όχι εκπαίδευση μοντέλου
              Fast, correlation-based, no model training
        - Wrapper: Αργό, βασισμένο σε μοντέλο, λαμβάνει υπόψη αλληλεπιδράσεις
                  Slow, model-based, considers interactions
        - CFS καλό για γρήγορη επιλογή / CFS good for quick feature selection
        - Wrapper καλύτερο για τελική βελτιστοποίηση μοντέλου
                  Wrapper better for final model optimization

    Πολυπλοκότητα Χρόνου / Time Complexity:
        - O(n² · model_training_time)
        - Με / With n=24 χαρακτηριστικά / features, επιλέγει / selects k=10 → ~170 μοντέλα εκπαιδεύονται / models trained
    """
    print(
        f"  Έναρξη Διαδοχικής Προς-τα-Εμπρός Επιλογής / Starting Sequential Forward Selection..."
    )
    print(f"  Στόχος / Target: {n_features_to_select} χαρακτηριστικά / features")
    print(f"  Αυτό μπορεί να πάρει ένα λεπτό / This may take a minute...")

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

    print(f"\n  Αποτελέσματα Προς-τα-Εμπρός Επιλογής / Forward Selection Results:")
    print(
        f"    Επιλέχθηκαν / Selected {len(selected_indices)} χαρακτηριστικά / features"
    )
    print(f"    Επιλεγμένα χαρακτηριστικά / Selected features: {selected_names}")

    return X_train_selected, X_val_selected, selected_indices, selected_names


def train_and_evaluate(X_train, y_train, X_val, y_val, method_name):
    """
    Εκπαιδεύει γραμμική παλινδρόμηση και αξιολογεί και στα δύο σύνολα.
    Trains linear regression and evaluates on both sets.

    Παράμετροι / Args:
        X_train, y_train: Δεδομένα εκπαίδευσης / Training data
        X_val, y_val: Δεδομένα επικύρωσης / Validation data
        method_name: Όνομα για αναφορά / Name for reporting

    Επιστρέφει / Returns:
        dict: Αποτελέσματα συμπεριλαμβανομένου μοντέλου και μετρικών / Results including model and metrics
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

    print(f"\n{method_name} - Απόδοση Μοντέλου / Model Performance:")
    print(f"  Εκπαίδευση / Training:")
    print(f"    RMSE: ${train_metrics['RMSE']:.2f}")
    print(f"    MAE:  ${train_metrics['MAE']:.2f}")
    print(f"    R²:   {train_metrics['R2']:.4f}")
    print(f"  Επικύρωση / Validation:")
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
    Δημιουργεί οπτικοποίηση που συγκρίνει όλες τις μεθόδους μείωσης διαστάσεων.
    Creates visualization comparing all dimensionality reduction methods.

    Παράμετροι / Args:
        results_list: Λίστα λεξικών αποτελεσμάτων / List of result dictionaries
        output_dir: Κατάλογος εξόδου / Output directory
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
    print(f"✓ Αποθηκεύτηκε γράφημα σύγκρισης / Saved comparison plot: {plot_path}")
    plt.close()


def save_results(results_list, output_dir="results"):
    """
    Αποθηκεύει αποτελέσματα μείωσης διαστάσεων.
    Saves dimensionality reduction results.
    """
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
    print(f"✓ Αποθηκεύτηκε πίνακας αποτελεσμάτων / Saved results table: {csv_path}")

    # Save models
    models_dir = "models"
    for r in results_list:
        method_name = r["method"].replace(" ", "_").lower()
        model_path = os.path.join(models_dir, f"{method_name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(r, f)
        print(
            f"✓ Αποθηκεύτηκε {r['method']} μοντέλο / Saved {r['method']} model: {model_path}"
        )


def main():
    """
    Κύρια συνάρτηση εκτέλεσης για ανάλυση μείωσης διαστάσεων.
    Main execution function for dimensionality reduction analysis.

    Στρατηγική / Strategy:
    1. Δοκιμή ΌΛΩΝ των 16 baseline ρυθμίσεων / Test ALL 16 baseline configurations
    2. Εφαρμογή PCA (95% διακύμανση), CFS, και Διαδοχική Προς-τα-Εμπρός Επιλογή σε κάθε ρύθμιση
       Apply PCA (95% variance), CFS, and Sequential Forward Selection to each
    3. Σύγκριση αποτελεσμάτων σε όλες τις μεθόδους και ρυθμίσεις
       Compare results across all methods and configurations
    """
    print("=" * 80)
    print("ΠΡΟΒΛΕΨΗ ΤΙΜΗΣ ΜΕΤΟΧΗΣ NFLX - ΜΕΙΩΣΗ ΔΙΑΣΤΑΣΕΩΝ (ΕΡΓΑΣΙΑ Γ)")
    print("NFLX STOCK PRICE PREDICTION - DIMENSIONALITY REDUCTION (TASK C)")
    print("=" * 80)
    print()

    # Find all feature configurations
    features_dir = "features"
    feature_files = [
        f
        for f in os.listdir(features_dir)
        if f.startswith("features_") and f.endswith(".npz")
    ]

    print(
        f"Βρέθηκαν / Found {len(feature_files)} ρυθμίσεις χαρακτηριστικών / feature configurations"
    )
    print()

    # Results storage
    all_configs_results = []
    all_dim_red_models = {}

    # Process each configuration
    for feature_file in feature_files:
        # Extract config name
        config_name = feature_file.replace("features_", "").replace(".npz", "")
        parts = config_name.split("_")
        smoothing = parts[0]
        n_lags = int(parts[1].replace("lags", ""))

        print("=" * 80)
        print(f"Επεξεργασία / Processing: {smoothing}, {n_lags} υστερήσεις / lags")
        print("=" * 80)

        # Load data
        features_path = os.path.join(features_dir, feature_file)
        data = load_feature_set(features_path)
        print(f"✓ Φορτώθηκε / Loaded: {features_path}")
        print(
            f"  Εκπαίδευση / Training: {len(data['y_train'])} δείγματα / samples, {data['X_train'].shape[1]} χαρακτηριστικά / features"
        )
        print(f"  Επικύρωση / Validation: {len(data['y_val'])} δείγματα / samples")

        # 1. PCA
        print("\n  Εφαρμογή PCA (95% διακύμανση) / Applying PCA (95% variance)...")
        X_train_pca, X_val_pca, pca_model, n_components = apply_pca(
            data["X_train"], data["X_val"], variance_threshold=0.95
        )
        result_pca = train_and_evaluate(
            X_train_pca, data["y_train"], X_val_pca, data["y_val"], "PCA"
        )

        # 2. CFS
        print(
            "\n  Εφαρμογή CFS (Επιλογή Βασισμένη σε Συσχέτιση) / Applying CFS (Correlation-based Feature Selection)..."
        )
        X_train_cfs, X_val_cfs, cfs_indices, cfs_names = apply_cfs(
            data["X_train"], data["y_train"], data["X_val"], data["feature_names"]
        )
        result_cfs = train_and_evaluate(
            X_train_cfs, data["y_train"], X_val_cfs, data["y_val"], "CFS"
        )

        # 3. Sequential Forward Selection
        print(
            "\n  Εφαρμογή Διαδοχικής Προς-τα-Εμπρός Επιλογής / Applying Sequential Forward Selection..."
        )
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

        # Store results for this configuration
        for method, result in [
            ("PCA", result_pca),
            ("CFS", result_cfs),
            ("Forward_Selection", result_sfs),
        ]:
            all_configs_results.append(
                {
                    "smoothing": smoothing,
                    "n_lags": n_lags,
                    "method": method,
                    "n_features": result["n_features"],
                    "train_rmse": result["train_metrics"]["RMSE"],
                    "train_mae": result["train_metrics"]["MAE"],
                    "train_r2": result["train_metrics"]["R2"],
                    "val_rmse": result["val_metrics"]["RMSE"],
                    "val_mae": result["val_metrics"]["MAE"],
                    "val_r2": result["val_metrics"]["R2"],
                }
            )

        # Store models
        all_dim_red_models[config_name] = {
            "pca": {"model": pca_model, "result": result_pca},
            "cfs": {"indices": cfs_indices, "names": cfs_names, "result": result_cfs},
            "sfs": {"indices": sfs_indices, "names": sfs_names, "result": result_sfs},
        }

        print(
            f"\n✓ Η ρύθμιση ολοκληρώθηκε / Configuration complete: {smoothing}, {n_lags} υστερήσεις / lags"
        )
        print(
            f"  PCA: {n_components} συνιστώσες / components, Επικύρωση RMSE / Val RMSE: ${result_pca['val_metrics']['RMSE']:.2f}"
        )
        print(
            f"  CFS: {len(cfs_indices)} χαρακτηριστικά / features, Επικύρωση RMSE / Val RMSE: ${result_cfs['val_metrics']['RMSE']:.2f}"
        )
        print(
            f"  SFS: {len(sfs_indices)} χαρακτηριστικά / features, Επικύρωση RMSE / Val RMSE: ${result_sfs['val_metrics']['RMSE']:.2f}"
        )
        print()

    # Create results DataFrame
    results_df = pd.DataFrame(all_configs_results)
    results_df = results_df.sort_values("val_rmse")

    # Display summary
    print("=" * 80)
    print("ΣΥΝΟΨΗ ΟΛΩΝ ΤΩΝ ΡΥΘΜΙΣΕΩΝ - ΜΕΙΩΣΗ ΔΙΑΣΤΑΣΕΩΝ")
    print("ALL CONFIGURATIONS SUMMARY - DIMENSIONALITY REDUCTION")
    print("=" * 80)
    print("\nΤοπ 10 Μοντέλα βάσει Επικύρωσης RMSE / Top 10 Models by Validation RMSE:")
    print(results_df.head(10).to_string(index=False))
    print()

    # Save results
    print("=" * 80)
    print("ΑΠΟΘΗΚΕΥΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ / SAVING RESULTS")
    print("=" * 80)

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save all results table
    csv_path = "results/dimensionality_reduction_all_models_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Αποθηκεύτηκε πίνακας αποτελεσμάτων / Saved results table: {csv_path}")

    # Save all models
    models_path = "models/all_dimensionality_reduction_models.pkl"
    with open(models_path, "wb") as f:
        pickle.dump(all_dim_red_models, f)
    print(f"✓ Αποθηκεύτηκαν όλα τα μοντέλα / Saved all models: {models_path}")

    # Final summary
    best = results_df.iloc[0]
    print()
    print("=" * 80)
    print(
        "✓ Η ΑΝΑΛΥΣΗ ΜΕΙΩΣΗΣ ΔΙΑΣΤΑΣΕΩΝ ΟΛΟΚΛΗΡΩΘΗΚΕ / DIMENSIONALITY REDUCTION ANALYSIS COMPLETED"
    )
    print("=" * 80)
    print(
        f"\nΔοκιμάστηκαν / Tested {len(feature_files)} ρυθμίσεις / configurations × 3 μέθοδοι / methods = {len(results_df)} μοντέλα / models"
    )
    print(f"\nΚαλύτερο Συνολικό Μοντέλο (Best Overall Model):")
    print(
        f"  Ρύθμιση (Configuration): {best['smoothing']}, {best['n_lags']} υστερήσεις (lags)"
    )
    print(f"  Μέθοδος (Method): {best['method']}")
    print(f"  Χαρακτηριστικά (Features): {best['n_features']}")
    print(f"  Επικύρωση RMSE (Validation RMSE): ${best['val_rmse']:.2f}")
    print(f"  Επικύρωση R² (Validation R²): {best['val_r2']:.4f}")
    print()
    print(
        "Επόμενο Βήμα (Next Step): Εκτέλεση (Run) step6_future_predictions.py για ολοκληρωμένη σύγκριση (for comprehensive comparison)"
    )


if __name__ == "__main__":
    main()
