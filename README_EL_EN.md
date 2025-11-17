# NFLX: Πρόβλεψη Τιμών Μετοχών με Γραμμική Παλινδρόμηση

# NFLX: Stock Price Prediction with Linear Regression

**Στατιστικές Μέθοδοι Μηχανικής Μάθησης - Εργασία 1**  
**Statistical Methods of Machine Learning - Task 1**

---

## 📊 Επισκόπηση Έργου / Project Overview

### Ελληνικά

Αυτό το έργο υλοποιεί ένα ολοκληρωμένο σύστημα μηχανικής μάθησης για την πρόβλεψη τιμών μετοχών της Netflix (NFLX) χρησιμοποιώντας γραμμική παλινδρόμηση και προηγμένες τεχνικές. Το έργο αντιμετωπίζει τέσσερις βασικές εργασίες:

- **Εργασία Α**: Baseline Γραμμική Παλινδρόμηση με χαρακτηριστικά υστέρησης
- **Εργασία Β**: Πολυωνυμική Παλινδρόμηση με κανονικοποίηση L1/L2
- **Εργασία Γ**: Μείωση Διαστάσεων (PCA, CFS, Wrapper Methods)
- **Εργασία Δ**: Πρόβλεψη Τιμών για Δεκέμβριο 2025 και Ιανουάριο 2026

### English

This project implements a comprehensive machine learning pipeline for predicting Netflix (NFLX) stock prices using linear regression and advanced techniques. The project addresses four core tasks:

- **Task A**: Baseline Linear Regression with lagged features
- **Task B**: Polynomial Regression with L1/L2 Regularization
- **Task C**: Dimensionality Reduction (PCA, CFS, Wrapper Methods)
- **Task D**: Price Predictions for December 2025 and January 2026

---

## 🎯 Βασικά Αποτελέσματα / Key Results

### Καλύτερο Μοντέλο / Best Model

| Μετρική / Metric              | Τιμή / Value                   |
| ----------------------------- | ------------------------------ |
| **Εξομάλυνση / Smoothing**    | sigma3 (Gaussian σ=3)          |
| **Υστερήσεις / Lags**         | 12 μήνες / 12 months           |
| **Χαρακτηριστικά / Features** | 24 (12 close + 12 volume lags) |
| **Training RMSE**             | $0.02                          |
| **Validation RMSE**           | $0.03                          |
| **Training R²**               | 1.0000                         |
| **Validation R²**             | 1.0000                         |

### Προβλέψεις Μελλοντικών Τιμών / Future Price Predictions

| Μήνας / Month                       | Προβλεφθείσα Τιμή / Predicted Price                |
| ----------------------------------- | -------------------------------------------------- |
| **Δεκέμβριος 2025 / December 2025** | **$1,175.48**                                      |
| **Ιανουάριος 2026 / January 2026**  | **$1,175.95** (καταρρακτώδης πρόβλεψη / cascading) |

---

## 📁 Δομή Έργου / Project Structure

```
stock-price-linear-regression/
│
├── 📄 README_EL_EN.md                          # Αυτό το αρχείο / This file
├── 📄 README.md                                # Αγγλικό README / English README
├── 📄 ML_TERMINOLOGY_GLOSSARY_EL_EN.md         # Γλωσσάρι όρων / Terms glossary
├── 📄 statistical_methods_of_ml.md             # Περιγραφή εργασίας / Assignment description
├── 📄 .env                                     # Κλειδί API / API key
│
├── 📜 Scripts Python / Python Scripts:
│   ├── step1_data_acquisition.py               # Συλλογή δεδομένων / Data acquisition
│   ├── step2_feature_engineering.py            # Μηχανική χαρακτηριστικών / Feature engineering
│   ├── step3_baseline_linear_regression.py     # Εργασία Α / Task A
│   ├── step4_polynomial_regression_regularization.py  # Εργασία Β / Task B
│   ├── step5_dimensionality_reduction.py       # Εργασία Γ / Task C
│   ├── step6_future_predictions.py             # Εργασία Δ (βασική) / Task D (basic)
│   └── step6_future_predictions_improved.py    # Εργασία Δ (βελτιωμένη) / Task D (improved)
│
├── 📂 data/                                    # Ακατέργαστα & επεξεργασμένα δεδομένα
│   ├── nflx_monthly_raw.csv                   # Raw data
│   ├── nflx_monthly_smoothed_sigma1.csv       # Εξομάλυνση σ=1 / Smoothing σ=1
│   ├── nflx_monthly_smoothed_sigma2.csv       # Εξομάλυνση σ=2 / Smoothing σ=2
│   └── nflx_monthly_smoothed_sigma3.csv       # Εξομάλυνση σ=3 / Smoothing σ=3
│
├── 📂 features/                                # Πίνακες χαρακτηριστικών / Feature matrices
│   ├── features_*.npz (16 ρυθμίσεις / 16 configurations)
│   ├── scaler_*.pkl (Scalers)
│   └── metadata_*.csv (Μεταδεδομένα / Metadata)
│
├── 📂 models/                                  # Εκπαιδευμένα μοντέλα / Trained models
│   └── best_baseline_linear_regression.pkl
│
├── 📂 results/                                 # Αποτελέσματα & απεικονίσεις / Results & visualizations
│   ├── 📊 Γραφήματα / Plots:
│   │   ├── comprehensive_predictions_comparison.png
│   │   ├── validation_rmse_comparison_improved.png
│   │   ├── best_model_forecast_with_history.png
│   │   ├── baseline_performance_by_config.png
│   │   ├── polynomial_regularization_paths.png
│   │   └── dimensionality_reduction_comparison.png
│   │
│   ├── 📋 Πίνακες CSV / CSV Tables:
│   │   ├── all_models_predictions.csv
│   │   ├── baseline_linear_regression_results.csv
│   │   ├── polynomial_regression_comparison.csv
│   │   └── dimensionality_reduction_results.csv
│   │
│   └── 📄 Αναφορές / Reports:
│       ├── FINAL_PREDICTIONS_REPORT_EL_EN.txt  # Δίγλωσση αναφορά / Bilingual report
│       └── FINAL_PROJECT_SUMMARY.txt           # Αγγλική περίληψη / English summary
│
└── 📂 Provided Code/                           # Παραδείγματα καθηγητή / Teacher's examples
    └── *.ipynb (11 notebooks)
```

---

## 🚀 Εγκατάσταση & Εκτέλεση / Installation & Execution

### Προαπαιτούμενα / Prerequisites

```bash
Python 3.8+
```

### Εγκατάσταση Βιβλιοθηκών / Install Libraries

```bash
pip install numpy pandas scikit-learn scipy matplotlib requests python-dateutil
```

### Ρύθμιση API / API Setup

1. Εγγραφή στο Alpha Vantage: https://www.alphavantage.co/
   Sign up at Alpha Vantage: https://www.alphavantage.co/

2. Δημιουργία αρχείου `.env`:
   Create `.env` file:

```
api_key=YOUR_API_KEY_HERE
```

### Πλήρης Εκτέλεση / Full Execution

```bash
# Βήμα 1: Συλλογή Δεδομένων / Step 1: Data Acquisition
python step1_data_acquisition.py

# Βήμα 2: Μηχανική Χαρακτηριστικών / Step 2: Feature Engineering
python step2_feature_engineering.py

# Βήμα 3: Baseline Παλινδρόμηση (Εργασία Α) / Step 3: Baseline Regression (Task A)
python step3_baseline_linear_regression.py

# Βήμα 4: Πολυωνυμική Παλινδρόμηση (Εργασία Β) / Step 4: Polynomial Regression (Task B)
python step4_polynomial_regression_regularization.py

# Βήμα 5: Μείωση Διαστάσεων (Εργασία Γ) / Step 5: Dimensionality Reduction (Task C)
python step5_dimensionality_reduction.py

# Βήμα 6: Προβλέψεις Μέλλοντος (Εργασία Δ) / Step 6: Future Predictions (Task D)
# Βελτιωμένη έκδοση - δοκιμάζει ΟΛΑ τα μοντέλα / Improved version - tests ALL models
python step6_future_predictions_improved.py
```

---

## 📊 Μεθοδολογία / Methodology

### 1. Συλλογή & Προεπεξεργασία Δεδομένων / Data Acquisition & Preprocessing

#### Ελληνικά

- **Πηγή**: Alpha Vantage API (TIME_SERIES_DAILY)
- **Χρονικό Εύρος**: Μάιος 2002 - Νοέμβριος 2025 (283 μήνες)
- **Συγκέντρωση**: Ημερήσια → Μηνιαία μέσα (close, volume)
- **Εξομάλυνση**: Φίλτρο Gauss με σ ∈ {1, 2, 3}
- **Αιτιολόγηση**: Μειώνει θόρυβο διατηρώντας τάσεις

#### English

- **Source**: Alpha Vantage API (TIME_SERIES_DAILY)
- **Time Range**: May 2002 - November 2025 (283 months)
- **Aggregation**: Daily → Monthly averages (close, volume)
- **Smoothing**: Gaussian filter with σ ∈ {1, 2, 3}
- **Rationale**: Reduces noise while preserving trends

### 2. Μηχανική Χαρακτηριστικών / Feature Engineering

#### Ελληνικά

- **Χαρακτηριστικά Υστέρησης**:
  - `close_t-1` έως `close_t-N`: Προηγούμενες τιμές κλεισίματος
  - `volume_t-1` έως `volume_t-N`: Προηγούμενοι όγκοι συναλλαγών
- **Δοκιμασμένα Παράθυρα**: N ∈ {3, 6, 9, 12} μήνες
- **Κλιμάκωση**: StandardScaler (z-score κανονικοποίηση)
- **Διαχωρισμός Δεδομένων**:
  - Εκπαίδευση: Πριν το 2025 (260-269 δείγματα)
  - Επικύρωση: 2025 (11 δείγματα)
  - **Κρίσιμο**: Χρονολογικός διαχωρισμός (χωρίς ανακάτεμα)

#### English

- **Lagged Features**:
  - `close_t-1` through `close_t-N`: Past closing prices
  - `volume_t-1` through `volume_t-N`: Past trading volumes
- **Tested Windows**: N ∈ {3, 6, 9, 12} months
- **Scaling**: StandardScaler (z-score normalization)
- **Data Split**:
  - Training: Pre-2025 (260-269 samples)
  - Validation: 2025 (11 samples)
  - **Critical**: Chronological split (no shuffling)

### 3. Εκπαίδευση & Αξιολόγηση Μοντέλων / Model Training & Evaluation

#### Εργασία Α: Baseline Γραμμική Παλινδρόμηση / Task A: Baseline Linear Regression

**Ελληνικά:**

- Δοκιμασμένες Ρυθμίσεις: 16 (4 επίπεδα εξομάλυνσης × 4 παράθυρα υστέρησης)
- Μοντέλο: Απλή Γραμμική Παλινδρόμηση (OLS)
- Μετρικές: RMSE, MAE, R²
- Καλύτερη: sigma3, 12 lags → RMSE $0.03, R² 1.0000

**English:**

- Configurations Tested: 16 (4 smoothing levels × 4 lag windows)
- Model: Ordinary Least Squares (OLS) Linear Regression
- Metrics: RMSE, MAE, R²
- Best: sigma3, 12 lags → RMSE $0.03, R² 1.0000

#### Εργασία Β: Πολυωνυμική Παλινδρόμηση / Task B: Polynomial Regression

**Ελληνικά:**

- Βαθμός: 2 (24 χαρακτηριστικά → 325 χαρακτηριστικά)
- **Ridge (L2)**:
  - Καλύτερο α: 0.1
  - Val RMSE: $8.98
  - Όλα τα χαρακτηριστικά διατηρούνται
- **Lasso (L1)**:
  - Καλύτερο α: 0.001
  - Val RMSE: $9.47
  - 263/325 χαρακτηριστικά επιλέχθηκαν (19.1% αραίωση)
- Συμπέρασμα: Το baseline υπερτερεί λόγω αποτελεσματικής εξομάλυνσης

**English:**

- Degree: 2 (24 features → 325 features)
- **Ridge (L2)**:
  - Best α: 0.1
  - Val RMSE: $8.98
  - All features retained
- **Lasso (L1)**:
  - Best α: 0.001
  - Val RMSE: $9.47
  - 263/325 features selected (19.1% sparsity)
- Conclusion: Baseline outperforms due to effective smoothing

#### Εργασία Γ: Μείωση Διαστάσεων / Task C: Dimensionality Reduction

| Μέθοδος / Method                    | Χαρακτηριστικά / Features            | Val RMSE | Val R² |
| ----------------------------------- | ------------------------------------ | -------- | ------ |
| **Baseline**                        | 24 (πλήρη / full)                    | $0.03    | 1.0000 |
| **PCA (95% διακύμανση / variance)** | 3 συνιστώσες / components            | $131.07  | -1.17  |
| **CFS**                             | 1 (close_t-1)                        | $21.91   | 0.9392 |
| **Sequential Forward Selection**    | 12 (όλα close lags / all close lags) | $0.03    | 1.0000 |

**Συμπέρασμα / Conclusion**: Οι υστερήσεις close επαρκούν· ο όγκος προσθέτει ελάχιστη αξία / Close lags sufficient; volume adds minimal value

#### Εργασία Δ: Προβλέψεις Μέλλοντος / Task D: Future Predictions

**Ελληνικά:**

- **Μέθοδος**: Καλύτερο baseline μοντέλο (sigma3, 12 lags)
- **Δεκέμβριος 2025**: $1,175.48 (άμεση πρόβλεψη)
- **Ιανουάριος 2026**: $1,175.95 (καταρρακτώδης πρόβλεψη)
- **Καταρρακτώδης Προσέγγιση**: Χρησιμοποιεί την πρόβλεψη Δεκεμβρίου ως χαρακτηριστικό υστέρησης
- **Σημείωση**: Μειωμένη ακρίβεια λόγω πολλαπλασιασμού σφαλμάτων

**English:**

- **Method**: Best baseline model (sigma3, 12 lags)
- **December 2025**: $1,175.48 (direct prediction)
- **January 2026**: $1,175.95 (cascading prediction)
- **Cascading Approach**: Uses December prediction as lag feature
- **Note**: Reduced accuracy due to error compounding

---

## 📈 Βασικά Ευρήματα / Key Findings

### Ελληνικά

1. **Η Προεπεξεργασία είναι Κρίσιμη**: Η βαριά εξομάλυνση Gauss (σ=3) ήταν ο πιο σημαντικός παράγοντας επιτυχίας, μετατρέποντας θορυβώδη δεδομένα σε πολύ προβλέψιμα μοτίβα.

2. **Επαρκή τα Γραμμικά Μοντέλα**: Με σωστή προεπεξεργασία, η απλή γραμμική παλινδρόμηση πέτυχε σχεδόν τέλεια αποτελέσματα. Τα σύνθετα πολυωνυμικά χαρακτηριστικά ήταν περιττά.

3. **Βέλτιστο Παράθυρο Αναδρομής**: Το παράθυρο 12 μηνών συνέλαβε αποτελεσματικά τόσο την βραχυπρόθεσμη ορμή όσο και τις μακροπρόθεσμες τάσεις.

4. **Σημασία Χαρακτηριστικών**: Οι υστερήσεις τιμών κλεισίματος πολύ πιο ενημερωτικές από τον όγκο. Η Sequential Forward Selection το επιβεβαίωσε επιλέγοντας μόνο υστερήσεις close.

5. **Περιορισμοί PCA**: Η PCA απέτυχε σε βαριά εξομαλυμένα δεδομένα επειδή:
   - Η εξομάλυνση ήδη μείωσε εννοιολογικά τις διαστάσεις
   - Ο γραμμικός μετασχηματισμός δεν μπορούσε να βελτιώσει τα εξομαλυμένα χαρακτηριστικά
   - Κρίσιμες χρονικές πληροφορίες χάθηκαν στον μετασχηματισμό

### English

1. **Preprocessing is Critical**: Heavy Gaussian smoothing (σ=3) was the most important success factor, transforming noisy data into highly predictable patterns.

2. **Linear Models Sufficient**: With proper preprocessing, simple linear regression achieved near-perfect results. Complex polynomial features were unnecessary.

3. **Optimal Lookback Window**: The 12-month lag window effectively captured both short-term momentum and long-term trends.

4. **Feature Importance**: Close price lags far more informative than volume. Sequential Forward Selection confirmed this by selecting only close lags.

5. **PCA Limitations**: PCA failed on heavily smoothed data because:
   - Smoothing already reduced dimensionality conceptually
   - Linear transformation couldn't improve on smoothed features
   - Critical temporal information was lost in transformation

---

## 📊 Απεικονίσεις / Visualizations

### Βασικά Γραφήματα / Main Plots

1. **comprehensive_predictions_comparison.png**

   - Σύγκριση όλων των 16 μοντέλων / All 16 models comparison
   - Προβλέψεις Δεκεμβρίου & Ιανουαρίου / December & January predictions
   - RMSE vs προβλέψεις / RMSE vs predictions
   - Heatmap τιμών / Price heatmap

2. **validation_rmse_comparison_improved.png**

   - Βελτιωμένη οπτικοποίηση RMSE / Improved RMSE visualization
   - Όλες οι ρυθμίσεις με χρώματα / All configurations color-coded
   - Επισήμανση καλύτερου μοντέλου / Best model highlighted
   - **Διόρθωση**: Σωστές συνδέσεις γραμμών / **Fixed**: Correct line connections

3. **best_model_forecast_with_history.png**
   - Ιστορικά δεδομένα + προβλέψεις / Historical data + predictions
   - Αννοτάτε προβλέψεις / Annotated predictions
   - Πλήρης χρονοσειρά / Full time series

---

## 🎓 Τεχνικές Λεπτομέρειες / Technical Details

### Πολυπλοκότητα Υπολογισμών / Computational Complexity

- **Συλλογή Δεδομένων / Data Acquisition**: O(n) κλήσεις API + επεξεργασία / O(n) API calls + processing
- **Μηχανική Χαρακτηριστικών / Feature Engineering**: O(n × m) όπου n=δείγματα, m=χαρακτηριστικά / where n=samples, m=features
- **Γραμμική Παλινδρόμηση / Linear Regression**: O(m² × n) για λύση OLS / for OLS solution
- **Πολυωνυμική (βαθμός 2) / Polynomial (degree 2)**: O(m⁴ × n)
- **Sequential Forward Selection**: O(m² × k) εκπαιδεύσεις μοντέλων / model trainings

### Απαιτήσεις Μνήμης / Memory Requirements

- **Ακατέργαστα Δεδομένα / Raw Data**: ~5,000 ημερήσια αρχεία → ~1 MB / ~5,000 daily records → ~1 MB
- **Πίνακες Χαρακτηριστικών / Feature Matrices**: 16 ρυθμίσεις × 2 σύνολα → ~10 MB / 16 configurations × 2 sets → ~10 MB
- **Μοντέλα / Models**: < 1 MB συνολικά / < 1 MB total

### Χρόνος Εκτέλεσης / Runtime (Προσεγγιστικός / Approximate)

- Βήμα 1 / Step 1: 30-60 δευτερόλεπτα / seconds (κλήση API / API call)
- Βήμα 2 / Step 2: 5-10 δευτερόλεπτα / seconds
- Βήμα 3 / Step 3: 2-3 δευτερόλεπτα / seconds
- Βήμα 4 / Step 4: 5-10 δευτερόλεπτα / seconds
- Βήμα 5 / Step 5: 30-60 δευτερόλεπτα / seconds (Sequential Selection)
- Βήμα 6 (βελτιωμένο) / Step 6 (improved): 10-15 δευτερόλεπτα / seconds

**Συνολικός Χρόνος / Total Runtime**: ~2-3 λεπτά / minutes

---

## ⚠️ Περιορισμοί / Limitations

### Περιορισμοί Μοντέλου / Model Limitations

#### Ελληνικά

1. **Ανταλλαγή Βαριάς Εξομάλυνσης**: Μπορεί να καθυστερήσει την αντίδραση σε ξαφνικές αλλαγές αγοράς
2. **Γραμμική Υπόθεση**: Υποθέτει ότι τα παρελθοντικά μοτίβα συνεχίζονται
3. **Εξωτερικά Γεγονότα**: Δεν μπορεί να συλλάβει ανακοινώσεις κερδών, κραχ αγοράς, ειδήσεις
4. **Περιορισμένα Δεδομένα Επικύρωσης**: Μόνο 11 μήνες δεδομένων 2025
5. **Καταρρακτώδης Πρόβλεψη**: Η πρόβλεψη Ιανουαρίου έχει αβεβαιότητα λόγω χρήσης προβλεφθέντος Δεκεμβρίου

#### English

1. **Heavy Smoothing Trade-off**: May delay reaction to sudden market changes
2. **Linear Assumption**: Assumes past patterns continue
3. **External Events**: Cannot capture earnings reports, market crashes, news
4. **Limited Validation Data**: Only 11 months of 2025 data
5. **Cascading Prediction**: January prediction has uncertainty due to using predicted December

---

## 💡 Συστάσεις / Recommendations

### Για Παραγωγική Χρήση / For Production Use

#### Ελληνικά

1. **Συχνότητα Ενημέρωσης**: Επανεκπαίδευση μοντέλου μηνιαία με νέα δεδομένα
2. **Παρακολούθηση**: Παρακολούθηση σφαλμάτων πρόβλεψης για ανίχνευση αλλαγών καθεστώτος
3. **Ensemble**: Συνδυασμός προβλέψεων από πολλαπλές τιμές σ
4. **Διαστήματα Εμπιστοσύνης**: Προσθήκη bootstrap ή Bayesian μεθόδων
5. **Επέκταση Χαρακτηριστικών**: Συμπερίληψη δεικτών αγοράς (S&P 500), απόδοσης κλάδου

#### English

1. **Update Frequency**: Retrain model monthly with new data
2. **Monitoring**: Track prediction errors to detect regime changes
3. **Ensemble**: Combine predictions from multiple σ values
4. **Confidence Intervals**: Add bootstrap or Bayesian methods
5. **Feature Expansion**: Include market indices (S&P 500), sector performance

---

## 📚 Αναφορές / References

### Απαιτήσεις Εργασίας / Assignment Requirements

- **Μάθημα / Course**: Στατιστικές Μέθοδοι Μηχανικής Μάθησης / Statistical Methods of Machine Learning
- **Εργασία / Task**: Πρόβλεψη Τιμών Μετοχών με Γραμμική Παλινδρόμηση / Predicting Stock Prices with Linear Regression
- **Σύμβολο / Symbol**: NFLX (Netflix, Inc.)
- **API**: Alpha Vantage (https://www.alphavantage.co/)

### Βασικές Βιβλιοθήκες / Key Libraries

- **scikit-learn**: Μοντέλα & μετρικές μηχανικής μάθησης / Machine learning models & metrics
- **pandas**: Χειρισμός & ανάλυση δεδομένων / Data manipulation & analysis
- **numpy**: Αριθμητικοί υπολογισμοί / Numerical computing
- **scipy**: Φιλτράρισμα Gauss / Gaussian filtering (scipy.ndimage.gaussian_filter1d)
- **matplotlib**: Οπτικοποίηση / Visualization

---

## 📞 Υποστήριξη / Support

### Για Ερωτήσεις / For Questions

1. Ελέγξτε το / Check `FINAL_PREDICTIONS_REPORT_EL_EN.txt` για λεπτομερή αποτελέσματα / for detailed results
2. Ανατρέξτε στο γλωσσάρι / Consult glossary: `ML_TERMINOLOGY_GLOSSARY_EL_EN.md`
3. Διαβάστε σχόλια κώδικα / Review inline code comments (εκτενής τεκμηρίωση / extensive documentation)

---

## 📄 Άδεια / License

Αυτό το έργο δημιουργήθηκε για ακαδημαϊκούς σκοπούς ως μέρος μιας εργασίας μαθήματος Μηχανικής Μάθησης.

This project was created for academic purposes as part of a Machine Learning course assignment.

---

**Τελευταία Ενημέρωση / Last Updated**: 17 Νοεμβρίου 2025 / November 17, 2025  
**Κατάσταση Έργου / Project Status**: Ολοκληρωμένο ✓ / Complete ✓

Όλες οι εργασίες (Α, Β, Γ, Δ) υλοποιήθηκαν επιτυχώς με εκτενή τεκμηρίωση.  
All tasks (A, B, C, D) successfully implemented with extensive documentation.
