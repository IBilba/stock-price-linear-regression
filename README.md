# ΠΡΟΒΛΕΨΗ ΤΙΜΩΝ ΜΕΤΟΧΩΝ NFLX ΜΕ ΓΡΑΜΜΙΚΗ ΠΑΛΙΝΔΡΟΜΗΣΗ

# NFLX Stock Price Prediction with Linear Regression

**Στατιστικές Μέθοδοι Μηχανικής Μάθησης - Εργασία 1**  
**Statistical Methods of Machine Learning - Task 1**

---

## ΕΠΙΣΚΟΠΗΣΗ ΕΡΓΟΥ / PROJECT OVERVIEW

Αυτό το έργο υλοποιεί μια ολοκληρωμένη διαδικασία μηχανικής μάθησης για πρόβλεψη τιμών μετοχών Netflix (NFLX) χρησιμοποιώντας γραμμική παλινδρόμηση και διάφορες προηγμένες τεχνικές.

This project implements a comprehensive machine learning pipeline to predict Netflix (NFLX) stock prices using linear regression and various advanced techniques.

### Εργασίες / Tasks

- **Εργασία Α / Task A**: Baseline Γραμμική Παλινδρόμηση / Baseline Linear Regression
- **Εργασία Β / Task B**: Πολυωνυμική Παλινδρόμηση με L1/L2 Κανονικοποίηση / Polynomial Regression with L1/L2 Regularization
- **Εργασία Γ / Task C**: Μείωση Διαστάσεων (PCA, CFS, Wrapper) / Dimensionality Reduction (PCA, CFS, Wrapper Methods)
- **Εργασία Δ / Task D**: Προβλέψεις Μελλοντικών Τιμών (Δεκέμβριος 2025, Ιανουάριος 2026) / Future Price Predictions (December 2025, January 2026)

**Σύμβολο Μετοχής / Stock Symbol**: NFLX (Netflix, Inc.)  
**Τομέας / Sector**: Communication Services  
**Πηγή Δεδομένων / Data Source**: Alpha Vantage API

---

## ΣΥΝΟΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ / COMPREHENSIVE RESULTS

### 📊 Μοντέλα που Αναλύθηκαν / Models Analyzed

**Συνολικά Μοντέλα / Total Models: 96**

- ✅ **16 Baseline** Linear Regression μοντέλα (4 smoothing × 4 lags)
- ✅ **32 Polynomial** Regression μοντέλα (Ridge + Lasso για κάθε ρύθμιση / for each config)
- ✅ **48 Dimensionality Reduction** μοντέλα (PCA + CFS + Sequential Forward Selection)

### 🏆 ΚΑΛΥΤΕΡΑ ΜΟΝΤΕΛΑ / BEST MODELS

#### 1. Καλύτερο Baseline / Best Baseline

- **Μοντέλο / Model**: Linear Regression
- **Προεπεξεργασία / Preprocessing**: Gaussian Smoothing (σ=3)
- **Παράθυρο Υστέρησης / Lag Window**: 12 μήνες / months
- **Χαρακτηριστικά / Features**: 24 (12 close price lags + 12 volume lags)
- **Training RMSE**: $0.02
- **Training R²**: 1.0000
- **Validation RMSE**: $0.03
- **Validation R²**: 1.0000

#### 2. Καλύτερο Polynomial / Best Polynomial

- **Μοντέλο / Model**: Ridge Regression (Degree 2)
- **Ρύθμιση / Configuration**: sigma3, 9 lags
- **Alpha**: 0.001
- **Validation RMSE**: $4.19
- **Validation R²**: 0.9978

#### 3. Καλύτερο Dimensionality Reduction / Best Dim-Reduction

- **Μέθοδος / Method**: Sequential Forward Selection
- **Ρύθμιση / Configuration**: sigma3, 12 lags
- **Χαρακτηριστικά / Features**: 12 (μειωμένα από 24 / reduced from 24)
- **Validation RMSE**: $0.03
- **Validation R²**: 1.0000

### 🔮 ΠΡΟΒΛΕΨΕΙΣ ΜΕΛΛΟΝΤΙΚΩΝ ΤΙΜΩΝ / FUTURE PREDICTIONS

**Δεκέμβριος 2025 / December 2025**: $1,175.48  
**Ιανουάριος 2026 / January 2026**: $1,175.95

_Βασισμένες στο καλύτερο baseline μοντέλο (sigma3, 12 lags)_  
_Based on best baseline model (sigma3, 12 lags)_

---

## ΔΟΜΗ ΕΡΓΟΥ / PROJECT STRUCTURE

```
stock-price-linear-regression/
│
├── step1_data_acquisition.py          # Συλλογή & προεπεξεργασία δεδομένων / Data fetching & preprocessing
├── step2_feature_engineering.py       # Δημιουργία χαρακτηριστικών με υστέρηση / Lagged feature creation
├── step3_baseline_linear_regression.py # Εργασία Α / Task A implementation
├── step4_polynomial_regression_regularization.py # Εργασία Β / Task B
├── step5_dimensionality_reduction.py  # Εργασία Γ / Task C implementation
├── step6_future_predictions_improved.py # Εργασία Δ & συνολική ανάλυση / Task D & comprehensive analysis
│
├── data/                              # Ακατέργαστα & επεξεργασμένα δεδομένα / Raw & processed data
│   ├── nflx_monthly_raw.csv           # 283 μήνες δεδομένων / months of data
│   ├── nflx_monthly_smoothed_sigma1.csv
│   ├── nflx_monthly_smoothed_sigma2.csv
│   ├── nflx_monthly_smoothed_sigma3.csv
│   └── smoothing_comparison.png
│
├── features/                          # Πίνακες χαρακτηριστικών & scalers / Feature matrices & scalers
│   ├── features_*.npz (16 ρυθμίσεις / configurations)
│   ├── scaler_*.pkl
│   ├── metadata_*.csv
│   └── train_val_split_*.png
│
├── models/                            # Εκπαιδευμένα μοντέλα / Trained models
│   ├── best_baseline_linear_regression.pkl
│   ├── all_polynomial_models.pkl      # 32 polynomial models
│   └── all_dimensionality_reduction_models.pkl # 48 dim-reduction models
│
├── results/                           # Απεικονίσεις & αναφορές / Visualizations & reports
│   ├── baseline_linear_regression_results.csv (16 models)
│   ├── polynomial_regression_all_models_results.csv (32 models)
│   ├── dimensionality_reduction_all_models_results.csv (48 models)
│   ├── baseline_predictions_dec_jan_2025_2026.csv
│   ├── COMPREHENSIVE_96_MODELS_REPORT_EL_EN.txt
│   ├── baseline_performance_by_config.png
│   ├── comprehensive_predictions_comparison.png
│   └── best_model_forecast_with_history.png
│
├── Provided Code/                     # Κώδικας παραδειγμάτων καθηγητή / Teacher's example code
│   ├── data_acquisition.ipynb
│   ├── regression_demo.ipynb
│   ├── feature_selection.ipynb
│   ├── pca_demo.ipynb
│   ├── training_L1_L2.ipynb
│   └── ...
│
├── .env                               # Διαμόρφωση API key / API key configuration
├── statistical_methods_of_ml.md       # Περιγραφή εργασίας / Assignment description
├── ML_TERMINOLOGY_GLOSSARY_EL_EN.md   # Γλωσσάριο όρων / Terminology glossary
└── README.md                          # Αυτό το αρχείο / This file
```

---

## ΕΓΚΑΤΑΣΤΑΣΗ & ΡΥΘΜΙΣΗ / INSTALLATION & SETUP

### Προαπαιτούμενα / Prerequisites

```bash
Python 3.8 ή νεότερο / or higher
```

### Απαιτούμενες Βιβλιοθήκες / Required Libraries

```bash
pip install numpy pandas scikit-learn scipy matplotlib requests python-dateutil
```

### Διαμόρφωση API Key / API Key Configuration

1. Εγγραφείτε για δωρεάν Alpha Vantage API key στο / Sign up for a free Alpha Vantage API key at:  
   https://www.alphavantage.co/

2. Δημιουργήστε αρχείο `.env` στη ρίζα του project / Create a `.env` file in the project root:

```
api_key=YOUR_API_KEY_HERE
```

---

## ΟΔΗΓΙΕΣ ΧΡΗΣΗΣ / USAGE INSTRUCTIONS

### Πλήρης Εκτέλεση Pipeline / Complete Pipeline Execution

Εκτελέστε όλα τα scripts με τη σειρά:  
Run all scripts in sequence:

```bash
# Βήμα 1: Συλλογή Δεδομένων / Step 1: Data Acquisition
python step1_data_acquisition.py

# Βήμα 2: Δημιουργία Χαρακτηριστικών / Step 2: Feature Engineering
python step2_feature_engineering.py

# Βήμα 3: Baseline Γραμμική Παλινδρόμηση / Step 3: Baseline Linear Regression (Εργασία Α / Task A)
python step3_baseline_linear_regression.py

# Βήμα 4: Πολυωνυμική Παλινδρόμηση / Step 4: Polynomial Regression (Εργασία Β / Task B)
python step4_polynomial_regression_regularization.py

# Βήμα 5: Μείωση Διαστάσεων / Step 5: Dimensionality Reduction (Εργασία Γ / Task C)
python step5_dimensionality_reduction.py

# Βήμα 6: Προβλέψεις Μελλοντικών Τιμών / Step 6: Future Predictions (Εργασία Δ / Task D)
python step6_future_predictions_improved.py
```

### Μεμονωμένη Εκτέλεση Βημάτων / Individual Step Execution

Κάθε script μπορεί να εκτελεστεί ανεξάρτητα αφού έχουν ολοκληρωθεί τα προηγούμενα βήματα.  
Each script can be run independently after previous steps are completed.

---

## ΛΕΠΤΟΜΕΡΕΙΕΣ ΜΕΘΟΔΟΛΟΓΙΑΣ / METHODOLOGY DETAILS

### 1. Συλλογή & Προεπεξεργασία Δεδομένων / Data Acquisition & Preprocessing

- **Πηγή / Source**: Alpha Vantage API (NFLX daily data)
- **Χρονική Περίοδος / Time Period**: Μάιος 2002 - Νοέμβριος 2025 / May 2002 - November 2025
- **Συνολικοί Μήνες / Total Months**: 283
- **Smoothing**: Gaussian filter (σ = 0, 1, 2, 3)
- **Μετρικές / Metrics**: Close price & Volume

### 2. Δημιουργία Χαρακτηριστικών / Feature Engineering

- **Χαρακτηριστικά με Υστέρηση / Lagged Features**: close_t-1, close_t-2, ..., close_t-N & volume_t-1, ..., volume_t-N
- **Παράθυρα Υστέρησης που Δοκιμάστηκαν / Lag Windows Tested**: 3, 6, 9, 12 μήνες / months
- **Διαίρεση Δεδομένων / Data Split**:
  - Εκπαίδευση / Training: < 2025 (260-269 δείγματα / samples)
  - Επικύρωση / Validation: 2025 (11 δείγματα / samples)
- **Κανονικοποίηση / Normalization**: StandardScaler (fitted on training data only)

### 3. Baseline Linear Regression (16 Μοντέλα / Models)

**Ρυθμίσεις που Δοκιμάστηκαν / Configurations Tested:**

- 4 smoothing levels (raw, sigma1, sigma2, sigma3)
- 4 lag windows (3, 6, 9, 12 months)
- **Σύνολο / Total**: 16 configurations

**Καλύτερη Ρύθμιση / Best Configuration:**

- sigma3, 12 lags → RMSE: $0.03, R²: 1.0000

### 4. Polynomial Regression (32 Μοντέλα / Models)

**Προσέγγιση / Approach:**

- Δοκιμάστηκαν ΟΛΑ τα 16 baseline configurations / Tested ALL 16 baseline configurations
- Πολυωνυμικά χαρακτηριστικά βαθμού 2 / Degree-2 polynomial features
- Ridge (L2) και Lasso (L1) regularization
- Grid search για alpha: [0.001, 0.01, 0.1, 1.0, 10.0]

**Καλύτερο Μοντέλο / Best Model:**

- sigma3, 9 lags, Ridge, α=0.001 → RMSE: $4.19, R²: 0.9978

### 5. Dimensionality Reduction (48 Μοντέλα / Models)

**Μέθοδοι / Methods:**

1. **PCA**: 95% explained variance threshold
2. **CFS**: Correlation-based Feature Selection
3. **Sequential Forward Selection**: Wrapper method (50% features target)

**Ανάλυση για ΟΛΑ τα 16 configurations / Applied to ALL 16 configurations**

**Καλύτερο Μοντέλο / Best Model:**

- sigma3, 12 lags, Forward Selection (12 features) → RMSE: $0.03, R²: 1.0000

### 6. Προβλέψεις Μελλοντικών Τιμών / Future Predictions

**Μεθοδολογία / Methodology:**

- Καταρρακτώδης πρόβλεψη / Cascading prediction
- Δεκέμβριος 2025: Χρήση ιστορικών δεδομένων / Using historical data
- Ιανουάριος 2026: Χρήση πρόβλεψης Δεκεμβρίου ως input / Using December prediction as input

---

## ΒΑΣΙΚΑ ΕΥΡΗΜΑΤΑ / KEY FINDINGS

### 1. Επίδραση Smoothing / Smoothing Impact

✅ **sigma3 (Gaussian σ=3) παράγει τα καλύτερα αποτελέσματα / produces best results**

- Μειώνει θόρυβο χωρίς απώλεια σημαντικών τάσεων / Reduces noise without losing important trends
- Validation RMSE: $0.03 vs $78.81 (raw data)

### 2. Παράθυρο Υστέρησης / Lag Window

✅ **12 μήνες είναι βέλτιστο / 12 months is optimal**

- Περισσότερα χαρακτηριστικά = καλύτερη πρόβλεψη / More features = better prediction
- Αποφυγή overfitting λόγω κανονικοποίησης / Avoiding overfitting through regularization

### 3. Σύγκριση Μοντέλων / Model Comparison

| Κατηγορία / Category | Καλύτερο RMSE / Best RMSE | Πλεονεκτήματα / Advantages                                               |
| -------------------- | ------------------------- | ------------------------------------------------------------------------ |
| **Baseline**         | $0.03                     | Απλό, ερμηνεύσιμο / Simple, interpretable                                |
| **Polynomial**       | $4.19                     | Συλλαμβάνει μη-γραμμικότητα / Captures non-linearity                     |
| **Dim-Reduction**    | $0.03                     | Λιγότερα χαρακτηριστικά, ίδια απόδοση / Fewer features, same performance |

### 4. Feature Selection

✅ **Sequential Forward Selection επιτυγχάνει άριστα αποτελέσματα / achieves excellent results**

- Μείωση από 24 → 12 χαρακτηριστικά / Reduction from 24 → 12 features
- Διατήρηση R²=1.0000 / Maintaining R²=1.0000
- Απλούστερο μοντέλο, ταχύτερη πρόβλεψη / Simpler model, faster prediction

---

## ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑ ΕΡΓΑΣΙΑ / RESULTS BY TASK

### ✅ Εργασία Α / Task A: Baseline Linear Regression

- **Εκπαιδευμένα Μοντέλα / Models Trained**: 16
- **Καλύτερη Ρύθμιση / Best Config**: sigma3, 12 lags
- **Validation RMSE**: $0.03
- **Validation R²**: 1.0000

### ✅ Εργασία Β / Task B: Polynomial Regression με Κανονικοποίηση / with Regularization

- **Εκπαιδευμένα Μοντέλα / Models Trained**: 32 (16 Ridge + 16 Lasso)
- **Καλύτερο Μοντέλο / Best Model**: Ridge (sigma3, 9 lags, α=0.001)
- **Validation RMSE**: $4.19
- **Validation R²**: 0.9978

### ✅ Εργασία Γ / Task C: Μείωση Διαστάσεων / Dimensionality Reduction

- **Εκπαιδευμένα Μοντέλα / Models Trained**: 48 (16 PCA + 16 CFS + 16 SFS)
- **Καλύτερη Μέθοδος / Best Method**: Sequential Forward Selection
- **Καλύτερη Ρύθμιση / Best Config**: sigma3, 12 lags (12 features)
- **Validation RMSE**: $0.03
- **Validation R²**: 1.0000

### ✅ Εργασία Δ / Task D: Προβλέψεις Μελλοντικών Τιμών / Future Predictions

- **Δεκέμβριος 2025 / December 2025**: $1,175.48
- **Ιανουάριος 2026 / January 2026**: $1,175.95
- **Μέθοδος / Method**: Καταρρακτώδης πρόβλεψη με καλύτερο baseline / Cascading prediction with best baseline

---

## ΑΡΧΕΙΑ ΑΝΑΦΟΡΩΝ / REPORT FILES

1. **COMPREHENSIVE_96_MODELS_REPORT_EL_EN.txt**

   - Δίγλωσση συνολική ανάλυση όλων των 96 μοντέλων
   - Bilingual comprehensive analysis of all 96 models

2. **baseline_linear_regression_results.csv**

   - Αναλυτικά αποτελέσματα 16 baseline μοντέλων
   - Detailed results for 16 baseline models

3. **polynomial_regression_all_models_results.csv**

   - Αναλυτικά αποτελέσματα 32 polynomial μοντέλων
   - Detailed results for 32 polynomial models

4. **dimensionality_reduction_all_models_results.csv**

   - Αναλυτικά αποτελέσματα 48 dim-reduction μοντέλων
   - Detailed results for 48 dim-reduction models

5. **baseline_predictions_dec_jan_2025_2026.csv**
   - Προβλέψεις για Δεκέμβριο 2025 & Ιανουάριο 2026
   - Predictions for December 2025 & January 2026

---

## ΑΠΕΙΚΟΝΙΣΕΙΣ / VISUALIZATIONS

### Δημιουργούμενα Γραφήματα / Generated Plots

1. **Data Smoothing Comparison** (`smoothing_comparison.png`)

   - Σύγκριση raw και smoothed data
   - Comparison of raw and smoothed data

2. **Baseline Performance** (`baseline_performance_by_config.png`)

   - Απόδοση όλων των 16 baseline configurations
   - Performance of all 16 baseline configurations

3. **Comprehensive Predictions** (`comprehensive_predictions_comparison.png`)

   - Σύγκριση προβλέψεων όλων των μοντέλων
   - Comparison of predictions across all models

4. **Best Model Forecast** (`best_model_forecast_with_history.png`)
   - Ιστορικά δεδομένα + προβλέψεις καλύτερου μοντέλου
   - Historical data + best model predictions

---

## ΤΕΧΝΙΚΕΣ ΛΕΠΤΟΜΕΡΕΙΕΣ / TECHNICAL DETAILS

### Χρησιμοποιούμενες Βιβλιοθήκες / Libraries Used

- **NumPy**: Αριθμητικοί υπολογισμοί / Numerical computations
- **Pandas**: Χειρισμός δεδομένων / Data manipulation
- **Scikit-learn**: Μοντέλα ML & μετρικές / ML models & metrics
- **SciPy**: Gaussian filtering
- **Matplotlib**: Απεικονίσεις / Visualizations
- **Requests**: API calls

### Αλγόριθμοι / Algorithms

1. **LinearRegression**: Baseline models
2. **Ridge**: L2 regularization (πολυωνυμικά / polynomial)
3. **Lasso**: L1 regularization (πολυωνυμικά / polynomial)
4. **PCA**: Unsupervised dimensionality reduction
5. **CFS**: Filter-based feature selection
6. **SequentialFeatureSelector**: Wrapper-based selection

---

## ΔΗΛΩΣΗ ΔΙΓΛΩΣΣΙΑΣ ΥΠΟΣΤΗΡΙΞΗΣ / BILINGUAL SUPPORT DECLARATION

### Ελληνική Υποστήριξη / Greek Language Support

Αυτό το έργο περιλαμβάνει πλήρη δίγλωσση υποστήριξη (Ελληνικά-Αγγλικά) σε όλα τα αρχεία:

This project includes full bilingual support (Greek-English) across all files:

✅ **Python Scripts**: Όλα τα modules περιέχουν docstrings στα Ελληνικά και Αγγλικά  
✅ **Python Scripts**: All modules contain docstrings in both Greek and English

✅ **Reports**: Όλες οι αναφορές δημιουργούνται σε δίγλωσση μορφή  
✅ **Reports**: All reports generated in bilingual format

✅ **Documentation**: README και τεχνικά έγγραφα σε αμφότερες τις γλώσσες  
✅ **Documentation**: README and technical documents in both languages

✅ **Terminology**: Γλωσσάριο ML όρων διαθέσιμο στο `ML_TERMINOLOGY_GLOSSARY_EL_EN.md`  
✅ **Terminology**: ML terminology glossary available in `ML_TERMINOLOGY_GLOSSARY_EL_EN.md`

---

## ΑΝΑΠΑΡΑΓΩΓΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ / REPRODUCIBILITY

Για αναπαραγωγή αποτελεσμάτων / To reproduce results:

1. Βεβαιωθείτε ότι έχετε Python 3.8+ / Ensure you have Python 3.8+
2. Εγκαταστήστε dependencies / Install dependencies
3. Ρυθμίστε Alpha Vantage API key στο `.env` / Configure Alpha Vantage API key in `.env`
4. Εκτελέστε όλα τα scripts με τη σειρά / Run all scripts in sequence
5. Ελέγξτε `results/` για αναφορές / Check `results/` for reports

**Σημείωση / Note**: Τα αποτελέσματα μπορεί να διαφέρουν ελαφρώς λόγω ενημερώσεων δεδομένων.  
Results may vary slightly due to data updates.

---

## ΣΥΓΓΡΑΦΕΑΣ & ΠΛΗΡΟΦΟΡΙΕΣ / AUTHOR & INFORMATION

**Μάθημα / Course**: Στατιστικές Μέθοδοι Μηχανικής Μάθησης / Statistical Methods of Machine Learning  
**Εργασία / Assignment**: Task 1 - Stock Price Prediction  
**Σύμβολο Μετοχής / Stock Symbol**: NFLX (Netflix, Inc.)  
**Χρονική Περίοδος / Time Period**: Μάιος 2002 - Νοέμβριος 2025 / May 2002 - November 2025  
**Συνολικά Μοντέλα / Total Models**: 96

---

## ΑΝΑΦΟΡΕΣ & ΠΗΓΕΣ / REFERENCES & SOURCES

1. **Alpha Vantage API**: https://www.alphavantage.co/
2. **Scikit-learn Documentation**: https://scikit-learn.org/
3. **Gaussian Filtering**: SciPy ndimage module
4. **ML Terminology Glossary**: `ML_TERMINOLOGY_GLOSSARY_EL_EN.md`

---

## ΑΔΕΙΑ / LICENSE

Αυτό το έργο δημιουργήθηκε για εκπαιδευτικούς σκοπούς.  
This project was created for educational purposes.

---

**Ημερομηνία Τελευταίας Ενημέρωσης / Last Updated**: Νοέμβριος 2025 / November 2025
