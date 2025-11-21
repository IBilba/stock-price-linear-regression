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

## ΣΥΝΟΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ / COMPREHENSIVE RESULTS

### 📊 Μοντέλα που Αναλύθηκαν / Models Analyzed

**Συνολικά Μοντέλα / Total Models: 96**

- ✅ **16 Baseline** Linear Regression μοντέλα (4 smoothing × 4 lags)
- ✅ **32 Polynomial** Regression μοντέλα (Ridge + Lasso για κάθε ρύθμιση / for each config)
- ✅ **48 Dimensionality Reduction** μοντέλα (PCA + CFS + Sequential Forward Selection)

### 🏆 ΚΑΛΥΤΕΡΑ ΜΟΝΤΕΛΑ / BEST MODELS

#### 1. Καλύτερο Baseline / Best Baseline

- **Μοντέλο / Model**: LR_sigma3_12lags (Linear Regression)
- **Προεπεξεργασία / Preprocessing**: Gaussian Smoothing (σ=3)
- **Παράθυρο Υστέρησης / Lag Window**: 12 μήνες / months
- **Χαρακτηριστικά / Features**: 24 (12 close price lags + 12 volume lags)
- **Training RMSE**: $0.06
- **Training R²**: 0.9999
- **Validation RMSE**: **$0.06**
- **Validation R²**: 1.0000
- **December 2025 Prediction**: $1,100.97
- **January 2026 Prediction**: $1,108.80

#### 2. Καλύτερο Polynomial / Best Polynomial

- **Μοντέλο / Model**: Ridge Regression (Degree 2)
- **Ρύθμιση / Configuration**: sigma3, 9 lags
- **Alpha**: 0.001
- **Validation RMSE**: $4.19
- **Validation R²**: 0.9978

#### 3. Καλύτερο Dimensionality Reduction / Best Dim-Reduction

- **Μοντέλο / Model**: LR_SFS_sigma3_12lags
- **Μέθοδος / Method**: Sequential Forward Selection (SFS)
- **Ρύθμιση / Configuration**: sigma3, 12 lags
- **Χαρακτηριστικά / Features**: 12 (μειωμένα από 24 / reduced from 24)
- **Validation RMSE**: **$0.06**
- **Validation R²**: 1.0000
- **December 2025 Prediction**: $1,100.98
- **January 2026 Prediction**: $1,108.88

### 🔮 ΠΡΟΒΛΕΨΕΙΣ ΜΕΛΛΟΝΤΙΚΩΝ ΤΙΜΩΝ / FUTURE PREDICTIONS

**Δεκέμβριος 2025 / December 2025**: **\$1,100.97**  
**Ιανουάριος 2026 / January 2026**: **\$1,108.80**

_Προβλέψεις από το καλύτερο μοντέλο: LR_sigma3_12lags (Baseline)_  
_Predictions from best model: LR_sigma3_12lags (Baseline)_

**Ensemble Στατιστικά / Ensemble Statistics** (96 μοντέλα / models):

- Δεκέμβριος / December: \$1,113.81 (±$2.84)
- Ιανουάριος / January: \$1,114.18 (±$4.21)
- 95% CI December: [\$1,108.25, $1,119.37]
- 95% CI January: [\$1,105.93, $1,122.43]

---

## ΔΟΜΗ ΕΡΓΟΥ / PROJECT STRUCTURE

```text
stock-price-linear-regression/
│
├── nflx_stock_prediction_complete_pipeline.ipynb  # 🆕 Ολοκληρωμένο Jupyter Notebook Pipeline
│                                                   # Complete Jupyter Notebook Pipeline (91 cells)
│
├── step1_data_acquisition.py          # Συλλογή & προεπεξεργασία δεδομένων / Data fetching & preprocessing
├── step2_feature_engineering.py       # Δημιουργία χαρακτηριστικών με υστέρηση / Lagged feature creation
├── step3_baseline_linear_regression.py # Εργασία Α / Task A implementation
├── step4_polynomial_regression_regularization.py # Εργασία Β / Task B
├── step5_dimensionality_reduction.py  # Εργασία Γ / Task C implementation
├── step6_future_predictions.py        # Εργασία Δ & συνολική ανάλυση / Task D & comprehensive analysis
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
│   ├── best_baseline_model.pkl        # Καλύτερο baseline μοντέλο / Best baseline model
│   ├── best_polynomial_model.pkl      # Καλύτερο polynomial μοντέλο / Best polynomial model
│   ├── best_dimred_model.pkl          # Καλύτερο dim-reduction μοντέλο / Best dimred model
│   ├── all_baseline_models.pkl        # Όλα τα 16 baseline μοντέλα / All 16 baseline models
│   ├── all_polynomial_models.pkl      # Όλα τα 32 polynomial μοντέλα / All 32 polynomial models
│   └── all_dimensionality_reduction_models.pkl # Όλα τα 48 dim-reduction μοντέλα / All 48 models
│
├── results/                           # Απεικονίσεις & αναφορές / Visualizations & reports
│   ├── all_96_models_results.csv      # Συνολικά αποτελέσματα 96 μοντέλων / Combined 96 models results
│   ├── baseline_linear_regression_results.csv (16 models)
│   ├── polynomial_regression_all_models_results.csv (32 models)
│   ├── dimensionality_reduction_all_models_results.csv (48 models)
│   ├── baseline_predictions_dec_jan_2025_2026.csv
│   ├── COMPLETE_96_MODELS_RANKING.csv
│   ├── BEST_MODELS_BY_APPROACH_COMPARISON.csv
│   ├── FINAL_PREDICTION_REPORT.txt
│   ├── baseline_performance_by_config.png
│   ├── future_predictions_visualization.png
│   └── best_model_forecast_with_history.png
│
├── .env                               # Διαμόρφωση API key / API key configuration
├── ML_TERMINOLOGY_GLOSSARY_EL_EN.md   # Γλωσσάριο όρων ML / ML terminology glossary
└── README.md                          # Αυτό το αρχείο / This file
```

---

## ΕΓΚΑΤΑΣΤΑΣΗ & ΡΥΘΜΙΣΗ / INSTALLATION & SETUP

### Προαπαιτούμενα / Prerequisites

```bash
Python 3.8 ή νεότερο / or higher
```

### Απαιτούμενες Βιβλιοθήκες / Required Libraries

**Για Python Scripts / For Python Scripts:**

```bash
pip install numpy pandas scikit-learn scipy matplotlib requests python-dateutil
```

**Για Jupyter Notebook (πρόσθετα) / For Jupyter Notebook (additional):**

```bash
# Εγκατάσταση Jupyter / Install Jupyter
pip install jupyter notebook

# Ή χρησιμοποιήστε VS Code με την επέκταση / Or use VS Code with extension:
# - Jupyter (Microsoft)
# - Python (Microsoft)
```

**Όλες οι βιβλιοθήκες μαζί / All libraries together:**

```bash
pip install numpy pandas scikit-learn scipy matplotlib requests python-dateutil jupyter notebook
```

### Διαμόρφωση API Key / API Key Configuration

1. Εγγραφείτε για δωρεάν Alpha Vantage API key στο / Sign up for a free Alpha Vantage API key at:  
   <https://www.alphavantage.co/>

2. Δημιουργήστε αρχείο `.env` στη ρίζα του project / Create a `.env` file in the project root:

```bash
api_key=YOUR_API_KEY_HERE
```

---

## 🚀 ΟΔΗΓΙΕΣ ΧΡΗΣΗΣ / USAGE INSTRUCTIONS

### Επιλογή 1: Jupyter Notebook (Συνιστάται για Εκμάθηση / Recommended for Learning)

**Πλεονεκτήματα / Advantages:**

- ✅ Διαδραστική εκτέλεση με άμεσα αποτελέσματα / Interactive execution with immediate results
- ✅ Ενσωματωμένες απεικονίσεις / Embedded visualizations (20+ plots)
- ✅ Δίγλωσσες επεξηγήσεις σε κάθε βήμα / Bilingual explanations at each step
- ✅ Εύκολη αλλαγή παραμέτρων και επανεκτέλεση / Easy parameter tuning and re-execution

**Εκτέλεση / Execution:**

```bash
# Άνοιγμα σε Jupyter / Open in Jupyter
jupyter notebook nflx_stock_prediction_complete_pipeline.ipynb

# Ή σε VS Code / Or in VS Code
code nflx_stock_prediction_complete_pipeline.ipynb
```

**Τρόπος Χρήσης / How to Use:**

1. **Run All Cells** (Jupyter: `Cell` → `Run All` | VS Code: `Run All` button)
2. Ή εκτελέστε βήμα-προς-βήμα για να κατανοήσετε κάθε φάση / Or run step-by-step to understand each phase
3. Δείτε απεικονίσεις και αποτελέσματα inline / View visualizations and results inline
4. Όλα τα μοντέλα και αποτελέσματα αποθηκεύονται αυτόματα / All models and results are saved automatically

**Τι Παράγει / What It Produces:**

- 96 εκπαιδευμένα μοντέλα στο [`models/`](models/) / 96 trained models in [`models/`](models/)
- Προβλέψεις για Δεκέμβριο 2025 & Ιανουάριο 2026 / Predictions for December 2025 & January 2026
- Αναλυτικές αναφορές στο [`results/`](results/) / Detailed reports in [`results/`](results/)
- 20+ απεικονίσεις ενσωματωμένες στο notebook / 20+ visualizations embedded in notebook

---

### Επιλογή 2: Python Scripts (Συνιστάται για Αυτοματισμό / Recommended for Automation)

**Πλεονεκτήματα / Advantages:**

- ✅ Κατάλληλο για παραγωγή και pipelines / Suitable for production and pipelines
- ✅ Εύκολη ενσωμάτωση σε CI/CD / Easy CI/CD integration
- ✅ Μεμονωμένη εκτέλεση βημάτων / Individual step execution
- ✅ Καλύτερο για batch processing / Better for batch processing

**Πλήρης Εκτέλεση Pipeline / Complete Pipeline Execution:**

```bash
# Βήμα 1: Συλλογή Δεδομένων / Step 1: Data Acquisition
python step1_data_acquisition.py
# Αποτέλεσμα / Output: 4 CSV files στο data/ (raw + 3 smoothed versions)

# Βήμα 2: Δημιουργία Χαρακτηριστικών / Step 2: Feature Engineering
python step2_feature_engineering.py
# Αποτέλεσμα / Output: 16 .npz files στο features/ (4 smoothing × 4 lags)

# Βήμα 3: Baseline Linear Regression (Εργασία Α / Task A)
python step3_baseline_linear_regression.py
# Αποτέλεσμα / Output: 16 μοντέλα → Καλύτερο: LR_sigma3_12lags (RMSE $0.06)

# Βήμα 4: Polynomial Regression (Εργασία Β / Task B)
python step4_polynomial_regression_regularization.py
# Αποτέλεσμα / Output: 32 μοντέλα (16 Ridge + 16 Lasso) → Καλύτερο: Ridge (RMSE $4.19)

# Βήμα 5: Dimensionality Reduction (Εργασία Γ / Task C)
python step5_dimensionality_reduction.py
# Αποτέλεσμα / Output: 48 μοντέλα (16 PCA + 16 CFS + 16 SFS) → Καλύτερο: LR_SFS_sigma3_12lags (RMSE $0.06)

# Βήμα 6: Προβλέψεις Μελλοντικών Τιμών (Εργασία Δ / Task D)
python step6_future_predictions.py
# Αποτέλεσμα / Output: Προβλέψεις Δεκέμβριος 2025: $1,100.97 | Ιανουάριος 2026: $1,108.80
```

**Μεμονωμένη Εκτέλεση / Individual Execution:**

Κάθε script μπορεί να εκτελεστεί ανεξάρτητα αφού τα προηγούμενα βήματα έχουν ολοκληρωθεί.  
Each script can be run independently after previous steps are completed.

**Τελικά Αποτελέσματα / Final Results:**

- 📊 **96 Μοντέλα / 96 Models**: Αποθηκευμένα σε [`models/`](models/) / Saved in [`models/`](models/)
- 📈 **Προβλέψεις / Predictions**: December 2025: **\$1,100.97** | January 2026: **\$1,108.80**
- 📄 **Αναφορές / Reports**:
  - [COMPLETE_96_MODELS_RANKING.csv](results/COMPLETE_96_MODELS_RANKING.csv)
  - [FINAL_PREDICTION_REPORT.txt](results/FINAL_PREDICTION_REPORT.txt)
  - [BEST_MODELS_BY_APPROACH_COMPARISON.csv](results/BEST_MODELS_BY_APPROACH_COMPARISON.csv)

---

### Σύγκριση Προσεγγίσεων / Approach Comparison

| Χαρακτηριστικό / Feature                     | 📓 Jupyter Notebook          | 🐍 Python Scripts                   |
| -------------------------------------------- | ---------------------------- | ----------------------------------- |
| **Εκτέλεση / Execution**                     | Διαδραστική / Interactive    | Σειριακή / Sequential               |
| **Απεικονίσεις / Visualizations**            | Ενσωματωμένες / Embedded     | Αποθηκευμένες PNG / Saved PNGs      |
| **Τεκμηρίωση / Documentation**               | Markdown cells + comments    | Docstrings                          |
| **Debugging**                                | Cell-by-cell inspection      | Print statements + logs             |
| **Παρουσίαση / Presentation**                | Self-contained notebook      | Αρχεία αποτελεσμάτων / Result files |
| **Χρόνος Εκτέλεσης / Execution Time**        | ~10-15 λεπτά / minutes       | ~10-15 λεπτά / minutes              |
| **Μοντέλα που Αποθηκεύονται / Models Saved** | ✅ Όλα / All 96              | ✅ Όλα / All 96                     |
| **Καλύτερο Για / Best For**                  | Exploration, teaching, demos | Production, automation, CI/CD       |

---

## 📓 JUPYTER NOTEBOOK PIPELINE / ΟΛΟΚΛΗΡΩΜΕΝΟ NOTEBOOK

### Επισκόπηση / Overview

Το **[`nflx_stock_prediction_complete_pipeline.ipynb`](nflx_stock_prediction_complete_pipeline.ipynb)** είναι ένα ολοκληρωμένο Jupyter Notebook που ενοποιεί όλα τα 6 βήματα της ανάλυσης σε μία διαδραστική διεπαφή.

The **[`nflx_stock_prediction_complete_pipeline.ipynb`](nflx_stock_prediction_complete_pipeline.ipynb)** is a comprehensive Jupyter Notebook that unifies all 6 steps of the analysis in an interactive interface.

### Δομή Notebook / Notebook Structure

**Σύνολο Κελιών / Total Cells**: 91 (44 code cells + 47 markdown cells)

#### Βήμα 1: Εισαγωγή & Ρύθμιση / Step 1: Introduction & Setup

- Εισαγωγή βιβλιοθηκών / Library imports
- API key configuration
- Επισκόπηση έργου / Project overview

#### Βήμα 2: Συλλογή Δεδομένων / Step 2: Data Acquisition

- **Εξόδος / Output**: 283 μήνες δεδομένων NFLX / 283 months of NFLX data
- Gaussian smoothing με σ = 1, 2, 3 / Gaussian smoothing with σ = 1, 2, 3
- **Απεικόνιση / Visualization**: 4-panel σύγκριση smoothing επιπέδων / 4-panel smoothing levels comparison

#### Βήμα 3: Δημιουργία Χαρακτηριστικών / Step 3: Feature Engineering

- **Ρυθμίσεις / Configurations**: 16 combinations (4 smoothing × 4 lags)
- Χρονολογική διαίρεση: Training (pre-2025) / Validation (2025)
- StandardScaler normalization
- **Απεικόνιση / Visualization**: Train/Val split timeline για κάθε ρύθμιση / for each configuration

#### Βήμα 4: Baseline Linear Regression (Εργασία Α / Task A)

- **Εκπαιδευμένα Μοντέλα / Trained Models**: 16
- **Καλύτερη Απόδοση / Best Performance**:
  - Model: `LR_sigma3_12lags`
  - Validation RMSE: **$16.65**
  - Validation R²: **0.9564**
  - Training RMSE: $22.82
- **Απεικόνιση / Visualization**:
  - 6-panel performance comparison
  - Actual vs Predicted για καλύτερο μοντέλο / for best model
- **Αποθήκευση / Saved**:
  - [`all_baseline_models.pkl`](models/all_baseline_models.pkl) (16 models)
  - [`best_baseline_model.pkl`](models/best_baseline_model.pkl)

#### Βήμα 5: Polynomial Regression (Εργασία Β / Task B)

- **Εκπαιδευμένα Μοντέλα / Trained Models**: 32 (16 Ridge + 16 Lasso)
- **Polynomial Degree**: 2
- **Alpha Grid Search**: [0.001, 0.01, 0.1, 1.0, 10.0]
- **Καλύτερη Απόδοση / Best Performance**:
  - Model: `Lasso_sigma2_9lags` (L1 regularization)
  - Validation RMSE: **$16.65**
  - Validation R²: **0.9504**
  - Alpha: 0.01
- **Απεικόνιση / Visualization**:
  - Ridge vs Lasso comparison (6 panels)
  - Regularization path για άλλα μοντέλα / for top models
- **Αποθήκευση / Saved**:
  - [`all_polynomial_models.pkl`](models/all_polynomial_models.pkl) (32 models)
  - [`best_polynomial_model.pkl`](models/best_polynomial_model.pkl)

#### Βήμα 6: Dimensionality Reduction (Εργασία Γ / Task C)

- **Εκπαιδευμένα Μοντέλα / Trained Models**: 48 (16 PCA + 16 CFS + 16 SFS)
- **Μέθοδοι / Methods**:
  1. **PCA**: 95% explained variance
  2. **CFS**: Correlation-based Feature Selection
  3. **Sequential Forward Selection**: 50% features target
- **Καλύτερη Απόδοση / Best Performance**:
  - Model: `LR_SFS_sigma1_9lags`
  - Validation RMSE: **$16.68**
  - Validation R²: **0.9564**
  - Features: 9 (reduced from 18)
- **Απεικόνιση / Visualization**:
  - 4-panel comparison (PCA vs CFS vs SFS)
  - Feature reduction analysis
- **Αποθήκευση / Saved**:
  - [`all_dimensionality_reduction_models.pkl`](models/all_dimensionality_reduction_models.pkl) (48 models)
  - [`best_dimred_model.pkl`](models/best_dimred_model.pkl)

#### Βήμα 7: Προβλέψεις Μελλοντικών Τιμών / Step 7: Future Predictions (Εργασία Δ / Task D)

- **Χρησιμοποιούμενα Μοντέλα / Models Used**: Top 3 από κάθε προσέγγιση = 9 μοντέλα / from each approach = 9 models
- **Μέθοδος / Method**: Καταρρακτώδης πρόβλεψη / Cascading prediction

  - Δεκέμβριος 2025: Χρήση ιστορικών δεδομένων / Using historical data
  - Ιανουάριος 2026: Χρήση πρόβλεψης Δεκεμβρίου / Using December prediction

- **Αποτελέσματα Καλύτερου Μοντέλου / Best Model Results**:

  - **Μοντέλο / Model**: `LR_sigma3_12lags` (Baseline)
  - **Δεκέμβριος 2025 / December 2025**: **$1,110.04**
  - **Ιανουάριος 2026 / January 2026**: **$1,108.80**
  - Εκτιμώμενη μεταβολή / Estimated change: -0.11% (Dec → Jan)

- **Ensemble Statistics / Στατιστικά Ensemble**:

  - Δεκέμβριος 2025: $1,113.81 (±$2.84)
  - Ιανουάριος 2026: $1,114.18 (±$4.21)
  - 95% CI December: [$1,108.25, $1,119.37]
  - 95% CI January: [$1,105.93, $1,122.43]

- **Απεικόνιση / Visualization**:

  - 4-panel comprehensive plot:
    - Historical trend + predictions με confidence intervals
    - December 2025 distribution με best model line
    - January 2026 distribution με best model line
    - Average predictions by approach (Baseline/Polynomial/DimRed)
  - Best model highlighted σε όλα τα plots / in all plots

- **Αποθήκευση / Saved**:
  - [`baseline_predictions_dec_jan_2025_2026.csv`](results/baseline_predictions_dec_jan_2025_2026.csv)
  - [`FINAL_PREDICTION_REPORT.txt`](results/FINAL_PREDICTION_REPORT.txt)
  - [`future_predictions_visualization.png`](results/future_predictions_visualization.png)

#### Βήμα 8: Συνολική Σύνοψη / Step 8: Overall Summary

- **Πλήρης Κατάταξη / Complete Ranking**: 96 μοντέλα ταξινομημένα κατά RMSE / 96 models sorted by RMSE
- **Σύγκριση Προσεγγίσεων / Approach Comparison**: Baseline vs Polynomial vs DimRed
- **Ανάλυση Προβλέψεων Μέλλοντος / Future Predictions Analysis**:
  - Στατιστική ανάλυση / Statistical analysis
  - Model agreement (CV < 1% = εξαιρετική συμφωνία / excellent agreement)
  - Προβλέψεις ανά προσέγγιση / Predictions by approach
- **Βασικά Συμπεράσματα / Key Conclusions**:
  - Επίδραση smoothing και lags / Effect of smoothing and lags
  - Ridge vs Lasso comparison
  - Feature reduction insights (PCA/CFS/SFS)
  - Dimensionality reduction effectiveness
- **Συστάσεις για Production / Production Recommendations**
- **Τελική Αναφορά / Final Report**: Comprehensive bilingual summary

### Βασικά Χαρακτηριστικά Notebook / Key Notebook Features

✅ **Διαδραστική Εκτέλεση / Interactive Execution**: Εκτέλεση κελιών βήμα-βήμα / Step-by-step cell execution  
✅ **Απεικονίσεις σε Real-Time / Real-Time Visualizations**: 20+ plots ενσωματωμένα / embedded plots  
✅ **Δίγλωσση Τεκμηρίωση / Bilingual Documentation**: Ελληνικά & Αγγλικά σε κάθε κελί / Greek & English in every cell  
✅ **Πλήρης Αναπαραγωγιμότητα / Full Reproducibility**: Όλα τα μοντέλα αποθηκεύονται / All models saved  
✅ **Comprehensive Outputs**: Λεπτομερή στατιστικά και μετρικές / Detailed statistics and metrics

---

## ΛΕΠΤΟΜΕΡΕΙΕΣ ΜΕΘΟΔΟΛΟΓΙΑΣ / METHODOLOGY DETAILS

### 1. Συλλογή & Προεπεξεργασία Δεδομένων / Data Acquisition & Preprocessing

- **Πηγή / Source**: Alpha Vantage API (NFLX daily data)
- **Χρονική Περίοδος / Time Period**: Μάιος 2002 - Νοέμβριος 2025 / May 2002 - November 2025
- **Συνολικοί Μήνες / Total Months**: 283
- **Smoothing**: Gaussian filter (σ = 0, 1, 2, 3)
- **Μετρικές / Metrics**: Close price & Volume
- **Script**: [`step1_data_acquisition.py`](step1_data_acquisition.py)

### 2. Δημιουργία Χαρακτηριστικών / Feature Engineering

- **Χαρακτηριστικά με Υστέρηση / Lagged Features**: close_t-1, close_t-2, ..., close_t-N & volume_t-1, ..., volume_t-N
- **Παράθυρα Υστέρησης που Δοκιμάστηκαν / Lag Windows Tested**: 3, 6, 9, 12 μήνες / months
- **Διαίρεση Δεδομένων / Data Split**:
  - Εκπαίδευση / Training: < 2025 (260-269 δείγματα / samples)
  - Επικύρωση / Validation: 2025 (11 δείγματα / samples)
- **Κανονικοποίηση / Normalization**: StandardScaler (fitted on training data only)
- **Script**: [`step2_feature_engineering.py`](step2_feature_engineering.py)

### 3. Baseline Linear Regression (16 Μοντέλα / Models)

**Ρυθμίσεις που Δοκιμάστηκαν / Configurations Tested:**

- 4 smoothing levels (raw, sigma1, sigma2, sigma3)
- 4 lag windows (3, 6, 9, 12 months)
- **Σύνολο / Total**: 16 configurations
- **Script**: [`step3_baseline_linear_regression.py`](step3_baseline_linear_regression.py)

**Καλύτερη Ρύθμιση / Best Configuration:**

- Model: LR_sigma3_12lags
- Config: sigma3, 12 lags
- Validation RMSE: **$0.06**
- Validation R²: 1.0000
- December 2025: $1,100.97
- January 2026: $1,108.80

### 4. Polynomial Regression (32 Μοντέλα / Models)

**Προσέγγιση / Approach:**

- Δοκιμάστηκαν ΟΛΑ τα 16 baseline configurations / Tested ALL 16 baseline configurations
- Πολυωνυμικά χαρακτηριστικά βαθμού 2 / Degree-2 polynomial features
- Ridge (L2) και Lasso (L1) regularization
- Grid search για alpha: [0.001, 0.01, 0.1, 1.0, 10.0]
- **Script**: [`step4_polynomial_regression_regularization.py`](step4_polynomial_regression_regularization.py)

**Καλύτερο Μοντέλο / Best Model:**

- sigma3, 9 lags, Ridge, α=0.001 → RMSE: $4.19, R²: 0.9978

### 5. Dimensionality Reduction (48 Μοντέλα / Models)

**Μέθοδοι / Methods:**

1. **PCA**: 95% explained variance threshold
2. **CFS**: Correlation-based Feature Selection
3. **Sequential Forward Selection**: Wrapper method (50% features target)

**Ανάλυση για ΟΛΑ τα 16 configurations / Applied to ALL 16 configurations**

- **Script**: [`step5_dimensionality_reduction.py`](step5_dimensionality_reduction.py)

**Καλύτερο Μοντέλο / Best Model:**

- Model: LR_SFS_sigma3_12lags
- Config: sigma3, 12 lags
- Method: Sequential Forward Selection (12 features)
- Validation RMSE: **$0.06**
- Validation R²: 1.0000

### 6. Προβλέψεις Μελλοντικών Τιμών / Future Predictions

**Μεθοδολογία / Methodology:**

- Καταρρακτώδης πρόβλεψη / Cascading prediction
- Δεκέμβριος 2025: Χρήση ιστορικών δεδομένων / Using historical data
- Ιανουάριος 2026: Χρήση πρόβλεψης Δεκεμβρίου ως input / Using December prediction as input
- **Script**: [`step6_future_predictions.py`](step6_future_predictions.py)

---

## ΒΑΣΙΚΑ ΕΥΡΗΜΑΤΑ / KEY FINDINGS

### 1. Επίδραση Smoothing / Smoothing Impact

✅ **sigma3 (Gaussian σ=3) παράγει τα καλύτερα αποτελέσματα / produces best results**

- Μειώνει θόρυβο χωρίς απώλεια σημαντικών τάσεων / Reduces noise without losing important trends
- Best Validation RMSE: **$0.06** (sigma3, 12 lags)
- Significant improvement over raw data configurations

### 2. Παράθυρο Υστέρησης / Lag Window

✅ **12 μήνες είναι βέλτιστο / 12 months is optimal**

- Περισσότερα χαρακτηριστικά = καλύτερη πρόβλεψη / More features = better prediction
- Αποφυγή overfitting λόγω κανονικοποίησης / Avoiding overfitting through regularization

### 3. Σύγκριση Μοντέλων / Model Comparison

| Κατηγορία / Category | Καλύτερο RMSE / Best RMSE | Καλύτερο Μοντέλο / Best Model | Πλεονεκτήματα / Advantages                                               |
| -------------------- | ------------------------- | ----------------------------- | ------------------------------------------------------------------------ |
| **Baseline**         | **$0.06**                 | LR_sigma3_12lags              | Απλό, ερμηνεύσιμο / Simple, interpretable                                |
| **Polynomial**       | $4.19                     | Ridge_sigma3_9lags            | Συλλαμβάνει μη-γραμμικότητα / Captures non-linearity                     |
| **Dim-Reduction**    | **$0.06**                 | LR_SFS_sigma3_12lags          | Λιγότερα χαρακτηριστικά, ίδια απόδοση / Fewer features, same performance |

### 4. Feature Selection

✅ **Sequential Forward Selection επιτυγχάνει άριστα αποτελέσματα / achieves excellent results**

- Μείωση από 24 → 12 χαρακτηριστικά / Reduction from 24 → 12 features
- Validation RMSE: $0.06 (σχεδόν ίδιο με baseline / nearly identical to baseline)
- Διατήρηση R²=1.0000 / Maintaining R²=1.0000
- 50% λιγότερα features, ίδια απόδοση / 50% fewer features, same performance
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

1. **[all_96_models_results.csv](results/all_96_models_results.csv)**

   - Συνολικά αποτελέσματα όλων των 96 μοντέλων
   - Combined results of all 96 models

2. **[COMPLETE_96_MODELS_RANKING.csv](results/COMPLETE_96_MODELS_RANKING.csv)**

   - Πλήρης κατάταξη μοντέλων κατά RMSE
   - Complete model ranking by RMSE

3. **[BEST_MODELS_BY_APPROACH_COMPARISON.csv](results/BEST_MODELS_BY_APPROACH_COMPARISON.csv)**

   - Σύγκριση καλύτερων μοντέλων ανά προσέγγιση
   - Comparison of best models by approach

4. **[baseline_linear_regression_results.csv](results/baseline_linear_regression_results.csv)**

   - Αναλυτικά αποτελέσματα 16 baseline μοντέλων
   - Detailed results for 16 baseline models

5. **[polynomial_regression_all_models_results.csv](results/polynomial_regression_all_models_results.csv)**

   - Αναλυτικά αποτελέσματα 32 polynomial μοντέλων
   - Detailed results for 32 polynomial models

6. **[dimensionality_reduction_all_models_results.csv](results/dimensionality_reduction_all_models_results.csv)**

   - Αναλυτικά αποτελέσματα 48 dim-reduction μοντέλων
   - Detailed results for 48 dim-reduction models

7. **[baseline_predictions_dec_jan_2025_2026.csv](results/baseline_predictions_dec_jan_2025_2026.csv)**

   - Προβλέψεις για Δεκέμβριο 2025 & Ιανουάριο 2026
   - Predictions for December 2025 & January 2026

8. **[FINAL_PREDICTION_REPORT.txt](results/FINAL_PREDICTION_REPORT.txt)**
   - Δίγλωσση τελική αναφορά με όλα τα αποτελέσματα
   - Bilingual final report with all results

---

## ΑΠΕΙΚΟΝΙΣΕΙΣ / VISUALIZATIONS

### Δημιουργούμενα Γραφήματα / Generated Plots

1. **Data Smoothing Comparison** ([`smoothing_comparison.png`](data/smoothing_comparison.png))

   - Σύγκριση raw και smoothed data
   - Comparison of raw and smoothed data

2. **Baseline Performance** ([`baseline_performance_by_config.png`](results/baseline_performance_by_config.png))

   - Απόδοση όλων των 16 baseline configurations
   - Performance of all 16 baseline configurations

3. **Comprehensive Predictions** ([`comprehensive_predictions_comparison.png`](results/comprehensive_predictions_comparison.png))

   - Σύγκριση προβλέψεων όλων των μοντέλων
   - Comparison of predictions across all models

4. **Best Model Forecast** ([`best_model_forecast_with_history.png`](results/best_model_forecast_with_history.png))
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

✅ **Terminology**: Γλωσσάριο ML όρων διαθέσιμο στο [`ML_TERMINOLOGY_GLOSSARY_EL_EN.md`](ML_TERMINOLOGY_GLOSSARY_EL_EN.md)  
✅ **Terminology**: ML terminology glossary available in [`ML_TERMINOLOGY_GLOSSARY_EL_EN.md`](ML_TERMINOLOGY_GLOSSARY_EL_EN.md)

---

## ΑΝΑΠΑΡΑΓΩΓΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ / REPRODUCIBILITY

Για αναπαραγωγή αποτελεσμάτων / To reproduce results:

1. Βεβαιωθείτε ότι έχετε Python 3.8+ / Ensure you have Python 3.8+
2. Εγκαταστήστε dependencies / Install dependencies
3. Ρυθμίστε Alpha Vantage API key στο `.env` / Configure Alpha Vantage API key in `.env`
4. Εκτελέστε όλα τα scripts με τη σειρά / Run all scripts in sequence
5. Ελέγξτε [`results/`](results/) για αναφορές / Check [`results/`](results/) for reports

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

1. **Alpha Vantage API**: <https://www.alphavantage.co/>
2. **Scikit-learn Documentation**: <https://scikit-learn.org/>
3. **Gaussian Filtering**: SciPy ndimage module
4. **ML Terminology Glossary**: [ML_TERMINOLOGY_GLOSSARY_EL_EN.md](ML_TERMINOLOGY_GLOSSARY_EL_EN.md)

---

## ΑΔΕΙΑ / LICENSE

Αυτό το έργο δημιουργήθηκε για εκπαιδευτικούς σκοπούς από τον Βασίλειο Μπίτζα.  
This project was created for educational purposes from Vasileios Bitzas.

---

**Ημερομηνία Τελευταίας Ενημέρωσης / Last Updated**: Νοέμβριος 2025 / November 2025
