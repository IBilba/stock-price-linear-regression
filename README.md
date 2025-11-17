# NFLX Stock Price Prediction with Linear Regression

**Statistical Methods of Machine Learning - Task 1**

## Project Overview

This project implements a comprehensive machine learning pipeline to predict Netflix (NFLX) stock prices using linear regression and various advanced techniques. The project addresses all four tasks from the assignment:

- **Task A**: Baseline Linear Regression
- **Task B**: Polynomial Regression with L1/L2 Regularization
- **Task C**: Dimensionality Reduction (PCA, CFS, Wrapper Methods)
- **Task D**: Future Price Predictions (December 2025)

**Stock Symbol**: NFLX (Netflix, Inc.)  
**Sector**: Communication Services  
**Data Source**: Alpha Vantage API

## Key Results Summary

### Best Model Configuration

- **Model**: Linear Regression
- **Preprocessing**: Gaussian Smoothing (σ=3)
- **Lag Window**: 12 months
- **Features**: 24 (12 close price lags + 12 volume lags)

### Performance Metrics

- **Training RMSE**: $0.02
- **Training R²**: 1.0000
- **Validation RMSE**: $0.03
- **Validation R²**: 1.0000

### Future Prediction

- **December 2025**: $1,175.48

## Project Structure

```
stock-price-linear-regression/
│
├── step1_data_acquisition.py          # Data fetching and preprocessing
├── step2_feature_engineering.py       # Lagged feature creation
├── step3_baseline_linear_regression.py # Task A implementation
├── step4_polynomial_regression_regularization.py # Task B
├── step5_dimensionality_reduction.py  # Task C implementation
├── step6_future_predictions.py        # Task D and final report
│
├── data/                              # Raw and processed data
│   ├── nflx_monthly_raw.csv
│   ├── nflx_monthly_smoothed_sigma1.csv
│   ├── nflx_monthly_smoothed_sigma2.csv
│   ├── nflx_monthly_smoothed_sigma3.csv
│   └── smoothing_comparison.png
│
├── features/                          # Feature matrices and scalers
│   ├── features_*.npz (16 configurations)
│   ├── scaler_*.pkl
│   ├── metadata_*.csv
│   └── train_val_split_*.png
│
├── models/                            # Trained models
│   ├── best_baseline_linear_regression.pkl
│   ├── best_ridge_polynomial.pkl
│   ├── best_lasso_polynomial.pkl
│   └── dimensionality_reduction_models.pkl
│
├── results/                           # Visualizations and reports
│   ├── baseline_performance_by_config.png
│   ├── baseline_r2_comparison.png
│   ├── baseline_actual_vs_predicted.png
│   ├── polynomial_regularization_paths.png
│   ├── dimensionality_reduction_comparison.png
│   ├── future_predictions_visualization.png
│   ├── baseline_linear_regression_results.csv
│   ├── polynomial_regression_comparison.csv
│   ├── dimensionality_reduction_results.csv
│   └── FINAL_PROJECT_SUMMARY.txt
│
├── Provided Code/                     # Teacher's example code
│   ├── data_acquisition.ipynb
│   ├── regression_demo.ipynb
│   ├── feature_selection.ipynb
│   ├── pca_demo.ipynb
│   ├── training_L1_L2.ipynb
│   └── ...
│
├── .env                               # API key configuration
├── statistical_methods_of_ml.md       # Assignment description
└── README.md                          # This file
```

## Installation & Setup

### Prerequisites

```bash
Python 3.8 or higher
```

### Required Libraries

```bash
pip install numpy pandas scikit-learn scipy matplotlib requests python-dateutil
```

### API Key Configuration

1. Sign up for a free Alpha Vantage API key at https://www.alphavantage.co/
2. Create a `.env` file in the project root:

```
api_key=YOUR_API_KEY_HERE
```

## Usage Instructions

### Complete Pipeline Execution

Run all scripts in sequence:

```bash
# Step 1: Data Acquisition (fetches from Alpha Vantage)
python step1_data_acquisition.py

# Step 2: Feature Engineering (creates 16 configurations)
python step2_feature_engineering.py

# Step 3: Baseline Linear Regression (Task A)
python step3_baseline_linear_regression.py

# Step 4: Polynomial Regression with Regularization (Task B)
python step4_polynomial_regression_regularization.py

# Step 5: Dimensionality Reduction (Task C)
python step5_dimensionality_reduction.py

# Step 6: Future Predictions & Final Report (Task D)
python step6_future_predictions.py
```

### Individual Task Execution

Each script can be run independently if previous steps have been completed:

```bash
# Run only Task B (requires features/ directory)
python step4_polynomial_regression_regularization.py

# Run only Task D (requires models/ directory)
python step6_future_predictions.py
```

## Methodology

### 1. Data Acquisition & Preprocessing

- **Data Source**: Alpha Vantage API (TIME_SERIES_DAILY)
- **Time Range**: May 2002 - November 2025 (283 months)
- **Aggregation**: Daily → Monthly averages
- **Smoothing**: Gaussian filter with σ ∈ {1, 2, 3}
- **Rationale**: Reduces noise while preserving trends

### 2. Feature Engineering

- **Lagged Features**:
  - `close_t-1` through `close_t-N`: Past closing prices
  - `volume_t-1` through `volume_t-N`: Past trading volumes
- **Lag Windows Tested**: N ∈ {3, 6, 9, 12} months
- **Scaling**: StandardScaler (z-score normalization)
- **Train/Val Split**:
  - Training: Pre-2025 (260-269 samples depending on N)
  - Validation: 2025 (11 samples)
  - **Critical**: Chronological split (no shuffling)

### 3. Model Training & Evaluation

#### Task A: Baseline Linear Regression

- **Configurations Tested**: 16 (4 smoothing × 4 lag windows)
- **Model**: Ordinary Least Squares (OLS) Linear Regression
- **Metrics**: RMSE, MAE, R²
- **Best**: sigma3, 12 lags → RMSE $0.03, R² 1.0000

#### Task B: Polynomial Regression

- **Degree**: 2 (24 features → 325 features)
- **Ridge (L2)**:
  - Best α: 0.1
  - Val RMSE: $8.98
  - All features retained
- **Lasso (L1)**:
  - Best α: 0.001
  - Val RMSE: $9.47
  - 263/325 features selected (19.1% sparsity)
- **Conclusion**: Baseline outperforms due to effective smoothing

#### Task C: Dimensionality Reduction

1. **PCA (95% variance)**:

   - 3 components
   - Val RMSE: $131.07, R²: -1.17
   - Poor performance (lost information)

2. **CFS (Correlation-based)**:

   - 1 feature (close_t-1)
   - Val RMSE: $21.91, R²: 0.9392
   - Surprisingly effective

3. **Sequential Forward Selection**:

   - 12 features (all close lags)
   - Val RMSE: $0.03, R²: 1.0000
   - Matched baseline performance

4. **Conclusion**: Close lags sufficient; volume adds minimal value

#### Task D: Future Predictions

- **Method**: Best baseline model
- **December 2025 Prediction**: $1,175.48
- **January 2026**: Cannot predict (requires December 2025 actual data)

## Key Findings

### 1. Preprocessing is Critical

Heavy Gaussian smoothing (σ=3) was the most important factor for success. It transformed noisy data into highly predictable patterns.

### 2. Linear Models Sufficient

With proper preprocessing, simple linear regression achieved near-perfect results. Complex polynomial features were unnecessary.

### 3. Optimal Lookback Window

12-month lag window captured both short-term momentum and long-term trends effectively.

### 4. Feature Importance

Close price lags far more informative than volume. Sequential Forward Selection confirmed this by selecting only close lags.

### 5. PCA Limitations

PCA failed on heavily smoothed data because:

- Smoothing already reduced dimensionality conceptually
- Linear transformation couldn't improve on smoothed features
- Critical temporal information was lost in transformation

## Visualizations Generated

### Data & Features

- `data/smoothing_comparison.png`: Effect of different σ values
- `features/train_val_split_*.png`: Temporal data split visualization

### Model Performance

- `baseline_performance_by_config.png`: All 16 configurations compared
- `baseline_r2_comparison.png`: R² scores across configurations
- `baseline_actual_vs_predicted.png`: Scatter plots for best model
- `polynomial_regularization_paths.png`: Alpha vs RMSE curves
- `dimensionality_reduction_comparison.png`: All reduction methods

### Predictions

- `future_predictions_visualization.png`: Historical + predicted prices

## Results Files

### CSV Tables

- `baseline_linear_regression_results.csv`: All configurations, sorted by validation RMSE
- `polynomial_regression_comparison.csv`: Ridge & Lasso results for all α values
- `dimensionality_reduction_results.csv`: PCA, CFS, Wrapper comparison

### Final Report

- `FINAL_PROJECT_SUMMARY.txt`: Comprehensive project summary with:
  - All task results
  - Model parameters
  - Key findings
  - Recommendations
  - Limitations

## Model Interpretation

### Baseline Linear Model Equation

```
price(t) = β₀ + Σ(βᵢ · close_t-i) + Σ(γⱼ · volume_t-j)
           i=1..12              j=1..12
```

### Top 5 Most Influential Features

1. `close_t-4`: -$10,201 per std dev (oscillatory pattern)
2. `close_t-3`: +$8,383 per std dev
3. `close_t-5`: +$7,460 per std dev
4. `close_t-2`: -$4,373 per std dev
5. `close_t-8`: +$3,253 per std dev

**Note**: Large coefficients due to high correlation in smoothed data. Coefficients show complex inter-dependencies rather than simple trends.

## Limitations & Considerations

### Model Limitations

1. **Heavy Smoothing Trade-off**: May delay reaction to sudden market changes
2. **Linear Assumption**: Assumes past patterns continue
3. **External Events**: Cannot capture earnings reports, market crashes, news
4. **Limited Validation**: Only 11 months of 2025 data

### Data Limitations

1. **API Rate Limits**: Free tier has 5 calls/min, 500/day
2. **Historical Bias**: Model trained on 2002-2024 may not generalize to different market regimes
3. **Monthly Aggregation**: Loses intra-month volatility information

### Prediction Limitations

1. **January 2026**: Requires December 2025 actual data (not yet available)
2. **No Confidence Intervals**: Point predictions without uncertainty quantification
3. **No Regime Detection**: Cannot identify when model becomes unreliable

## Recommendations

### For Production Use

1. **Update Frequency**: Retrain model monthly with new data
2. **Monitoring**: Track prediction errors to detect regime changes
3. **Ensemble**: Combine predictions from multiple σ values
4. **Confidence Intervals**: Add bootstrap or Bayesian methods
5. **Feature Expansion**: Include market indices (S&P 500), sector performance

### For Academic Extension

1. **Multi-Stock Analysis**: Test generalization across different stocks
2. **External Features**: Sentiment analysis, macroeconomic indicators
3. **Non-Linear Models**: LSTM, GRU for capturing complex patterns
4. **Online Learning**: Implement incremental updates
5. **Uncertainty Quantification**: Bayesian regression, quantile regression

## Technical Details

### Computational Complexity

- **Data Acquisition**: O(n) API calls + O(n) processing
- **Feature Engineering**: O(n × m) where n=samples, m=features
- **Linear Regression**: O(m² × n) for OLS solution
- **Polynomial (degree 2)**: O((m²)² × n) ≈ O(m⁴ × n)
- **Sequential Forward Selection**: O(m² × k) model trainings

### Memory Requirements

- **Raw Data**: ~5,000 daily records → ~1 MB
- **Feature Matrices**: 16 configurations × 2 sets → ~10 MB
- **Models**: < 1 MB total

### Runtime (Approximate)

- Step 1 (Data Acquisition): 30-60 seconds (API call)
- Step 2 (Feature Engineering): 5-10 seconds
- Step 3 (Baseline): 2-3 seconds
- Step 4 (Polynomial): 5-10 seconds
- Step 5 (Dimensionality): 30-60 seconds (Sequential Selection)
- Step 6 (Predictions): 1-2 seconds

**Total Runtime**: ~2-3 minutes

## References

### Assignment Requirements

- Course: Statistical Methods of Machine Learning
- Task: Predicting Stock Prices with Linear Regression
- Symbol: NFLX (Netflix, Inc.)
- API: Alpha Vantage (https://www.alphavantage.co/)

### Key Libraries

- **scikit-learn**: Machine learning models and metrics
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Gaussian filtering (scipy.ndimage.gaussian_filter1d)
- **matplotlib**: Visualization

### Methodology Inspiration

- Teacher's provided code (`Provided Code/` directory)
- Scikit-learn documentation
- Time series forecasting best practices

## Contact & Support

For questions or issues:

1. Check the `FINAL_PROJECT_SUMMARY.txt` for detailed results
2. Review inline code comments (extensive documentation)
3. Consult provided code examples in `Provided Code/`

## License

This project was created for academic purposes as part of a Machine Learning course assignment.

---

**Last Updated**: November 17, 2025  
**Project Status**: Complete ✓

All tasks (A, B, C, D) successfully implemented with extensive documentation.
