# regress

CLI tool for comparing regression models on any CSV dataset. Fits linear, ridge, lasso, elastic net, and polynomial regression, reports metrics side-by-side, and flags overfitting automatically.

Built this because every time I needed to do regression analysis I'd end up writing the same boilerplate: load CSV, split, fit 4 models, compare metrics, plot residuals. Now it's just one command.

## Quick Start

```bash
pip install -r requirements.txt

# generate sample data
python data/generate_data.py 2000 data/housing.csv

# run regression comparison
python src/regress.py --data data/housing.csv --target price

# try polynomial regression
python src/regress.py --data data/housing.csv --target price --degree 2

# select specific features
python src/regress.py --data data/housing.csv --target price --features sqft bedrooms age
```

## Example Output

```
Loading data...
  2000 samples, 7 features
  Target 'price': mean=327421.50, std=112933.21

Comparing 5 models:
Model                      R² (test)    RMSE (test)        MAE      CV RMSE
------------------------------------------------------------------------
Linear                       0.9103      33842.12    26801.42    34122.09
Ridge                        0.9103      33842.55    26801.84    34122.10
Lasso                        0.9092      34080.12    27004.55    34320.12
ElasticNet                   0.8915      37255.81    29804.80    37490.87
Polynomial (d=2)             0.9287      30183.66    23412.91    31050.22

  Best model: Polynomial (d=2) (R²=0.9287)
```

## What It Does

1. **Loads** any CSV with numeric features
2. **Splits** into train/test (80/20 by default)
3. **Fits** multiple regression models:
   - **Linear**: The baseline. No regularization
   - **Ridge (L2)**: Penalizes large coefficients. Helps with multicollinearity
   - **Lasso (L1)**: Can zero out coefficients → automatic feature selection
   - **Elastic Net**: Mix of L1 and L2
   - **Polynomial**: Captures non-linear relationships (specify `--degree`)
4. **Evaluates**: R², RMSE, MAE, 5-fold cross-validation
5. **Detects overfitting**: warns when train R² >> test R²
6. **Plots**: model comparison bars, residual plots, predicted vs actual

## Project Structure

```
src/
└── regress.py             # Main CLI tool
data/
└── generate_data.py       # Synthetic housing data generator
results/                   # Output plots and metrics (auto-generated)
```

## Requirements

- Python 3.8+
- NumPy, Pandas, Scikit-learn, Matplotlib
