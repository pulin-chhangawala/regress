"""
regress.py - CLI regression analysis tool

Compare linear, polynomial, ridge, lasso, and elastic net regression
on any CSV dataset. Reports metrics, plots residuals, and helps you
pick the right model.

Usage:
    python regress.py --data data/housing.csv --target price
    python regress.py --data data/housing.csv --target price --features sqft bedrooms
    python regress.py --data data/housing.csv --target price --degree 3
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                   ElasticNet)
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score)
from sklearn.pipeline import Pipeline


def load_data(path, target, features=None):
    """Load CSV and split into features/target."""
    df = pd.read_csv(path)
    
    if target not in df.columns:
        print(f"Error: target '{target}' not in columns: {list(df.columns)}")
        sys.exit(1)
    
    # auto-select numeric columns if features not specified
    if features:
        X = df[features]
    else:
        X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    
    y = df[target]
    
    # drop rows with NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    return X, y


def build_models(degree=1, alpha=1.0):
    """Build a dictionary of regression pipelines."""
    models = {}
    
    # linear regression (baseline)
    models['Linear'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression()),
    ])
    
    # ridge (L2 regularization)
    models['Ridge'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=alpha)),
    ])
    
    # lasso (L1 regularization, performs feature selection)
    models['Lasso'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=alpha, max_iter=10000)),
    ])
    
    # elastic net (L1 + L2)
    models['ElasticNet'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)),
    ])
    
    # polynomial regression (if degree > 1)
    if degree > 1:
        models[f'Polynomial (d={degree})'] = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', LinearRegression()),
        ])
    
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, X_all, y_all):
    """Train and evaluate a single model."""
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # cross-validation for more robust estimate
    cv_scores = cross_val_score(model, X_all, y_all, cv=5, 
                                scoring='neg_mean_squared_error')
    
    return {
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'cv_rmse': np.sqrt(-cv_scores.mean()),
        'cv_rmse_std': np.sqrt(cv_scores.std()),
        'y_pred_test': y_pred_test,
    }


def plot_results(results, y_test, output_dir):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. model comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = list(results.keys())
    
    r2_vals = [results[n]['r2_test'] for n in names]
    axes[0].barh(names, r2_vals, color='steelblue')
    axes[0].set_xlabel('R² (test)')
    axes[0].set_title('Goodness of Fit')
    axes[0].set_xlim(min(0, min(r2_vals) - 0.1), 1.05)
    
    rmse_vals = [results[n]['rmse_test'] for n in names]
    axes[1].barh(names, rmse_vals, color='coral')
    axes[1].set_xlabel('RMSE (test)')
    axes[1].set_title('Prediction Error')
    
    cv_vals = [results[n]['cv_rmse'] for n in names]
    cv_stds = [results[n]['cv_rmse_std'] for n in names]
    axes[2].barh(names, cv_vals, xerr=cv_stds, color='mediumseagreen')
    axes[2].set_xlabel('CV RMSE')
    axes[2].set_title('Cross-Validation Error')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150)
    plt.close()
    
    # 2. residual plots for each model
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, res) in zip(axes, results.items()):
        residuals = y_test.values - res['y_pred_test']
        ax.scatter(res['y_pred_test'], residuals, alpha=0.5, s=10, color='steelblue')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residual')
        ax.set_title(name)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals.png'), dpi=150)
    plt.close()
    
    # 3. predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 6))
    best_name = max(results.keys(), key=lambda n: results[n]['r2_test'])
    y_pred = results[best_name]['y_pred_test']
    ax.scatter(y_test, y_pred, alpha=0.5, s=10, color='steelblue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', linewidth=1)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{best_name}: Predicted vs Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=150)
    plt.close()
    
    print(f"  Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Compare regression models')
    parser.add_argument('--data', required=True, help='Path to CSV file')
    parser.add_argument('--target', '-y', required=True, help='Target column name')
    parser.add_argument('--features', '-x', nargs='+', 
                        help='Feature columns (default: all numeric)')
    parser.add_argument('--degree', type=int, default=1,
                        help='Polynomial degree (default: 1 = linear only)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Regularization strength')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--output', '-o', default='results/',
                        help='Output directory for plots')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    args = parser.parse_args()
    
    # load data
    print("Loading data...")
    X, y = load_data(args.data, args.target, args.features)
    print(f"  {len(y)} samples, {X.shape[1]} features")
    print(f"  Target '{args.target}': mean={y.mean():.2f}, std={y.std():.2f}")
    print(f"  Features: {', '.join(X.columns)}")
    
    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42)
    
    # build and evaluate models
    models = build_models(degree=args.degree, alpha=args.alpha)
    results = {}
    
    print(f"\nComparing {len(models)} models:")
    print(f"{'Model':<24} {'R² (test)':>10} {'RMSE (test)':>12} {'MAE':>10} {'CV RMSE':>12}")
    print("-" * 72)
    
    for name, model in models.items():
        try:
            res = evaluate_model(model, X_train, X_test, y_train, y_test, X, y)
            results[name] = res
            
            # detect overfitting
            overfit = ""
            if res['r2_train'] - res['r2_test'] > 0.1:
                overfit = " ⚠ overfitting"
            
            print(f"{name:<24} {res['r2_test']:>10.4f} {res['rmse_test']:>12.4f} "
                  f"{res['mae_test']:>10.4f} {res['cv_rmse']:>12.4f}{overfit}")
        except Exception as e:
            print(f"{name:<24} {'FAILED':>10}: {e}")
    
    # recommend best model
    if results:
        best = max(results.keys(), key=lambda n: results[n]['r2_test'])
        print(f"\n  Best model: {best} (R²={results[best]['r2_test']:.4f})")
        
        if not args.no_plot:
            print("\nGenerating plots...")
            plot_results(results, y_test, args.output)
        
        # save summary
        summary = {name: {k: v for k, v in res.items() if k != 'y_pred_test'}
                    for name, res in results.items()}
        with open(os.path.join(args.output, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
