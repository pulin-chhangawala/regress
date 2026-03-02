"""
feature_engineer.py - Automated feature engineering for regression

Detects interaction terms, polynomial features, and log transforms
that improve model performance. Uses a greedy search approach:
try each transformation, keep it if it improves CV score.

This automates the tedious part of feature engineering that
people usually do by hand.

Usage:
    python feature_engineer.py --data data/housing.csv --target price
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import itertools


def detect_log_candidates(df, threshold=3.0):
    """
    Find features that might benefit from log transformation.
    
    Heuristic: if the feature has high skewness (> threshold),
    a log transform will make it more normal, which helps linear
    regression.
    """
    candidates = []
    for col in df.columns:
        if df[col].min() > 0:  # log requires positive values
            skew = df[col].skew()
            if abs(skew) > threshold:
                candidates.append((col, skew))
    return candidates


def detect_interactions(X, y, top_n=5):
    """
    Find the most useful pairwise interaction terms.
    
    Tests each pair of features: if X_i * X_j has higher
    correlation with y than either X_i or X_j alone,
    it's a useful interaction.
    """
    interactions = []
    cols = X.columns.tolist()
    
    for i, j in itertools.combinations(range(len(cols)), 2):
        c1, c2 = cols[i], cols[j]
        # Spearman correlation with target (more robust than Pearson)
        interaction = X[c1] * X[c2]
        corr_interaction = abs(interaction.corr(y))
        corr_c1 = abs(X[c1].corr(y))
        corr_c2 = abs(X[c2].corr(y))
        
        improvement = corr_interaction - max(corr_c1, corr_c2)
        if improvement > 0.01:  # meaningful improvement
            interactions.append({
                'features': (c1, c2),
                'interaction_corr': corr_interaction,
                'base_corr': max(corr_c1, corr_c2),
                'improvement': improvement,
            })
    
    interactions.sort(key=lambda x: x['improvement'], reverse=True)
    return interactions[:top_n]


def detect_polynomial_features(X, y, max_degree=3):
    """
    Test polynomial terms (X^2, X^3) for each feature.
    Keep the ones that improve correlation with target.
    """
    poly_features = []
    
    for col in X.columns:
        base_corr = abs(X[col].corr(y))
        
        for degree in range(2, max_degree + 1):
            poly = X[col] ** degree
            poly_corr = abs(poly.corr(y))
            
            if poly_corr > base_corr + 0.05:
                poly_features.append({
                    'feature': col,
                    'degree': degree,
                    'base_corr': base_corr,
                    'poly_corr': poly_corr,
                    'improvement': poly_corr - base_corr,
                })
    
    poly_features.sort(key=lambda x: x['improvement'], reverse=True)
    return poly_features


def auto_engineer(X, y, verbose=True):
    """
    Run all feature engineering detections and apply the best ones.
    Returns augmented feature matrix.
    """
    X_aug = X.copy()
    report = {'log_transforms': [], 'interactions': [], 'polynomials': []}
    
    # 1. Log transforms
    log_candidates = detect_log_candidates(X)
    for col, skew in log_candidates:
        new_col = f'log_{col}'
        X_aug[new_col] = np.log1p(X[col])
        report['log_transforms'].append({'feature': col, 'skewness': skew})
        if verbose:
            print(f"  + log({col}) [skew={skew:.2f}]")
    
    # 2. Interactions
    interactions = detect_interactions(X, y)
    for inter in interactions:
        c1, c2 = inter['features']
        new_col = f'{c1}_x_{c2}'
        X_aug[new_col] = X[c1] * X[c2]
        report['interactions'].append(inter)
        if verbose:
            print(f"  + {c1} × {c2} [Δcorr={inter['improvement']:.3f}]")
    
    # 3. Polynomial features
    polys = detect_polynomial_features(X, y)
    for p in polys[:5]:  # top 5
        new_col = f"{p['feature']}^{p['degree']}"
        X_aug[new_col] = X[p['feature']] ** p['degree']
        report['polynomials'].append(p)
        if verbose:
            print(f"  + {new_col} [Δcorr={p['improvement']:.3f}]")
    
    return X_aug, report


def evaluate_engineering(X_original, X_augmented, y, verbose=True):
    """Compare model performance before and after feature engineering."""
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0)),
    ])
    
    scores_before = cross_val_score(model, X_original, y, cv=5,
                                     scoring='neg_mean_squared_error')
    scores_after = cross_val_score(model, X_augmented, y, cv=5,
                                    scoring='neg_mean_squared_error')
    
    rmse_before = np.sqrt(-scores_before.mean())
    rmse_after = np.sqrt(-scores_after.mean())
    improvement = (rmse_before - rmse_after) / rmse_before * 100
    
    if verbose:
        print(f"\n  Before: RMSE = {rmse_before:.2f}")
        print(f"  After:  RMSE = {rmse_after:.2f}")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Features: {X_original.shape[1]} → {X_augmented.shape[1]}")
    
    return rmse_before, rmse_after, improvement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--target', required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.data)
    X = df.select_dtypes(include=[np.number]).drop(columns=[args.target], errors='ignore')
    y = df[args.target]
    
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    
    print(f"Feature Engineering for '{args.target}'")
    print(f"  {len(y)} samples, {X.shape[1]} original features\n")
    
    print("Detecting useful transformations:")
    X_aug, report = auto_engineer(X, y)
    
    print("\nEvaluating impact:")
    evaluate_engineering(X, X_aug, y)


if __name__ == '__main__':
    main()
