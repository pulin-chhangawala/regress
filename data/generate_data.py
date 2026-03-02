#!/usr/bin/env python3
"""
generate_data.py - Generate a realistic housing dataset for testing

Creates a synthetic dataset with realistic feature correlations so the
regression models have something interesting to work with.

Usage:
    python generate_data.py [num_samples] [output_path]
"""

import csv
import random
import math
import sys

random.seed(42)


def generate_housing_data(n=1000):
    """
    Generate synthetic housing data with realistic correlations.
    
    price ≈ f(sqft, bedrooms, bathrooms, age, distance_downtown)
    with noise and non-linearities.
    """
    rows = []
    
    for _ in range(n):
        sqft = random.gauss(1800, 600)
        sqft = max(500, min(5000, sqft))
        
        bedrooms = max(1, min(6, round(sqft / 500 + random.gauss(0, 0.5))))
        bathrooms = max(1, min(4, round(bedrooms * 0.7 + random.gauss(0, 0.3))))
        
        age = max(0, random.gauss(25, 15))
        distance_downtown = max(0.5, random.gauss(10, 5))
        
        lot_size = sqft * random.uniform(1.5, 4.0) / 1000  # in 1000s sqft
        garage = 1 if sqft > 1500 and random.random() > 0.3 else 0
        
        # price model (semi-realistic)
        price = (
            50000 +                              # base
            150 * sqft +                         # main driver
            15000 * bedrooms +                   # bedroom premium
            20000 * bathrooms +                  # bathroom premium
            -1500 * age +                        # depreciation
            -5000 * distance_downtown +          # location premium
            25000 * garage +                     # garage premium
            8000 * lot_size +                    # lot premium
            0.01 * sqft * sqft / 100 +          # non-linearity (diminishing returns)
            random.gauss(0, 30000)               # noise
        )
        price = max(50000, round(price, -3))  # round to nearest 1000
        
        rows.append({
            'sqft': round(sqft),
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': round(age, 1),
            'distance_downtown': round(distance_downtown, 1),
            'lot_size': round(lot_size, 2),
            'garage': garage,
            'price': int(price),
        })
    
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    outpath = sys.argv[2] if len(sys.argv) > 2 else "data/housing.csv"
    
    import os
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    
    rows = generate_housing_data(n)
    
    with open(outpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    
    print(f"Generated {n} housing records → {outpath}")
    
    prices = [r['price'] for r in rows]
    print(f"  Price range: ${min(prices):,} to ${max(prices):,}")
    print(f"  Median: ${sorted(prices)[len(prices)//2]:,}")


if __name__ == '__main__':
    main()
