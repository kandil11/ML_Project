import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from utils import load_data, calculate_metrics, print_metrics, save_results

# 1. Load data
print("Loading data for Baseline...")
df = load_data()

# 2. Simple Naive Baseline
from sklearn.model_selection import train_test_split
drop_cols = [c for c in ['id', 'd', 'date', 'sales'] if c in df.columns]
X = df.drop(drop_cols, axis=1)
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Constant baseline: mean of train sales
baseline_pred = np.full_like(y_test, fill_value=y_train.mean())

# 3. Evaluation
metrics = calculate_metrics(y_test, baseline_pred)

# 4. Output metrics
print_metrics("Baseline (Mean)", metrics)

# Save metrics for comparison
save_results('Baseline', metrics, 'results_baseline.csv')
