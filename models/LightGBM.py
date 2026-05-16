import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from utils import load_data, calculate_metrics, print_metrics, save_results

# 1. Load data
print("Loading data for LightGBM...")
df = load_data()

# 2. Prepare features and target
drop_cols = [c for c in ['id', 'd', 'date', 'sales'] if c in df.columns]
X = df.drop(drop_cols, axis=1)
y = df['sales']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build LightGBM model
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbosity=-1
)

print("Training LightGBM...")
lgb_model.fit(X_train, y_train)

# 5. Predictions and Evaluation
preds = lgb_model.predict(X_test)
metrics = calculate_metrics(y_test, preds)

# 6. Output metrics
print_metrics("LightGBM", metrics)

# Save metrics for comparison
save_results('LightGBM', metrics, 'results_lgb.csv')
