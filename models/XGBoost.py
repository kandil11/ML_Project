import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from utils import load_data, calculate_metrics, print_metrics, save_results

# 1. Load data using shared utility
print("Loading data for XGBoost...")
df = load_data()

# 2. Prepare features and target
drop_cols = [c for c in ['id', 'd', 'date', 'sales'] if c in df.columns]
X = df.drop(drop_cols, axis=1)
y = df['sales']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build XGBoost model
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    tree_method='hist',
    random_state=42
)

print("Training XGBoost...")
xgb_model.fit(X_train, y_train)

# 5. Predictions and Evaluation
preds = xgb_model.predict(X_test)
metrics = calculate_metrics(y_test, preds)

# 6. Output metrics
print_metrics("XGBoost", metrics)

# Save metrics for comparison
save_results('XGBoost', metrics, 'results_xgboost.csv')