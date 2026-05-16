import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils import load_data, calculate_metrics, print_metrics, save_results
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
print("Loading data for Random Forest...")
df = load_data()

# 2. Prepare features and target
drop_cols = [c for c in ['id', 'd', 'date', 'sales'] if c in df.columns]
X = df.drop(drop_cols, axis=1)
y = df['sales']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

# 5. Predictions and Evaluation
preds = rf_model.predict(X_test)
metrics = calculate_metrics(y_test, preds)

# 6. Output metrics
print_metrics("Random Forest", metrics)

# Save metrics for comparison
save_results('Random Forest', metrics, 'results_rf.csv')

# 7. Feature Importance
importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
importance = importance.sort_values('importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance)
plt.title('Random Forest Feature Importance')
os.makedirs('results/plots', exist_ok=True)
plot_path = 'results/plots/rf_feature_importance.png'
plt.savefig(plot_path)
print(f"Feature importance plot saved as {plot_path}")