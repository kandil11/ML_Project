import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from prophet import Prophet
from utils import calculate_metrics, print_metrics, save_results
import matplotlib.pyplot as plt

# 1. Load data
print("Loading data for Prophet...")
from utils import load_data
df = load_data()

# Reshape one item for Prophet (e.g. the first unique item in the full dataset)
first_id = df['id'].iloc[0]
df_item = df[df['id'] == first_id].copy()
df_item = df_item[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
df_item['ds'] = pd.to_datetime(df_item['ds'])

# 2. Split data (Time-based split)
train_size = int(len(df_item) * 0.8)
train_df = df_item.iloc[:train_size]
test_df = df_item.iloc[train_size:]

# 3. Build and train Prophet model
print("Training Prophet...")
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(train_df)

# 4. Predictions
future = model.make_future_dataframe(periods=len(test_df))
forecast = model.predict(future)

# 5. Evaluation
preds = forecast.iloc[train_size:]['yhat'].values
y_test = test_df['y'].values
metrics = calculate_metrics(y_test, preds)

# 6. Output metrics
print_metrics("Prophet", metrics)

# Save metrics for comparison
save_results('Prophet', metrics, 'results_prophet.csv')

# Optional: Plot
fig = model.plot(forecast)
plt.title('Prophet Forecast')
os.makedirs('results/plots', exist_ok=True)
plot_path = 'results/plots/prophet_forecast.png'
plt.savefig(plot_path)
print(f"Prophet forecast plot saved as {plot_path}")
