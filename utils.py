import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

def load_data(nrows=None):
    """
    Loads the processed M5 dataset from the data/processed directory.
    """
    processed_path = 'data/processed/processed_data.csv'
    
    print(f"Loading processed data from {processed_path}...")
    df = pd.read_csv(processed_path, nrows=nrows)
    
    # Recreate 'id' if missing (needed for TFT)
    if 'id' not in df.columns and 'item_id' in df.columns and 'store_id' in df.columns:
        df['id'] = df['item_id'].astype(str) + "_" + df['store_id'].astype(str)
        
    # Ensure categorical columns are encoded if they are strings
    from pandas.api.types import is_numeric_dtype
    le = LabelEncoder()
    for col in df.columns:
        if not is_numeric_dtype(df[col]) and col != 'date':
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Add d_num if not present
    if 'd' in df.columns and 'd_num' not in df.columns:
        df['d_num'] = df['d'].apply(lambda x: int(x.split('_')[1]) if isinstance(x, str) else x)
        
    return df

def calculate_metrics(y_true, y_pred):
    """
    Calculates standard ML regression metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Metric calculations
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    
    # MAPE calculation (handle zero division)
    mask = y_true_arr != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100
    else:
        mape = np.nan

    # Forecast Accuracy (1 - WAPE)
    sum_abs_err = np.sum(np.abs(y_true_arr - y_pred_arr))
    sum_actuals = np.sum(np.abs(y_true_arr))
    if sum_actuals != 0:
        accuracy = max(0, 100 * (1 - (sum_abs_err / sum_actuals)))
    else:
        accuracy = np.nan
        
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Accuracy': accuracy
    }

import os

def save_results(model_name, metrics, filename):
    """
    Saves metrics to the results/metrics/ directory.
    """
    os.makedirs('results/metrics', exist_ok=True)
    df = pd.DataFrame([metrics])
    df['Model'] = model_name
    path = os.path.join('results/metrics', filename)
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")

def print_metrics(model_name, metrics):
    print(f"\n--- {model_name} Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("-" * (len(model_name) + 15))
