import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from utils import calculate_metrics, print_metrics, save_results

# 1. Load and Preprocess Data
print("Loading data for TFT...")
from utils import load_data
df = load_data()

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
if 'time_idx' not in df.columns and 'd' in df.columns:
    df['time_idx'] = df['d'].apply(lambda x: int(x.split('_')[1]) if isinstance(x, str) else x)

df['sales'] = df['sales'].astype(float)

# TFT needs strings for categorical variables. Detect what's available.
potential_cats = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
cat_cols = [c for c in potential_cats if c in df.columns]
for col in cat_cols:
    df[col] = df[col].astype(str)

# 2. Define Dataset
max_prediction_length = 28
max_encoder_length = 56
training_cutoff = df['time_idx'].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="sales",
    group_ids=["id"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=[c for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"] if c in df.columns],
    time_varying_known_reals=[c for c in ["time_idx", "wday", "month", "year", "price_in_dollars", "snap"] if c in df.columns],
    time_varying_unknown_reals=["sales"],
    target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Create validation set
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

# Create dataloaders
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

# 3. Build Model
pl.seed_everything(42)
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7, # default for QuantileLoss
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# 4. Train
trainer = pl.Trainer(
    max_epochs=1, # Just 1 epoch for demonstration
    accelerator="auto",
    enable_model_summary=True,
    gradient_clip_val=0.1,
)

print("Training TFT (simplified)...")
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# 5. Evaluation
print("Evaluating TFT...")
raw_predictions = tft.predict(val_dataloader, mode="raw", return_x=True)
preds = raw_predictions.output.prediction[:, :, 3].flatten().cpu().numpy() # 3 is the median quantile (0.5)
y_test = torch.cat([y[0] for x, y in iter(val_dataloader)]).flatten().cpu().numpy()

metrics = calculate_metrics(y_test, preds)

# 6. Output metrics
print_metrics("TFT", metrics)

# Save metrics for comparison
save_results('TFT', metrics, 'results_tft.csv')
