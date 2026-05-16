

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Determine paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(project_root, "data", "m5-forecasting-accuracy")

print("--- Loading Datasets ---")
sales_validation=pd.read_csv(os.path.join(data_dir, 'sales_train_validation.csv'))
cal=pd.read_csv(os.path.join(data_dir, 'calendar.csv'))
sell_price = pd.read_csv(os.path.join(data_dir, 'sell_prices.csv'))

print("--- Melting Sales Validation Data (This may take a while) ---")
data = sales_validation.melt(
    id_vars=["id","item_id", "dept_id","cat_id","store_id","state_id"],  # columns to keep
    var_name="d",                          # new column name for day labels (d_1, d_2, ...)
    value_name="sales"                     # new column name for sales values
)

data.drop(columns=['id'],inplace=True)

print("--- Processing Calendar Data ---")
cal["date"]= pd.to_datetime(cal["date"])

cal["event_name_1"]= cal["event_name_1"].fillna("No event")
cal["event_type_1"]= cal["event_type_1"].fillna("No event")
cal["event_name_2"]= cal["event_name_2"].fillna("No event")
cal["event_type_2"]= cal["event_type_2"].fillna("No event")

# Merge data and cal dataframes on the 'd' column
print("--- Merging Sales and Calendar Data ---")
merged_data = pd.merge(data, cal, on='d', how='left')

print("--- Calculating SNAP Variables ---")
conditions = [
    merged_data["state_id"] == "CA",
    merged_data["state_id"] == "TX",
    merged_data["state_id"] == "WI"
]
choices= [
    merged_data["snap_CA"],
    merged_data["snap_TX"],
    merged_data["snap_WI"]
]
merged_data["snap"]= np.select(conditions, choices)

merged_data.drop(columns=['snap_CA','snap_TX','snap_WI'],inplace=True)

print("--- Merging with Sell Prices Data ---")
data = merged_data.merge(
        sell_price,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="m:1"  # many sales rows per unique price row is expected
    )

data["price_in_dollars"]=data["sell_price"]*data["sales"]

# data.to_csv("data/processed_data.csv") # Will save at the end of the script instead

data.isna().sum()

print(data["event_name_1"].unique())
print(data["event_name_2"].unique())
print(data["event_type_1"].unique())
print(data["event_type_2"].unique())

data.head(10)

print(data.info())

#Display summary statistics for all numerical columns
print(data.describe())

#Count how many unique products and stores are in the dataset
print("Number of unique items:", data["item_id"].nunique())
print("Number of unique stores:", data["store_id"].nunique())

'''Check correlation between Sales column and Price_in_dollars column
A positive correlation means: when price increases, sales also increase'''
print(data[["sales", "price_in_dollars"]].corr())

'''Check which events appear most frequently in the dataset
This can help identify which holidays might affect sales'''

print("Most frequent event_name_1 values:")
print(data["event_name_1"].value_counts().head())
print("\nMost frequent event_type_1 values:")
print(data["event_type_1"].value_counts().head())

# ==========================================
# Data Visualization (According to Project Objectives)
# ==========================================
print("--- Starting Data Visualization (EDA) ---")

# 1. Data Distribution graph: Target Variable (Sales)
plt.figure(figsize=(10, 6))
sns.histplot(data[data['sales'] > 0]['sales'], bins=50, kde=True, color='blue')
plt.title('Distribution of Non-Zero Sales (Target Variable)')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.savefig('results/plots/sales_distribution.png')
plt.close()

# 2. Date Distribution: Total Daily Sales over Time
if 'date' in data.columns:
    # Sample for plotting
    daily_sales = data.sample(min(100000, len(data)), random_state=42).groupby('date')['sales'].sum().reset_index()
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=daily_sales, x='date', y='sales', color='orange')
    plt.title('Total Daily Sales Over Time (Sampled)')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.savefig('results/plots/daily_sales.png')
    plt.close()

# 3. Objective: Impact of SNAP on Sales
if 'snap' in data.columns:
    plt.figure(figsize=(8, 6))
    sample_data = data.sample(min(100000, len(data)), random_state=42)
    sns.barplot(data=sample_data, x='snap', y='sales', hue='snap', palette='Set2', legend=False)
    plt.title('Average Sales: SNAP Days vs Non-SNAP Days (Sampled)')
    plt.xlabel('SNAP Day (0 = No, 1 = Yes)')
    plt.ylabel('Average Sales')
    plt.savefig('results/plots/snap_impact.png')
    plt.close()

# 4. Objective: Impact of SNAP across different Product Categories
if 'cat_id' in data.columns and 'snap' in data.columns:
    plt.figure(figsize=(10, 6))
    sample_data = data.sample(min(100000, len(data)), random_state=42)
    sns.barplot(data=sample_data, x='cat_id', y='sales', hue='snap', palette='Set1')
    plt.title('Average Sales by Category and SNAP Status (Sampled)')
    plt.xlabel('Category')
    plt.ylabel('Average Sales')
    plt.legend(title='SNAP Day')
    plt.savefig('results/plots/category_snap_impact.png')
    plt.close()

# 5. Price vs Sales distribution during SNAP vs non-SNAP
if 'price_in_dollars' in data.columns and 'snap' in data.columns:
    os.makedirs('results/plots', exist_ok=True)
    # Sample data for scatter plot to avoid memory/performance issues
    sample_data = data[data['sales'] > 0].sample(min(10000, len(data[data['sales'] > 0])), random_state=42)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sample_data, x='price_in_dollars', y='sales', hue='snap', alpha=0.5, palette='coolwarm')
    plt.title('Price vs Sales (Sampled) - SNAP vs Non-SNAP')
    plt.xlabel('Price in Dollars')
    plt.ylabel('Sales Volume')
    plt.savefig('results/plots/price_vs_sales_snap.png')
    plt.close()

# ==========================================

print("--- Cleaning up unused columns ---")
data.drop(columns = "sell_price",inplace = True)
data.drop(columns="dept_id",inplace = True)
data.drop(columns = "cat_id",inplace = True)
data.drop(columns = "state_id",inplace = True)
data.drop(columns="month",inplace = True)
data.drop(columns="year",inplace = True)
data.drop(columns="weekday",inplace=True)

data.drop(columns="Unnamed: 0",inplace = True, errors='ignore')

data.fillna(0,inplace = True)

print("--- Sorting Data ---")
data.sort_values(by=["store_id", "item_id", "d"], inplace=True)
data.reset_index(drop=True, inplace=True)

print("--- Saving Processed Data to CSV ---")
output_dir = os.path.join(project_root, "data", "processed")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "processed_data.csv")
data.to_csv(output_path, index=False)
print(f"--- Processed data saved to {output_path} ---")
print("--- Script Completed Successfully ---")