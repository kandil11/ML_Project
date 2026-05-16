import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

def compare_results():
    # 1. Collect all results files from the results/metrics directory
    metrics_dir = 'results/metrics'
    files = glob.glob(os.path.join(metrics_dir, 'results_*.csv'))
    
    if not files:
        print(f"No results files found in {metrics_dir}. Run the model scripts first.")
        return
    
    all_results = []
    for f in files:
        try:
            df = pd.read_csv(f)
            all_results.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    comparison_df = pd.concat(all_results, ignore_index=True)
    
    # 2. Display the table
    print("\n" + "="*30)
    print("   MODEL COMPARISON REPORT")
    print("="*30)
    comparison_df = comparison_df.sort_values(by='RMSE')
    print(comparison_df.to_string(index=False))
    
    report_path = os.path.join(metrics_dir, 'model_comparison.csv')
    comparison_df.to_csv(report_path, index=False)
    print(f"\nFull comparison saved to {report_path}")
    
    # 3. Plot comparison
    metrics_to_plot = ['RMSE', 'MAE', 'R2']
    plt.figure(figsize=(18, 6))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i+1)
        sns.barplot(x='Model', y=metric, data=comparison_df, palette='viridis')
        plt.title(f'{metric} Comparison', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = 'results/plots/model_comparison_plot.png'
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig(plot_path)
    print(f"Comparison plot saved as {plot_path}")

if __name__ == "__main__":
    compare_results()
