import subprocess
import os

scripts = [
    'models/baseline.py',
    'models/random_forest.py',
    'models/XGBoost.py',
    'models/LightGBM.py',
    'models/Prophet.py',
    'models/TFT.py'
]

print("Starting model training and evaluation...")

for script in scripts:
    print(f"\n>>> Running {script}...")
    try:
        # We use a smaller nrows in the scripts themselves for this demo
        subprocess.run(['python3', script], check=True)
    except Exception as e:
        print(f"Error running {script}: {e}")

print("\n>>> Generating comparison report...")
subprocess.run(['python3', 'compare_models.py'], check=True)

print("\nAll tasks completed!")
