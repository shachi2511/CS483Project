import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
import os
import warnings
import numpy as np # Ensure numpy is available inside main if needed


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data', 'model_ready_dataset.csv')

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # Re-calculate the transformations we used in the model
    df['Log_Ret'] = np.log(df['Gold_Price'] / df['Gold_Price'].shift(1)) * 10
    df['Oil_Ret'] = df['Crude_Oil'].pct_change()
    df.dropna(inplace=True)
    return df

def run_adf_test(series, name):
    print(f"\n--- Augmented Dickey-Fuller (ADF) Test: {name} ---")
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value:       {result[1]:.4f}")
    
    if result[1] < 0.05:
        print(">> RESULT: Stationary (Good for Modeling)")
    else:
        print(">> RESULT: Non-Stationary (Bad for Modeling - Needs Differencing)")

def run_kpss_test(series, name):
    print(f"\n--- KPSS Test: {name} ---")
    # KPSS Null Hypothesis: Series IS Stationary
    result = kpss(series.dropna(), regression='c', nlags="auto")
    print(f"KPSS Statistic: {result[0]:.4f}")
    print(f"p-value:        {result[1]:.4f}")
    
    if result[1] > 0.05:
        print(">> RESULT: Stationary (Good for Modeling)")
    else:
        print(">> RESULT: Non-Stationary (Bad for Modeling)")

def main():
    print("Loading Data for Statistical Validation...")    
    df = load_data()
    
    # 1. TEST RAW PRICE (The "Wrong" Way)
    # We expect this to fail, proving why your first LSTM model (RMSE 44) failed.
    print("\n" + "="*50)
    print("TEST 1: RAW GOLD PRICE (Level Data)")
    print("="*50)
    run_adf_test(df['Gold_Price'], "Gold Price ($)")
    run_kpss_test(df['Gold_Price'], "Gold Price ($)")
    
    # 2. TEST LOG RETURNS (The "Right" Way)
    # We expect this to pass, proving why your Alpha/Tree models (RMSE 6-7) worked.
    print("\n" + "="*50)
    print("TEST 2: LOG RETURNS (Transformed Data)")
    print("="*50)
    run_adf_test(df['Log_Ret'], "Gold Log Returns (%)")
    run_kpss_test(df['Log_Ret'], "Gold Log Returns (%)")

    # 3. VISUAL PROOF
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot Raw Price
    axes[0].plot(df['Gold_Price'], color='blue')
    axes[0].set_title('Raw Gold Price (Non-Stationary)')
    axes[0].set_ylabel('Price ($)')
    axes[0].grid(True)
    
    # Plot Returns
    axes[1].plot(df['Log_Ret'], color='green')
    axes[1].set_title('Gold Log Returns (Stationary)')
    axes[1].set_ylabel('Daily Return (%)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('stationarity_proof.png')
    print("\nSaved visualization to 'stationarity_proof.png'")

if __name__ == "__main__":
    main()