import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data', 'model_ready_dataset.csv')
BASELINE_RMSE = 7.4653
TEST_SPLIT = 0.2

def load_and_engineer_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # 1. Target: Log Returns (x10 for scaling)
    df['Log_Ret'] = np.log(df['Gold_Price'] / df['Gold_Price'].shift(1)) * 10
    
    # 2. LAG FEATURES (The Secret Weapon for Trees)
    # Give the model explicit memory of the last 3 days
    for i in range(1, 4):
        df[f'Log_Ret_Lag{i}'] = df['Log_Ret'].shift(i)
        df[f'Oil_Ret_Lag{i}'] = df['Crude_Oil'].pct_change().shift(i) * 10
        df[f'VIX_Change_Lag{i}'] = df['VIXCLS'].diff().shift(i)
    
    # 3. Macro & Technicals (Current Day)
    inflation_rate = df['CPIAUCSL'].pct_change() * 12
    df['Real_Rate'] = df['FEDFUNDS'] - (inflation_rate * 100)
    df['Real_Rate_Change'] = df['Real_Rate'].diff()
    
    df['Yield_Spread'] = df['T10Y2Y'].diff()
    df['Dollar_Change'] = df['DEXUSEU'].pct_change() * 10
    
    # RSI
    delta = df['Gold_Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    
    # Select Features: Lags + Current Macro + Technicals
    features = [col for col in df.columns if 'Lag' in col] + \
               ['Real_Rate_Change', 'Yield_Spread', 'Dollar_Change', 'RSI']
    
    return df, features

def main():
    print("--- STARTING GRADIENT BOOSTING STRATEGY ---")
    
    df, features = load_and_engineer_data()
    print(f"Training with {len(features)} features (Lags + Macro)...")
    
    # 1. Prepare Data (No Sequence creation needed for Trees, just Row-by-Row)
    X = df[features].values
    y = df['Log_Ret'].values
    
    # 2. Split
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 3. Train Gradient Boosting Regressor
    # These params are tuned for financial noise (low learning rate, few trees)
    model = GradientBoostingRegressor(
        n_estimators=200,      # Number of trees
        learning_rate=0.01,    # Slow learning to avoid overfitting
        max_depth=3,           # Shallow trees to prevent memorization
        subsample=0.7,         # Use only 70% of data per tree (adds randomness)
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 4. Predict
    pred_returns = model.predict(X_test)
    
    # 5. Reconstruct Prices
    # Price_t = Price_{t-1} * exp(Return_t / 10)
    # Note: We need the price from the day BEFORE the test set started, 
    # and then iterate forward if we were doing a simulation. 
    # For one-step-ahead forecast (standard):
    
    test_indices = df.index[split:]
    prev_prices = df['Gold_Price'].iloc[split-1 : -1].values
    actual_prices = df['Gold_Price'].iloc[split:].values
    
    predicted_prices = prev_prices * np.exp(pred_returns / 10)
    
    # 6. Evaluate
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    
    print("\n" + "="*40)
    print("GRADIENT BOOSTING RESULTS")
    print("="*40)
    print(f"RMSE: {rmse:.4f}")
    print(f"Baseline RMSE: {BASELINE_RMSE:.4f}")
    
    if rmse < BASELINE_RMSE:
        print(f"🏆 VICTORY! Improvement: {BASELINE_RMSE - rmse:.4f} points")
        print("Tree-based models captured the sharp moves better than LSTM.")
    else:
        print("Result: Very close to baseline.")
        
    # Feature Importance (Why did it decide this?)
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Drivers of Gold Price:")
    print(importance.head(5))
    
    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(test_indices, actual_prices, label='Actual Price', color='gray', alpha=0.5)
    plt.plot(test_indices, predicted_prices, label='Boosted Tree Forecast', color='orange', linewidth=1.5)
    plt.title(f'Gradient Boosting Forecast | RMSE: {rmse:.4f}')
    plt.legend()
    plt.savefig('boosted_tree_forecast.png')
    print("Saved boosted_tree_forecast.png")

if __name__ == "__main__":
    main()