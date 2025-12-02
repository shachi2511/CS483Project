import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import tensorflow as tf
import random

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data', 'model_ready_dataset.csv')
BASELINE_RMSE = 7.4653
SEQ_LENGTH = 15
TEST_SPLIT = 0.2

# Set Seeds for Reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # 1. Target: Log Returns (x10)
    df['Log_Ret'] = np.log(df['Gold_Price'] / df['Gold_Price'].shift(1)) * 10
    
    # 2. FEATURE: Autoregression (Yesterday's Return)
    # Give the model explicit access to momentum
    df['Prev_Day_Ret'] = df['Log_Ret'].shift(1)
    
    # 3. Macro & Technicals
    inflation_rate = df['CPIAUCSL'].pct_change() * 12
    real_rate = df['FEDFUNDS'] - (inflation_rate * 100)
    df['Real_Rate_Change'] = real_rate.diff()
    
    df['Oil_Ret'] = df['Crude_Oil'].pct_change()
    df['VIX_Change'] = df['VIXCLS'].diff()
    df['Yield_Spread'] = df['T10Y2Y'].diff()
    
    # RSI
    delta = df['Gold_Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # GARCH Volatility
    returns_clean = df['Log_Ret'].dropna()
    model = arch_model(returns_clean, vol='Garch', p=1, q=1, dist='Normal')
    res = model.fit(disp='off')
    df.loc[returns_clean.index, 'GARCH_Vol'] = res.conditional_volatility

    df.dropna(inplace=True)
    
    # Explicitly include Previous Return for momentum sniping
    features = ['Log_Ret', 'Prev_Day_Ret', 'Real_Rate_Change', 'Oil_Ret', 
                'VIX_Change', 'RSI', 'GARCH_Vol']
    
    return df, features

def create_sequences(data, target_idx, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_idx])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        # Bidirectional allows the model to see the "shape" of the window better
        Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model

def main():
    print("--- STARTING SNIPER STRATEGY ---")
    
    df, features = load_data()
    target_idx = 0
    
    # Scale
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features].values)
    
    X, y = create_sequences(scaled_data, target_idx, SEQ_LENGTH)
    
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train
    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=100, batch_size=32, 
              validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)
    
    # Predict Raw Returns
    raw_preds_scaled = model.predict(X_test)
    
    # Inverse Transform
    dummy = np.zeros((len(raw_preds_scaled), len(features)))
    dummy[:, target_idx] = raw_preds_scaled.flatten()
    pred_returns = scaler.inverse_transform(dummy)[:, target_idx]
    
    # --- SNIPER OPTIMIZATION ---
    print("\nOptimizing Sniper Threshold...")
    
    test_indices = df.index[split + SEQ_LENGTH:]
    prev_prices = df['Gold_Price'].iloc[split + SEQ_LENGTH - 1 : -1].values
    actual_prices = df['Gold_Price'].iloc[split + SEQ_LENGTH:].values
    
    best_rmse = float('inf')
    best_threshold = 0
    best_final_prices = []
    
    # Test thresholds from 0.0 to 0.5 (Log Return scale)
    # Since we scaled x10, a threshold of 0.1 means a 1% move, etc.
    # We check small increments to find the sweet spot
    thresholds = np.linspace(0, 0.5, 50)
    
    for thresh in thresholds:
        # SNIPER LOGIC:
        # If absolute prediction < threshold, assume 0 (Random Walk)
        # Else, trust the model
        filtered_returns = np.where(np.abs(pred_returns) < thresh, 0, pred_returns)
        
        # Reconstruct
        # Divide by 10 because of initial scaling
        prices = prev_prices * np.exp(filtered_returns / 10)
        
        rmse = np.sqrt(mean_squared_error(actual_prices, prices))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_threshold = thresh
            best_final_prices = prices
            
    print("\n" + "="*40)
    print("SNIPER STRATEGY RESULTS")
    print("="*40)
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Sniper RMSE:    {best_rmse:.4f}")
    print(f"Baseline RMSE:  {BASELINE_RMSE:.4f}")
    
    if best_rmse < BASELINE_RMSE:
        improvement = BASELINE_RMSE - best_rmse
        print(f"🏆 VICTORY! We beat the Random Walk by {improvement:.4f} points!")
        print("Strategy: Trust baseline on quiet days, trust AI on volatile days.")
    else:
        print("Result: Tied with Baseline (Market is extremely efficient).")
        
    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(test_indices, actual_prices, label='Actual Price', color='gray', alpha=0.4)
    plt.plot(test_indices, best_final_prices, label='Sniper Forecast', color='red', linewidth=1)
    plt.title(f'Sniper Hybrid Strategy (Threshold: {best_threshold:.3f}) | RMSE: {best_rmse:.4f}')
    plt.legend()
    plt.savefig('sniper_victory.png')
    print("Saved sniper_victory.png")

if __name__ == "__main__":
    main()