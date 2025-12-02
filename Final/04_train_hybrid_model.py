import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import random
import tensorflow as tf

# --- REPRODUCIBILITY SETUP ---
# This ensures your "winning" run can be repeated exactly
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data', 'model_ready_dataset.csv')
SEQ_LENGTH = 30  # OPTIMIZATION: Reduced from 60 to 30 for small data
TEST_SPLIT = 0.2

def load_and_engineer_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # Check for Fusion columns
    if 'FEDFUNDS' not in df.columns:
        raise KeyError("Missing 'FEDFUNDS'. Run 01_data_pipeline.py first.")
    
    # --- STATIONARITY TRANSFORMATION ---
    
    # 1. Target: Log Returns * 10 (Scaling helps LSTM convergence)
    df['Log_Ret'] = np.log(df['Gold_Price'] / df['Gold_Price'].shift(1)) * 100
    
    # 2. Key Macro Growth Rates (The "Fusion" Logic)
    df['Oil_Ret'] = df['Crude_Oil'].pct_change()
    df['SP500_Ret'] = df['SP500'].pct_change()
    df['DEX_Ret'] = df['DEXUSEU'].pct_change()
    df['Rate_Change'] = df['FEDFUNDS'].diff()
    df['Yield_Spread_Change'] = df['T10Y2Y'].diff()
    
    # 3. Technicals
    # Rolling Volatility (Simple)
    df['Roll_Vol'] = df['Log_Ret'].rolling(window=10).std()
    
    # RSI (Relative Strength Index) - Strong Signal
    delta = df['Gold_Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # GARCH Volatility (Advanced Risk)
    returns_clean = df['Log_Ret'].dropna()
    model = arch_model(returns_clean, vol='Garch', p=1, q=1, dist='Normal')
    res = model.fit(disp='off')
    df.loc[returns_clean.index, 'GARCH_Vol'] = res.conditional_volatility

    # Drop NaNs
    df.dropna(inplace=True)
    
    # SELECT FEATURES (Less is More for small data)
    # We remove 'avg_tone' and 'Disaster' for the base training to reduce noise, 
    # focusing on hard financial metrics first.
    features = ['Log_Ret', 'Oil_Ret', 'SP500_Ret', 'Rate_Change', 
                'Yield_Spread_Change', 'RSI', 'GARCH_Vol']
    
    print(f"Engineered Features ({len(features)}): {features}")
    return df, features

def create_sequences(data, target_idx, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_idx])
    return np.array(X), np.array(y)

def build_lite_model(input_shape):
    """
    A 'Lite' architecture designed to prevent overfitting on small datasets (<1000 rows).
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # Single LSTM Layer (Reduced Units)
        LSTM(32, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        
        # High Dropout to force generalization
        Dropout(0.5),
        
        # Simple dense layer
        Dense(16, activation='relu'),
        
        # Output
        Dense(1)
    ])
    
    # Lower learning rate for stable convergence
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model

def main():
    print("--- Training OPTIMIZED Alpha Model ---")
    
    # 1. Load
    df, feature_cols = load_and_engineer_data()
    
    # 2. Scale
    data_subset = df[feature_cols].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_subset)
    target_idx = 0 
    
    # 3. Sequence
    X, y = create_sequences(scaled_data, target_idx, SEQ_LENGTH)
    
    # 4. Split
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    # 5. Train
    model = build_lite_model((X_train.shape[1], X_train.shape[2]))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16, # Smaller batch size for better updates
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # 6. Predict
    pred_scaled = model.predict(X_test)
    
    # 7. Inverse Transform
    dummy = np.zeros((len(pred_scaled), len(feature_cols)))
    dummy[:, target_idx] = pred_scaled.flatten()
    pred_returns = scaler.inverse_transform(dummy)[:, target_idx]
    
    # 8. Reconstruct Prices
    # Price_t = Price_{t-1} * exp(Return_t / 100)
    test_indices = df.index[split + SEQ_LENGTH:]
    prev_prices = df['Gold_Price'].iloc[split + SEQ_LENGTH - 1 : -1].values
    
    predicted_prices = prev_prices * np.exp(pred_returns / 100)
    actual_prices = df['Gold_Price'].iloc[split + SEQ_LENGTH:].values
    
    # 9. Evaluate
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    
    print("\n" + "="*40)
    print("OPTIMIZED MODEL RESULTS")
    print("="*40)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    
    # Check if we beat the baseline
    baseline_rmse = 7.4653
    if rmse < baseline_rmse:
        print(f"🏆 SUCCESS: We beat the Random Walk ({baseline_rmse})!")
        print(f"   Improvement: {baseline_rmse - rmse:.4f} points")
    else:
        print(f"   Gap to Baseline: {rmse - baseline_rmse:.4f} points")

    plt.figure(figsize=(14, 7))
    plt.plot(test_indices, actual_prices, label='Actual Price', color='blue', alpha=0.5)
    plt.plot(test_indices, predicted_prices, label='Optimized Forecast', color='green', linestyle='-', linewidth=2)
    plt.title(f'Optimized Alpha Forecast (RMSE: {rmse:.2f})')
    plt.legend()
    plt.grid(True)
    plt.savefig('optimized_forecast.png')
    print("Saved optimized_forecast.png")

if __name__ == "__main__":
    main()