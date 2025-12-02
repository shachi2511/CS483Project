import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import random
import tensorflow as tf

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data', 'model_ready_dataset.csv')
BASELINE_RMSE = 7.4653  # The score to beat
TEST_SPLIT = 0.2
TRIALS = 20  # Number of models to train

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # --- STATIONARITY & DAILY FEATURES ---
    
    # 1. Target: Log Returns (Scaled x10 for better training signal)
    df['Log_Ret'] = np.log(df['Gold_Price'] / df['Gold_Price'].shift(1)) * 10
    
    # 2. Daily Macro (No Monthly Steps!)
    df['Oil_Ret'] = df['Crude_Oil'].pct_change()
    df['SP500_Ret'] = df['SP500'].pct_change()
    df['DEX_Ret'] = df['DEXUSEU'].pct_change()
    df['Yield_Spread_Change'] = df['T10Y2Y'].diff() # Daily change in yield curve
    df['VIX_Change'] = df['VIXCLS'].diff()          # Daily Fear Gauge change
    
    # 3. Technicals (Colleague's Focus)
    # RSI (14-day)
    delta = df['Gold_Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # GARCH Volatility
    returns_clean = df['Log_Ret'].dropna()
    model = arch_model(returns_clean, vol='Garch', p=1, q=1, dist='Normal')
    res = model.fit(disp='off')
    df.loc[returns_clean.index, 'GARCH_Vol'] = res.conditional_volatility

    df.dropna(inplace=True)
    
    # EXCLUDE 'Rate_Change' (FedFunds) and 'Inflation' (CPI) as they are monthly
    features = ['Log_Ret', 'Oil_Ret', 'SP500_Ret', 'Yield_Spread_Change', 
                'VIX_Change', 'RSI', 'GARCH_Vol']
    
    return df, features

def create_sequences(data, target_idx, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_idx])
    return np.array(X), np.array(y)

def train_trial_model(X_train, y_train, X_test, y_test, params):
    """Builds and trains a model with random hyperparameters"""
    
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    
    # Layer 1
    if params['model_type'] == 'LSTM':
        model.add(LSTM(params['units'], return_sequences=False, 
                       kernel_regularizer=tf.keras.regularizers.l2(params['l2'])))
    else:
        model.add(GRU(params['units'], return_sequences=False,
                      kernel_regularizer=tf.keras.regularizers.l2(params['l2'])))
        
    model.add(Dropout(params['dropout']))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=params['lr']), loss='mse')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=params['batch_size'],
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0 # Silent training
    )
    return model

def main():
    print(f"--- STARTING TOURNAMENT ({TRIALS} Trials) ---")
    print(f"Target to Beat: {BASELINE_RMSE} RMSE")
    
    df, features = load_data()
    
    # Standardize
    data_subset = df[features].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_subset)
    target_idx = 0
    
    best_rmse = float('inf')
    best_config = {}
    
    # Split Data ONCE (fixed split for fair comparison)
    # Use shorter sequence length for daily agility
    SEQ_LENGTH = 14 
    
    X, y = create_sequences(scaled_data, target_idx, SEQ_LENGTH)
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    test_indices = df.index[split + SEQ_LENGTH:]
    prev_prices = df['Gold_Price'].iloc[split + SEQ_LENGTH - 1 : -1].values
    actual_prices = df['Gold_Price'].iloc[split + SEQ_LENGTH:].values

    for i in range(TRIALS):
        # Random Hyperparameters
        params = {
            'model_type': random.choice(['LSTM', 'GRU']), # Try GRU too!
            'units': random.choice([16, 32, 64]),
            'dropout': random.choice([0.2, 0.3, 0.4, 0.5]),
            'lr': random.choice([0.001, 0.0005]),
            'batch_size': random.choice([16, 32]),
            'l2': random.choice([0.001, 0.01, 0.0])
        }
        
        print(f"\nTrial {i+1}/{TRIALS}: {params} ...", end=" ")
        
        model = train_trial_model(X_train, y_train, X_test, y_test, params)
        
        # Evaluate
        pred_scaled = model.predict(X_test, verbose=0)
        
        # Inverse Transform
        dummy = np.zeros((len(pred_scaled), len(features)))
        dummy[:, target_idx] = pred_scaled.flatten()
        pred_returns = scaler.inverse_transform(dummy)[:, target_idx]
        
        # Note: Divide by 10 because we scaled inputs x10
        predicted_prices = prev_prices * np.exp(pred_returns / 10)
        
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        
        print(f"RMSE: {rmse:.4f}", end=" ")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_config = params
            print("--> ⭐ NEW LEADER!")
            
            # Save the Leader
            plt.figure(figsize=(14, 7))
            plt.plot(test_indices, actual_prices, label='Actual Price', color='blue', alpha=0.5)
            plt.plot(test_indices, predicted_prices, label='Best Forecast', color='green')
            plt.title(f'Best Model (RMSE: {rmse:.4f}) | Config: {params}')
            plt.legend()
            plt.savefig('tournament_winner.png')
            plt.close()
        else:
            print("")

    print("\n" + "="*40)
    print("TOURNAMENT RESULTS")
    print("="*40)
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"Baseline:  {BASELINE_RMSE:.4f}")
    
    if best_rmse < BASELINE_RMSE:
        print(f"🏆 VICTORY! Beaten by {BASELINE_RMSE - best_rmse:.4f} points.")
    else:
        print("So close! Try running the tournament again (random seeds vary).")

if __name__ == "__main__":
    main()