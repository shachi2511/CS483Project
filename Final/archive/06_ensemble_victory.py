import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import tensorflow as tf

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data', 'model_ready_dataset.csv')
BASELINE_RMSE = 7.4653
ENSEMBLE_MEMBERS = 5  # We will train 5 models and average them
SEQ_LENGTH = 20       # Short memory for daily agility

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # 1. Target: Log Returns (x10 scaling)
    df['Log_Ret'] = np.log(df['Gold_Price'] / df['Gold_Price'].shift(1)) * 10
    
    # 2. FEATURE: Real Interest Rate Interaction (Colleague's Insight)
    # Gold hates it when Real Rates (FedFunds - Inflation) go up.
    # We calculate the monthly change in Real Rates.
    inflation_rate = df['CPIAUCSL'].pct_change() * 12 # Annualized Inflation
    real_rate = df['FEDFUNDS'] - (inflation_rate * 100)
    df['Real_Rate_Change'] = real_rate.diff()
    
    # 3. Standard Macro Features
    df['Oil_Ret'] = df['Crude_Oil'].pct_change()
    df['DEX_Ret'] = df['DEXUSEU'].pct_change()
    df['Yield_Spread'] = df['T10Y2Y'].diff()
    df['VIX_Change'] = df['VIXCLS'].diff()

    # 4. Technicals
    df['Roll_Vol'] = df['Log_Ret'].rolling(10).std()
    
    # RSI
    delta = df['Gold_Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    
    # Feature Selection: Focus on the "Real Rate" impact + Momentum
    features = ['Log_Ret', 'Real_Rate_Change', 'Oil_Ret', 'DEX_Ret', 
                'Yield_Spread', 'VIX_Change', 'RSI']
    
    return df, features

def create_sequences(data, target_idx, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_idx])
    return np.array(X), np.array(y)

def train_single_model(X_train, y_train, X_test, y_test, seed):
    """Trains one GRU model with a specific random seed"""
    
    # Set seed for diversity between models
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        # GRU is simpler/faster than LSTM for small noisy data
        GRU(32, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        Dropout(0.4),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0 # Silent
    )
    return model

def main():
    print(f"--- STARTING ENSEMBLE TRAINING ({ENSEMBLE_MEMBERS} Models) ---")
    
    df, features = load_data()
    print(f"Features used: {features}")
    
    # Scale
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features].values)
    target_idx = 0
    
    # Sequence
    X, y = create_sequences(scaled_data, target_idx, SEQ_LENGTH)
    
    # Split
    split = int(len(X) * (0.8)) # 80% Train, 20% Test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Matrix to store predictions from all models
    all_predictions = np.zeros((len(X_test), ENSEMBLE_MEMBERS))
    
    for i in range(ENSEMBLE_MEMBERS):
        print(f"Training Model {i+1}...", end=" ")
        model = train_single_model(X_train, y_train, X_test, y_test, seed=42+i)
        
        # Predict
        pred = model.predict(X_test, verbose=0)
        all_predictions[:, i] = pred.flatten()
        print("Done.")
        
    # --- ENSEMBLE AVERAGE ---
    print("\nCalculating Ensemble Average...")
    avg_pred_scaled = np.mean(all_predictions, axis=1)
    
    # Inverse Transform
    dummy = np.zeros((len(avg_pred_scaled), len(features)))
    dummy[:, target_idx] = avg_pred_scaled
    pred_returns = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Reconstruct Prices
    # Price_t = Price_{t-1} * exp(Return_t / 10)
    test_indices = df.index[split + SEQ_LENGTH:]
    prev_prices = df['Gold_Price'].iloc[split + SEQ_LENGTH - 1 : -1].values
    actual_prices = df['Gold_Price'].iloc[split + SEQ_LENGTH:].values
    
    final_predicted_prices = prev_prices * np.exp(pred_returns / 10)
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(actual_prices, final_predicted_prices))
    
    print("\n" + "="*40)
    print("FINAL ENSEMBLE RESULTS")
    print("="*40)
    print(f"Ensemble RMSE: {rmse:.4f}")
    print(f"Baseline RMSE: {BASELINE_RMSE:.4f}")
    
    if rmse < BASELINE_RMSE:
        print(f"🏆 VICTORY! Improvement: {BASELINE_RMSE - rmse:.4f} points")
    else:
        print("Still fighting! The Random Walk is very strong.")
        
    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(test_indices, actual_prices, label='Actual Price', color='gray', alpha=0.5)
    plt.plot(test_indices, final_predicted_prices, label='Ensemble Forecast', color='green', linewidth=1.5)
    plt.title(f'Ensemble (Bagging) Strategy | RMSE: {rmse:.4f}')
    plt.legend()
    plt.savefig('ensemble_victory.png')
    print("Saved ensemble_victory.png")

if __name__ == "__main__":
    main()