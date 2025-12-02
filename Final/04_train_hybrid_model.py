import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data', 'model_ready_dataset.csv')
SEQ_LENGTH = 60
TEST_SPLIT = 0.2

def load_and_engineer_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # 2. Target Engineering: Log Returns (The Secret Weapon)
    # Instead of predicting $1200, we predict +0.05% or -0.02%
    # We multiply by 100 to make the numbers easier for the neural net to learn
    df['Log_Ret'] = np.log(df['Gold_Price'] / df['Gold_Price'].shift(1)) * 100
    
    # 3. Technical Indicators (Give the model "Trader Eyes")
    # SMA (Trend)
    df['SMA_14'] = df['Gold_Price'].rolling(window=14).mean()
    
    # Momentum (Price Rate of Change)
    df['ROC_5'] = df['Gold_Price'].pct_change(periods=5)
    
    # Volatility (Rolling Standard Deviation)
    df['Roll_Vol_20'] = df['Gold_Price'].rolling(window=20).std()

    # 4. GARCH Volatility Feature
    # We model the volatility of returns to give the LSTM a "Fear Gauge"
    returns_clean = df['Log_Ret'].dropna()
    model = arch_model(returns_clean, vol='Garch', p=1, q=1, dist='Normal')
    res = model.fit(disp='off')
    df.loc[returns_clean.index, 'GARCH_Vol'] = res.conditional_volatility

    # Drop NaNs created by lagging/rolling
    df.dropna(inplace=True)
    
    print(f"Data Engineered. Features: {len(df.columns)}")
    return df

def create_sequences(data, target_col_idx, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_col_idx]) 
    return np.array(X), np.array(y)

def build_alpha_model(input_shape):
    """
    A deeper, more aggressive LSTM architecture
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # Layer 1: Capture broad trends
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        
        # Layer 2: Capture fine details
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        # Layer 3: Decision making
        Dense(32, activation='relu'),
        
        # Output: Predicted Return
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    print("--- Training Alpha Model (Target: Log Returns) ---")
    
    # 1. Load & Engineer
    df = load_and_engineer_data()
    
    # We need to know where 'Log_Ret' and 'Gold_Price' are for reconstruction later
    target_col_idx = df.columns.get_loc('Log_Ret')
    
    # 2. Scale features (0 to 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # 3. Create Sequences
    X, y = create_sequences(scaled_data, target_col_idx, SEQ_LENGTH)
    
    # 4. Split Train/Test
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 5. Train
    model = build_alpha_model((X_train.shape[1], X_train.shape[2]))
    
    # Callbacks to prevent overfitting and improve learning
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    history = model.fit(
        X_train, y_train,
        epochs=100, # High epochs, let early_stop decide when to quit
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # 6. Predict Returns (Scaled)
    pred_scaled = model.predict(X_test)
    
    # 7. Inverse Transform Predictions
    # We need a dummy array because scaler expects all columns
    dummy_pred = np.zeros((len(pred_scaled), scaled_data.shape[1]))
    dummy_pred[:, target_col_idx] = pred_scaled.flatten()
    pred_returns = scaler.inverse_transform(dummy_pred)[:, target_col_idx]
    
    # Inverse Transform Actuals (for verification)
    dummy_act = np.zeros((len(y_test), scaled_data.shape[1]))
    dummy_act[:, target_col_idx] = y_test
    actual_returns = scaler.inverse_transform(dummy_act)[:, target_col_idx]
    
    # 8. RECONSTRUCT PRICES (The most important step)
    # Formula: Price_t = Price_{t-1} * exp(Return_t / 100)
    
    # Get the prices from the day BEFORE each prediction
    # Test set starts at `split`. We need prices from `split + SEQ_LENGTH - 1`
    test_indices = df.index[split + SEQ_LENGTH:]
    prev_prices = df['Gold_Price'].iloc[split + SEQ_LENGTH - 1 : -1].values
    
    # Calculate Predicted Prices
    # Note: We divide returns by 100 because we multiplied them earlier
    predicted_prices = prev_prices * np.exp(pred_returns / 100)
    
    # Get Actual Prices
    actual_prices = df['Gold_Price'].iloc[split + SEQ_LENGTH:].values
    
    # 9. Evaluate
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    
    print("\n" + "="*40)
    print("ALPHA MODEL RESULTS (Price Reconstruction)")
    print("="*40)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    
    # 10. Plot
    plt.figure(figsize=(14, 7))
    plt.plot(test_indices, actual_prices, label='Actual Price', color='blue', alpha=0.6)
    plt.plot(test_indices, predicted_prices, label='Alpha LSTM Forecast', color='green', linestyle='--', alpha=0.8)
    plt.title(f'Alpha Strategy Forecast (RMSE: {rmse:.2f})')
    plt.legend()
    plt.grid(True)
    plt.savefig('alpha_forecast.png')
    print("Saved alpha_forecast.png")

if __name__ == "__main__":
    main()