import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data', 'model_ready_dataset.csv')
SEQ_LENGTH = 60  # Look back 60 days to predict tomorrow
TEST_SPLIT = 0.2 # 20% of data for testing

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

def fit_garch_volatility(df):
    """
    Step 1: GARCH Model
    Calculates the 'Conditional Volatility' of Gold Returns.
    This captures 'market fear' and clustering volatility.
    """
    print("Fitting GARCH(1,1) Model to capture volatility...")
    
    # Calculate Log Returns (GARCH requires returns, not prices)
    # We multiply by 100 to help the GARCH optimizer converge faster
    returns = 100 * df['Gold_Price'].pct_change().dropna()
    
    # Fit GARCH(1,1)
    model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
    res = model.fit(disp='off')
    
    # Extract Volatility and align it with the original dataframe
    # We pad the first NaN (day 0) with the first valid volatility to keep lengths equal
    volatility = res.conditional_volatility
    df = df.iloc[1:].copy() # Drop the first row (NaN return)
    df['GARCH_Volatility'] = volatility
    
    print(f"GARCH Model Fitted. Current Volatility Factor: {volatility.iloc[-1]:.4f}")
    return df

def create_sequences(data, seq_length):
    """
    Converts time series data into (X, y) sequences for LSTM.
    X: Data from (t - seq_length) to (t-1)
    y: Price at (t)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0]) # Index 0 is Gold_Price (Target)
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Step 2: LSTM Architecture
    """
    model = Sequential([
        # First LSTM Layer
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2), # Prevents overfitting
        
        # Second LSTM Layer
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        
        # Output Layer (Predicting 1 value: Price)
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # 1. Load Data
    df = load_data()
    print(f"Loaded Data: {df.shape}")

    # 2. Apply GARCH
    df = fit_garch_volatility(df)
    
    # 3. Scale Data (Neural Networks require 0-1 scaling)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # 4. Create Sequences
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    # 5. Split Train/Test
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training Shape: {X_train.shape}")
    print(f"Testing Shape:  {X_test.shape}")

    # 6. Train LSTM
    print("\nTraining LSTM Model...")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Training for 20 epochs (iterations)
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 7. Predictions
    predictions_scaled = model.predict(X_test)
    
    # Inverse Transform to get real USD prices back
    # We need to create a dummy array to inverse transform because scaler expects all columns
    dummy_array = np.zeros((len(predictions_scaled), scaled_data.shape[1]))
    dummy_array[:, 0] = predictions_scaled.flatten() # Fill 0th column (Price)
    predictions_usd = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Do the same for y_test (Actuals)
    dummy_array_y = np.zeros((len(y_test), scaled_data.shape[1]))
    dummy_array_y[:, 0] = y_test
    actuals_usd = scaler.inverse_transform(dummy_array_y)[:, 0]

    # 8. Evaluate
    rmse = np.sqrt(mean_squared_error(actuals_usd, predictions_usd))
    mae = mean_absolute_error(actuals_usd, predictions_usd)
    
    print("-" * 30)
    print("HYBRID GARCH-LSTM RESULTS")
    print("-" * 30)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print("-" * 30)

    # 9. Visualize
    plt.figure(figsize=(14, 7))
    
    # Get dates for the test period
    test_dates = df.index[split_idx + SEQ_LENGTH:]
    
    plt.plot(test_dates, actuals_usd, color='blue', label='Actual Gold Price')
    plt.plot(test_dates, predictions_usd, color='red', label='GARCH-LSTM Forecast', linestyle='--')
    
    plt.title('Hybrid GARCH-LSTM Gold Price Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    output_img = 'final_forecast.png'
    plt.savefig(output_img)
    print(f"Forecast plot saved to {output_img}")
    
    # Save Model Loss Curve (to check for overfitting)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.legend()
    plt.savefig('training_loss.png')

if __name__ == "__main__":
    main()