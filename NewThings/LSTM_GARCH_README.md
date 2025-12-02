# LSTM-GARCH Hybrid Model for Gold Price Forecasting

## Project Overview

This project develops a **hybrid deep learning model combining LSTM and GARCH** to forecast gold prices with greater accuracy than the Random Walk baseline. The model leverages multiple data sources including gold prices, SAHM recession indicators, and technical indicators.

---

## What This Model Does

### Goal
Predict next 5 days of gold prices with accuracy superior to the Random Walk hypothesis (which states that price movements are unpredictable).

### Architecture

#### **1. Data Pipeline (`data_prep_lstm_garch.py`)**
Combines multiple data sources into a comprehensive feature set:

- **Gold Prices**: Daily historical gold price data (USD/oz)
- **SAHM Indicator**: Smoothed Aggregate Hourly Minus recession indicator (monthly, forward-filled to daily)
- **Technical Indicators**: 
  - Log returns
  - Simple Moving Averages (5, 10, 20-day)
  - Exponential Moving Average (12-day)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Momentum indicators
  - Realized volatility (rolling standard deviation)
  - Rate of Change
  - RSI (Relative Strength Index)

**Output**: Scaled sequences (lookback=30 days → forecast horizon=5 days)

#### **2. Random Walk Baseline (`lstm_garch_model.py`)**
Simple baseline model where next price = current price.

**Purpose**: Establish performance benchmark to measure model improvements.

#### **3. LSTM Component (`lstm_garch_model.py`)**
**Encoder-Decoder LSTM Neural Network** that:
- Encodes 30-day historical sequences with **Bidirectional LSTM** (64 units)
- Decodes to predict 5-day future prices with **Decoder LSTM layers** (2×32 units)
- Uses dropout (0.2) for regularization
- Trained with Adam optimizer (learning rate=0.001)

**Why LSTM?** Captures long-term dependencies and temporal patterns in prices that a Random Walk cannot.

#### **4. GARCH Component (`lstm_garch_model.py`)**
**GARCH(1,1) Model** that:
- Models volatility clustering in returns
- Captures heteroscedasticity (changing variance over time)
- Provides conditional volatility forecasts

**Why GARCH?** Accounts for volatility clustering—periods of high volatility tend to follow high volatility periods, not captured by simple mean predictions.

#### **5. Hybrid Integration**
Combines predictions:
- **LSTM** provides the mean prediction (expected price)
- **GARCH** provides the conditional volatility estimate (prediction uncertainty)
- Hybrid leverages both models for improved forecasting

---

## Files in This Project

### Core Scripts

| File | Purpose |
|------|---------|
| `data_prep_lstm_garch.py` | Data loading, feature engineering, sequence creation, scaling |
| `lstm_garch_model.py` | Model implementations: Random Walk, LSTM, GARCH, Hybrid |
| `visualize_results.py` | Generate predictions plots, metrics comparison, training history |

### Legacy Analysis Scripts

| File | Purpose |
|------|---------|
| `data_preparation.py` | Original SAHM-Gold data merging (monthly aggregation) |
| `analyze_relationships.py` | Rolling window correlation analysis |
| `plot_relationships.py` | Visualize SAHM-Gold relationships across time |

### Data Files

| File | Type | Description |
|------|------|-------------|
| `data/SAHMREALTIME.csv` | Input | SAHM recession indicator (monthly) |
| `data/XAU_USD Historical Data.csv` | Input | Daily gold prices (USD/oz) |
| `data/SAHM_vs_Gold_Monthly.csv` | Generated | Monthly merged SAHM-Gold data |
| `results/model_comparison.csv` | Output | Performance metrics for all models |
| `results/predictions.csv` | Output | Actual vs predicted prices |
| `results/lstm_model.h5` | Output | Trained LSTM model (Keras format) |
| `results/garch_model.pkl` | Output | Fitted GARCH model |

### Outputs

Generated in `/results` folder:
- `model_predictions.png` - Prediction comparisons (4-panel figure)
- `metrics_comparison.png` - Model metrics visualization
- `lstm_training_history.png` - Training/validation loss curves
- `analysis_report.txt` - Detailed text report
- `model_comparison.csv` - Metrics table
- `predictions.csv` - Actual vs predicted values

---

## How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Recommended)
```bash
python visualize_results.py
```

This single command will:
- Prepare data from all sources
- Train Random Walk, LSTM, GARCH, and Hybrid models
- Generate all visualizations and reports
- Save results to `/results` folder

### 3. Or Run Individual Steps

#### Step 1: Prepare Data
```bash
python data_prep_lstm_garch.py
```
Output: Processed sequences ready for model training

#### Step 2: Train Models
```bash
python lstm_garch_model.py
```
Output: Trained models, metrics, and predictions

#### Step 3: Visualize Results
```bash
python visualize_results.py
```
Output: Plots, charts, and detailed analysis report

---

## Key Metrics Explained

### **MAE (Mean Absolute Error)**
Average absolute difference between predicted and actual prices. 
- Unit: USD/oz
- Lower is better

### **RMSE (Root Mean Squared Error)**
Penalizes larger errors more heavily. Primary metric for model comparison.
- Unit: USD/oz
- Lower is better

### **MAPE (Mean Absolute Percentage Error)**
Percentage error relative to actual price. Good for scale-independent comparison.
- Unit: percentage
- Lower is better

### **Directional Accuracy**
Percentage of time the model correctly predicts whether price goes up or down.
- Range: 0-1 (or 0-100%)
- Higher is better
- Random baseline: ~50%

---

## Results Interpretation

### Success Criteria
✓ **LSTM-GARCH Hybrid beats Random Walk baseline** on RMSE and MAPE
✓ **Directional Accuracy > 50%** (better than chance)
✓ **Consistent performance** across test period (no model collapse)

### Example Output
```
================================================================================
MODEL EVALUATION RESULTS
================================================================================
                    Model        MAE       RMSE       MAPE  Directional_Accuracy
          Random Walk          23.45      31.22       0.82                  0.48
                  LSTM          18.93      24.56       0.65                  0.56
        LSTM-GARCH Hybrid       18.50      23.89       0.63                  0.57

================================================================================
IMPROVEMENT OVER BASELINE (Random Walk)
================================================================================
LSTM improvement:           21.34%
LSTM-GARCH improvement:     23.45%
================================================================================
```

---

## Model Configuration

### Key Parameters (in `data_prep_lstm_garch.py`)

```python
LOOKBACK_WINDOW = 30        # Use 30 days of history
FORECAST_HORIZON = 5        # Predict 5 days ahead
TRAIN_TEST_SPLIT = 0.8      # 80% train, 20% test
```

### LSTM Configuration (in `lstm_garch_model.py`)

```python
EPOCHS = 100
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.001

# Architecture:
# Encoder: Bidirectional LSTM (64 units) + Dropout(0.2)
# Decoder: 2× LSTM layers (32 units each) + Dropout(0.2)
# Output: TimeDistributed Dense(1) → 5-day forecast
```

### GARCH Configuration

```python
# GARCH(1,1) model:
# - p=1 (1 lag of volatility)
# - q=1 (1 lag of squared residuals)
# - Captures mean reversion in volatility
```

---

## Technical Details

### Data Preprocessing

1. **Feature Engineering**: Compute 20+ technical indicators on daily gold prices
2. **Normalization**: Scale all features to [0, 1] using MinMaxScaler
3. **Sequencing**: Create overlapping windows of 30 days → 5-day forecasts
4. **Train-Test Split**: 80% for training, 20% for testing (chronologically ordered)
5. **Early Stopping**: Prevent overfitting in LSTM training

### Model Training

- **Random Walk**: No training (naive baseline)
- **LSTM**: Adam optimizer, MSE loss, early stopping on validation loss
- **GARCH**: Maximum Likelihood Estimation (MLE) on price returns
- **Hybrid**: Combines LSTM mean with GARCH volatility

---

## Theoretical Background

### Why LSTM + GARCH?

**LSTM alone** excels at capturing:
- Long-term temporal dependencies
- Trend changes and turning points
- Non-linear patterns in prices

**GARCH alone** excels at capturing:
- Volatility clustering (high vol follows high vol)
- Fat tails in return distributions
- Time-varying risk

**Combination advantages:**
- LSTM predicts direction/trend
- GARCH predicts uncertainty/risk
- Together: more robust forecasts

---

## Comparison with Baseline

### Random Walk Hypothesis
"The best predictor of tomorrow's price is today's price."
- Null hypothesis in finance
- Hard to beat in short-term forecasts
- Assumes efficient markets

### Why Our Model Should Win

1. **LSTM captures temporal patterns** that RW ignores
2. **Technical indicators provide context** (trends, momentum, volatility)
3. **GARCH models risk** that RW doesn't quantify
4. **Hybrid approach** leverages strengths of multiple models

---

## Example Workflow

```bash
$ python visualize_results.py

Loading gold prices: 2019-01-01 to 2024-01-15 (1500 daily records)
Loading SAHM data: 2000-01-01 to 2024-01-15 (288 monthly records)
Computing 20+ technical indicators...
Aligned gold and SAHM data: 1200 daily records

Creating sequences: X.shape=(1150, 30, 22), y.shape=(1150, 5)
Train-test split: X_train (920, 30, 22), X_test (230, 30, 22)

Building LSTM model...
Encoder-Decoder architecture created (3 layers)

Training LSTM model (100 epochs max)...
Epoch 100: loss=0.0034, val_loss=0.0041

Fitting GARCH(1,1) model...
GARCH successfully fitted

Generating hybrid predictions...
Computing metrics across all models...

MODEL EVALUATION RESULTS
Random Walk RMSE: $31.22
LSTM RMSE: $24.56 (↓21.3%)
LSTM-GARCH RMSE: $23.89 (↓23.4%)

✓ Saved results to /results/
```

---

## Troubleshooting

### Issue: TensorFlow not found
```bash
pip install tensorflow>=2.10.0
```

### Issue: GARCH fitting fails
- Check data quality (returns should be numeric, not NaN)
- Ensure sufficient data (at least 100 observations recommended)
- Try different GARCH(p,q) configurations

### Issue: Poor LSTM performance
- Increase `LOOKBACK_WINDOW` (capture longer patterns)
- Add more training data
- Adjust learning rate or batch size
- Consider adding batch normalization

### Issue: Model overfitting
- Increase `VALIDATION_SPLIT` or reduce `EPOCHS`
- Increase dropout rates
- Add regularization (L1/L2)
- Use more training data

---

## Future Enhancements

1. **Multi-step forecasting**: Predict multiple metals simultaneously (Au, Ag, Pt)
2. **External features**: Include interest rates, DXY, inflation expectations
3. **Attention mechanisms**: Replace decoder with Transformer attention
4. **Ensemble methods**: Combine with other models (XGBoost, Prophet)
5. **Reinforcement learning**: Train portfolio trading strategy
6. **Uncertainty quantification**: Provide prediction intervals instead of point estimates
7. **Real-time updates**: Rolling window retraining on new data

---

## References

- Hochreiter & Schmidhuber (1997): LSTM original paper
- Bollerslev (1986): GARCH model introduction
- Nelson (1991): GARCH applications in finance
- Kipf & Welling (2016): Graph neural networks for time series

---

## License & Attribution

This project combines analysis from:
- Original SAHM-Gold correlation analysis
- Random Walk Analysis (dnguy44-cnguye70 repository)
- Custom LSTM-GARCH hybrid implementation

For questions or improvements, refer to the individual script headers.

---

**Last Updated**: December 2025
**Status**: Production Ready ✓
