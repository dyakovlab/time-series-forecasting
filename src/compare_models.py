import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor

# Load LSTM test labels (actual test values for comparison)
y_lstm_test = np.load('data/processed/lstm_test_labels.npy') 

# Load LSTM test features and make predictions
X_lstm_test = np.load('data/processed/lstm_test_features.npy')
lstm_model = load_model('models/lstm_model.keras')

# Load the scaler used during preprocessing to reverse scaling
scaler = pickle.load(open('data/processed/scaler.pkl', 'rb'))

# Make predictions with LSTM model and inverse transform
lstm_predictions_scaled = lstm_model.predict(X_lstm_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled).flatten()

# Load XGBoost model and make predictions
xgb_model = XGBRegressor()
xgb_model.load_model('models/xgboost_model.json')
X_xgb_test = pd.read_csv('data/processed/xgboost_test_features.csv')
xgb_predictions = xgb_model.predict(X_xgb_test)

# Load Prophet predictions
prophet_forecast = pd.read_csv('data/forecast/prophet_forecast.csv')
prophet_predictions = prophet_forecast['yhat'].values

# If Prophet forecast is shorter or longer, adjust it to match test length
if len(prophet_predictions) > len(y_lstm_test):
    print(f"Truncating Prophet predictions from {len(prophet_predictions)} to {len(y_lstm_test)}")
    prophet_predictions = prophet_predictions[-len(y_lstm_test):]
elif len(prophet_predictions) < len(y_lstm_test):
    print(f"Extending Prophet predictions from {len(prophet_predictions)} to {len(y_lstm_test)}")
    prophet_predictions = np.pad(prophet_predictions, (0, len(y_lstm_test) - len(prophet_predictions)), mode='edge')

# Ensure XGBoost predictions match LSTM test length
if len(xgb_predictions) > len(y_lstm_test):
    print(f"Truncating XGBoost predictions from {len(xgb_predictions)} to {len(y_lstm_test)}")
    xgb_predictions = xgb_predictions[:len(y_lstm_test)]
elif len(xgb_predictions) < len(y_lstm_test):
    print(f"Warning: XGBoost predictions are shorter than expected ({len(xgb_predictions)} vs {len(y_lstm_test)})")

# Verify that all predictions match the test label length
assert len(lstm_predictions) == len(y_lstm_test), f"LSTM predictions length mismatch: {len(lstm_predictions)} vs {len(y_lstm_test)}"
assert len(prophet_predictions) == len(y_lstm_test), f"Prophet predictions length mismatch: {len(prophet_predictions)} vs {len(y_lstm_test)}"
assert len(xgb_predictions) == len(y_lstm_test), f"XGBoost predictions length mismatch: {len(xgb_predictions)} vs {len(y_lstm_test)}"

# Reverse transform the true values (y_lstm_test) if necessary
test_y_aligned = scaler.inverse_transform(y_lstm_test.reshape(-1, 1)).flatten()

# Calculate metrics for each model
models = ['Prophet', 'LSTM', 'XGBoost']
predictions = [prophet_predictions, lstm_predictions, xgb_predictions]

# Compute error metrics: MAE, RMSE, MAPE
mae_scores = [mean_absolute_error(test_y_aligned, pred) for pred in predictions]
rmse_scores = [np.sqrt(mean_squared_error(test_y_aligned, pred)) for pred in predictions]
mape_scores = [np.mean(np.abs((test_y_aligned - pred) / test_y_aligned)) * 100 for pred in predictions]

# Print evaluation metrics for each model
print("Model Comparison:")
for i, model in enumerate(models):
    print(f"\n{model}:")
    print(f"  MAE:  {mae_scores[i]:.2f}")
    print(f"  RMSE: {rmse_scores[i]:.2f}")
    print(f"  MAPE: {mape_scores[i]:.2f}%")

# Plot the true values and predictions for each model
plt.figure(figsize=(14, 8))
plt.plot(test_y_aligned, label='True Values', color='blue', linestyle='dashed')
for i, pred in enumerate(predictions):
    plt.plot(pred, label=f'{models[i]} Predictions')
plt.title('Comparison of Predictions vs True Values')
plt.legend()
plt.show()

# Display a sample of true values and predictions
print("\nSample of true values (test_y_aligned):", test_y_aligned[:10])
print("Sample of Prophet predictions:", prophet_predictions[:10])
print("Sample of LSTM predictions:", lstm_predictions[:10])
print("Sample of XGBoost predictions:", xgb_predictions[:10])

# Display summaries of test and Prophet forecast datasets
print("\nTest data summary:")
print(f"Length of y_lstm_test: {len(y_lstm_test)}")

print("\nProphet forecast summary:")
print(prophet_forecast.head())
