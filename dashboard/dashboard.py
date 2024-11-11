import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from prophet import Prophet
import pickle
from xgboost import XGBRegressor

# Load true test labels and LSTM test features
y_lstm_test = np.load('data/processed/lstm_test_labels.npy')
X_lstm_test = np.load('data/processed/lstm_test_features.npy')

# Load LSTM model and make predictions
lstm_model = load_model('models/lstm_model.keras')
scaler = pickle.load(open('data/processed/scaler.pkl', 'rb'))
lstm_predictions_scaled = lstm_model.predict(X_lstm_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled).flatten()

# Reverse transform the true test values
test_y_aligned = scaler.inverse_transform(y_lstm_test.reshape(-1, 1)).flatten()

# Load XGBoost model and make predictions
xgb_model = XGBRegressor()
xgb_model.load_model('models/xgboost_model.json')
X_xgb_test = pd.read_csv('data/processed/xgboost_test_features.csv')
xgb_predictions = xgb_model.predict(X_xgb_test)

# Load Prophet predictions
prophet_predictions = pd.read_csv('data/forecast/prophet_forecast.csv')['yhat']

# Ensure Prophet predictions match the length of test_y_aligned
if len(prophet_predictions) > len(test_y_aligned):
    prophet_predictions = prophet_predictions[:len(test_y_aligned)]
elif len(prophet_predictions) < len(test_y_aligned):
    prophet_predictions = np.pad(prophet_predictions, (0, len(test_y_aligned) - len(prophet_predictions)), mode='edge')

# Dashboard title
st.title("Time Series Forecasting Dashboard")

# Introduction
st.markdown("""
This dashboard provides an interactive way to explore the performance of different forecasting models.
You can view the true values alongside the predictions made by Prophet, LSTM, and XGBoost.
""")

# Model selection
model_to_display = st.selectbox("Choose a model to display:", ["Prophet", "LSTM", "XGBoost"])

# Display corresponding graph
if model_to_display == "Prophet":
    st.line_chart(pd.concat([pd.Series(test_y_aligned, name='True Values'),
                             pd.Series(prophet_predictions[:len(test_y_aligned)], name='Prophet Predictions')], axis=1))
elif model_to_display == "LSTM":
    st.line_chart(pd.concat([pd.Series(test_y_aligned, name='True Values'),
                             pd.Series(lstm_predictions, name='LSTM Predictions')], axis=1))
elif model_to_display == "XGBoost":
    st.line_chart(pd.concat([pd.Series(test_y_aligned, name='True Values'),
                             pd.Series(xgb_predictions[:len(test_y_aligned)], name='XGBoost Predictions')], axis=1))

# Metrics calculation
st.header("Model Metrics")

# Calculate evaluation metrics dynamically
models = ['Prophet', 'LSTM', 'XGBoost']
predictions = [
    prophet_predictions[:len(test_y_aligned)],
    lstm_predictions,
    xgb_predictions[:len(test_y_aligned)]
]

mae_scores = [np.mean(np.abs(test_y_aligned - pred)) for pred in predictions]
rmse_scores = [np.sqrt(np.mean((test_y_aligned - pred) ** 2)) for pred in predictions]
mape_scores = [np.mean(np.abs((test_y_aligned - pred) / test_y_aligned)) * 100 for pred in predictions]

# Create a DataFrame for metrics
metrics_data = pd.DataFrame({
    "Model": models,
    "MAE": mae_scores,
    "RMSE": rmse_scores,
    "MAPE": mape_scores
})

st.dataframe(metrics_data)

# Future Predictions
st.header("Future Predictions")

# User input for number of days to predict
days_to_predict = st.slider("Select number of days to predict:", min_value=1, max_value=30, value=7)

# Prophet Future Predictions
st.subheader("Prophet Future Predictions")
prophet_model = Prophet()
data = pd.read_csv('data/processed/processed_sales_data.csv')[['ds', 'y']]
prophet_model.fit(data.rename(columns={'ds': 'ds', 'y': 'y'}))
future = prophet_model.make_future_dataframe(periods=days_to_predict)
forecast = prophet_model.predict(future)
future_forecast = forecast[['ds', 'yhat']].tail(days_to_predict)
st.write(future_forecast)
st.line_chart(future_forecast.set_index('ds'))

# LSTM Future Predictions
st.subheader("LSTM Future Predictions")
last_sequence = X_lstm_test[-1].reshape(1, -1, 1) 
lstm_future_predictions = []
for _ in range(days_to_predict):
    prediction = lstm_model.predict(last_sequence)
    lstm_future_predictions.append(scaler.inverse_transform(prediction).flatten()[0])
    prediction = prediction.reshape((1, 1, 1))
    last_sequence = np.append(last_sequence[:, 1:, :], prediction, axis=1)  

future_dates_lstm = pd.date_range(start=pd.to_datetime(data['ds'].iloc[-1]), periods=days_to_predict + 1, freq='D')[1:]
lstm_future_forecast = pd.DataFrame({'Date': future_dates_lstm, 'LSTM Predictions': lstm_future_predictions})
st.write(lstm_future_forecast)
st.line_chart(lstm_future_forecast.set_index('Date'))

# XGBoost Future Predictions
st.subheader("XGBoost Future Predictions")
last_xgb_features = X_xgb_test.iloc[-1].copy()
xgb_future_predictions = []

for _ in range(days_to_predict):
    prediction = xgb_model.predict(last_xgb_features.values.reshape(1, -1))[0]
    xgb_future_predictions.append(prediction)
    for lag in [1, 7, 14]:  
        last_xgb_features[f'lag_{lag}'] = prediction

future_dates_xgb = pd.date_range(start=pd.to_datetime(data['ds'].iloc[-1]), periods=days_to_predict + 1, freq='D')[1:]
xgb_future_forecast = pd.DataFrame({'Date': future_dates_xgb, 'XGBoost Predictions': xgb_future_predictions})
st.write(xgb_future_forecast)
st.line_chart(xgb_future_forecast.set_index('Date'))
