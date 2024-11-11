import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load preprocessed data
data = pd.read_csv('data/processed/processed_sales_data.csv')

# Feature engineering: Create lag features
def create_lag_features(data, lags):
    for lag in lags:
        data[f'lag_{lag}'] = data['y'].shift(lag)
    return data

# Create lag features (e.g., 1-day, 7-day, and 14-day lag)
lags = [1, 7, 14]
data = create_lag_features(data, lags)

# Drop rows with NaN values (introduced by lagging)
data = data.dropna()

# Add day of week and month as features
data['DayOfWeek'] = pd.to_datetime(data['ds']).dt.dayofweek
data['Month'] = pd.to_datetime(data['ds']).dt.month

# Split data into features (X) and target (y)
X = data.drop(columns=['ds', 'y'])
y = data['y']

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Save test features and labels for comparison
X_test.to_csv('data/processed/xgboost_test_features.csv', index=False)
y_test.to_csv('data/processed/xgboost_test_labels.csv', index=False)

# Initialize and train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test set
xgb_predictions = model.predict(X_test)

# Evaluate model performance
rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot true values vs predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='True Values', color='blue')
plt.plot(xgb_predictions, label='Predictions', color='red')
plt.title('XGBoost Model: True Sales vs Predicted Sales')
plt.legend()
plt.show()

# Save the model
model.save_model('models/xgboost_model.json')
