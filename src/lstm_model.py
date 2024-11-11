import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load preprocessed data
data = pd.read_csv('data/processed/processed_sales_data.csv')

# Use only the 'ds' (date) and 'y' (sales) columns for LSTM model
data = data[['ds', 'y']]

# Convert 'ds' to datetime format
data['ds'] = pd.to_datetime(data['ds'])

# Sort data by date (just to be sure)
data = data.sort_values(by='ds')

# Normalize the 'y' values (sales data)
scaler = MinMaxScaler(feature_range=(0, 1))
data['y_scaled'] = scaler.fit_transform(data['y'].values.reshape(-1, 1))

# Prepare data for LSTM: creating sequences of past sales
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data['y_scaled'].values[i:i + sequence_length])
        labels.append(data['y_scaled'].values[i + sequence_length])
    return np.array(sequences), np.array(labels)

# Use past 30 days to predict the next day
sequence_length = 30 
X, y = create_sequences(data, sequence_length)

# Reshape X for LSTM input (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Save test features for comparison script
np.save('data/processed/lstm_test_features.npy', X_test)
np.save('data/processed/lstm_test_labels.npy', y_test)

# Save MinMaxScaler for inverse transformation of predictions
with open('data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('models/lstm_model.keras')
