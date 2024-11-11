import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load preprocessed sales data
data = pd.read_csv('data/processed/processed_sales_data.csv')

# Sort data by date to ensure proper ordering
data = data.sort_values(by='ds')

# Normalize the 'y' column using MinMaxScaler (same as during training)
scaler = MinMaxScaler(feature_range=(0, 1))
data['y_scaled'] = scaler.fit_transform(data['y'].values.reshape(-1, 1))

# Split the data into training and test sets
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
test_data = data.iloc[split_index:]  

# Extract test labels (true values)
test_y_aligned = test_data['y'].values  

# Save test_y_aligned to CSV for future use
pd.DataFrame({'y': test_y_aligned}).to_csv('data/processed/test_y_aligned.csv', index=False)
