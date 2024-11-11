import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import pickle

# Load preprocessed data
data = pd.read_csv('data/processed/processed_sales_data.csv')

# Aggregate data by date ('ds') to prepare it for Prophet (sum sales 'y' for each day)
aggregated_data = data.groupby('ds').agg({'y': 'mean'}).reset_index()

# Sort data by date to ensure correct temporal order
aggregated_data = aggregated_data.sort_values('ds')

# Split data into train and test sets
split_ratio = 0.8
split_index = int(len(aggregated_data) * split_ratio)
train = aggregated_data.iloc[:split_index]
test = aggregated_data.iloc[split_index:]

# Initialize Prophet model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

# Fit the model on the training data
model.fit(train)

# Create a dataframe for future dates with the same length as the test set
future = model.make_future_dataframe(periods=len(test), freq='D')

# Generate forecast for the future dates
forecast = model.predict(future)

# Extract the forecast only for the test set period
prophet_forecast = forecast[['ds', 'yhat']].iloc[-len(test):].reset_index(drop=True)

# Save the forecast to a CSV file
prophet_forecast.to_csv('data/forecast/prophet_forecast.csv', index=False)

# Plot the complete forecast
fig = model.plot(forecast)
plt.title('Prophet Forecast with Aggregated Data')
plt.show()

# Plot the forecast components (trend, weekly seasonality)
fig2 = model.plot_components(forecast)
plt.show()

# Save the trained Prophet model using pickle for future use
with open('models/prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)
