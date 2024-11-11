import pandas as pd

# Load sales and wholesale data
sales_data = pd.read_csv('data/raw/annex2.csv')
wholesale_data = pd.read_csv('data/raw/annex3.csv')

# Convert 'Date' column to datetime format
sales_data['Date'] = pd.to_datetime(sales_data['Date'])

# Filter out returns, keeping only sales
sales_data = sales_data[sales_data['Sale or Return'] == 'sale']

# Convert 'Discount (Yes/No)' column to binary (1 for Yes, 0 for No)
sales_data['Discount'] = sales_data['Discount (Yes/No)'].apply(lambda x: 1 if x == 'Yes' else 0)

# Aggregate sales data by date and item code
daily_sales = sales_data.groupby(['Date', 'Item Code']).agg({
    'Quantity Sold (kilo)': 'sum',  
    'Unit Selling Price (RMB/kg)': 'mean',  
    'Discount': 'max'  
}).reset_index()

# Convert 'Date' column in wholesale data to datetime format
wholesale_data['Date'] = pd.to_datetime(wholesale_data['Date'])

# Merge sales data with wholesale prices
data = pd.merge(daily_sales, wholesale_data, on=['Date', 'Item Code'], how='left')

# Create new features for day of the week and month
data['DayOfWeek'] = data['Date'].dt.dayofweek  
data['Month'] = data['Date'].dt.month  

# Rename columns to match model input requirements
data = data.rename(columns={
    'Date': 'ds', 
    'Quantity Sold (kilo)': 'y', 
    'Unit Selling Price (RMB/kg)': 'UnitPrice',
    'Wholesale Price (RMB/kg)': 'WholesalePrice'
})

# Save preprocessed data to a CSV file
data.to_csv('data/processed/processed_sales_data.csv', index=False)

# Print the first few rows of the final dataset
print(data.head())
