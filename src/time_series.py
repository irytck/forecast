#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:24:03 2024

@author: user
"""

"""
Created on Wed Oct 23 12:40:39 2024

@author: user
"""
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')


# statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# skforecast
import skforecast
from skforecast.datasets import fetch_dataset
from skforecast.plot import set_dark_theme
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from skforecast.model_selection_sarimax import grid_search_sarimax

import warnings
warnings.filterwarnings("ignore")

# Set pandas options to display all columns
pd.set_option('display.max_columns', None)
set_dark_theme()

# Load Data
data = pd.read_excel("/Users/user/projects/forecast/data/energy_prices_raw.xlsx")

# Filter for Spain and USD, then pivot by PRODUCT
pivoted_data = (
    data[
        (data['COUNTRY'] == 'Spain') &
        (data['UNIT'] == 'USD')
    ]
    .pivot_table(index='TIME', columns='PRODUCT', values='VALUE')
)

pivoted_data.columns.name = None  # Elimina el nombre del índice de columnas

# Set TIME as index and sort it
df = pivoted_data.asfreq('MS').sort_index()

# Missing values
print(f'Number of rows with missing values: {df.isnull().any(axis=1).mean()}')

# Verify that a temporary index is complete
start_date = df.index.min()
end_date = df.index.max()
date_range = pd.date_range(start=start_date, end=end_date, freq=df.index.freq)
is_index_complete = (df.index == date_range).all()
print(f"Index complete: {is_index_complete}")

# Convert Light Fuel Oil prices from per 1000 litres to per litre
df['Light fuel oil (unit/Litre)'] = df['Light fuel oil (unit/1000 litres)'] / 1000

# Step 1: EDA

fig, ax = plt.subplots(figsize=(10, 6))
df['Gasoline (unit/Litre)'].plot(ax=ax, label='Gasoline')
df['Automotive diesel (unit/Litre)'].plot(ax=ax, label='Diesel')
df['Light fuel oil (unit/Litre)'].plot(ax=ax, label='Light Fuel')
plt.title('Monthly Fuel Price')
plt.xlabel('Date')
plt.ylabel('USD per Litre')
plt.legend()
plt.show()

# Drop columns
ts = df.drop(columns = ['Gasoline (unit/Litre)', 'Light fuel oil (unit/Litre)', 'Light fuel oil (unit/1000 litres)'])

# Original Time series decomposition
res_decompose = seasonal_decompose(ts, model='additive', extrapolate_trend='freq')

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(9, 6), sharex=True)

res_decompose.observed.plot(ax=axs[0])
axs[0].set_title('Original series', fontsize=12)
res_decompose.trend.plot(ax=axs[1])
axs[1].set_title('Trend', fontsize=12)
res_decompose.seasonal.plot(ax=axs[2])
axs[2].set_title('Seasonal', fontsize=12)
res_decompose.resid.plot(ax=axs[3])
axs[3].set_title('Residuals', fontsize=12)
fig.suptitle('Original time serie decomposition', fontsize=14)
fig.tight_layout()
plt.show()

# Log transformatioh to stabilize variance
ts['log_diesel']=np.log(ts['Automotive diesel (unit/Litre)'])

# Paso 2: Stationarity Test
adfuller_result_ts = adfuller(ts['log_diesel'])
kpss_result_ts = kpss(ts['log_diesel'])

# First Differentiate time serie
ts_diff_1 = ts['log_diesel'].diff().dropna()

adfuller_result_ts1 = adfuller(ts_diff_1)
kpss_result_ts1 = kpss(ts['log_diesel'].diff().dropna())

print('Test stationarity for original series')
print('-------------------------------------')
print(f'ADF Statistic: {adfuller_result_ts[0]}, p-value: {adfuller_result_ts[1]}')
print(f'KPSS Statistic: {kpss_result_ts[0]}, p-value: {kpss_result_ts[1]}')

print('\nTest stationarity for differenced series (order=1)')
print('--------------------------------------------------')
print(f'ADF Statistic: {adfuller_result_ts1[0]}, p-value: {adfuller_result_ts1[1]}')
print(f'KPSS Statistic: {kpss_result_ts1[0]}, p-value: {kpss_result_ts1[1]}')

# Transform and differenciate decomposition
res_descompose_diff_1 = seasonal_decompose(ts_diff_1, model='additive', extrapolate_trend='freq')
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(9, 6), sharex=True)
res_descompose_diff_1.observed.plot(ax=axs[0])
axs[0].set_title('Differenced series (order=1)', fontsize=12)
res_descompose_diff_1.trend.plot(ax=axs[1])
axs[1].set_title('Trend', fontsize=12)
res_descompose_diff_1.seasonal.plot(ax=axs[2])
axs[2].set_title('Seasonal', fontsize=12)
res_descompose_diff_1.resid.plot(ax=axs[3])
axs[3].set_title('Residuals', fontsize=12)
fig.suptitle('Differenced time serie decomposition', fontsize=14)
fig.tight_layout()
plt.show()

# Paso 4: Determine Parameters for SARIMA(p,d,q)(P,D,Q). Autocorrelation analysis

## ACF for original and differentiated series
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 4), sharex=True)
plot_acf(ts['Automotive diesel (unit/Litre)'], ax=axs[0], lags=50, alpha=0.05)
axs[0].set_title('Autocorrelation original series')
plot_acf(ts_diff_1, ax=axs[1], lags=50, alpha=0.05)
axs[1].set_title('Autocorrelation differentiated series (order=1)')
plt.show()

## PACF for original and differenced series
fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (6,4), sharex = True)
plot_pacf(ts['Automotive diesel (unit/Litre)'], ax=axs[0], lags = 50,  alpha=0.05)
axs[0].set_title('Partial autocorrelation original series')
plot_pacf(ts_diff_1, ax=axs[1], lags = 50, alpha =0.05)
axs[1].set_title('Partial autocorrelation diferenced serie (order = 1')
plt.show()

# Split Data in Train and Test
split_index = int(len(ts['log_diesel']) * 0.9)

data_train = ts['log_diesel'].iloc[:split_index]
data_test = ts['log_diesel'].iloc[split_index:]

print('\nTrain and Test Data range')
print('--------------------------------------------------')
print(
    f"Train dates : {ts.index.min()} --- {data_train.index.max()}  "
    f"(n={len(data_train)})"
)
print(
    f"Test dates  : {data_test.index.min()} --- {ts.index.max()}  "
    f"(n={len(data_test)})"
)
# Plot train test
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(data_train.index, data_train.values, label='Train', color='blue')
ax.plot(data_test.index, data_test.values, label='Test', color='orange')
ax.set_title('Monthly fuel consumption in Spain')
ax.legend()
plt.show()

# Fit SARIMA(1,1,1)(1,1,0,12) model
# SARIMA parameters 
sarima_params = {
    'order': (1, 1, 0),       # (p, d, q)
    'seasonal_order': (0, 1, 1, 12)  # (P, D, Q, s)
}

# Initialize and fit the SARIMA model
model = SARIMAX(data_train, order=sarima_params['order'], 
                seasonal_order=sarima_params['seasonal_order'])

results = model.fit(disp=False)

# Print model summary
print(results.summary())

# plot diagnostics to evaluate model assumptions
results.plot_diagnostics(figsize=(10, 8))


# Predictions
predictions = results.get_forecast(steps=len(data_test))
predicted_mean = predictions.predicted_mean
conf_int = predictions.conf_int()

# Plot predicted and observed data
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data_train.index, data_train.values, label='Train data', color='blue')  # Train data in blue
ax.plot(data_test.index, data_test.values, label='Test data', color='orange')  # Test data in orange
ax.plot(data_test.index, predicted_mean, label='Predicted data', color='green')  # Predicted data in green

# Add confidence interval
ax.fill_between(data_test.index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1],
                color='grey', alpha=0.3)

ax.set_title('Monthly Fuel Consumption in Spain with Predictions')
ax.set_xlabel('Date')
ax.set_ylabel('Log Diesel Price')
ax.legend()
plt.show()

import itertools
from sklearn.metrics import mean_absolute_error

# Define parameters for SARIMA
p = range(0, 3)  # AR terms
d = [1]  # Non-seasonal differencing
q = range(0, 3)  # MA terms
P = range(0, 2)  # Seasonal AR terms
D = [1]          # Seasonal differencing
Q = range(0, 2)  # Seasonal MA terms
s = [12]         # Seasonal period

# Create all combinations of parameters
param_grid = list(itertools.product(p, d, q, P, D, Q, s))

# Prepare for storing the best results
best_model = None
best_mae = float('inf')
best_params = None

# Rolling window parameters
n_splits = 6  # Number of splits for cross-validation
split_size = len(data_train) // n_splits

# Rolling window cross-validation
for params in param_grid:
    p, d, q, P, D, Q, s = params
    mae_values = []
    
    for i in range(n_splits):
        # Define the training and validation sets for the current fold
        train_start = 0
        train_end = (i + 1) * split_size
        val_start = train_end
        val_end = (i + 2) * split_size if (i + 2) * split_size < len(data_train) else len(data_train)
        
        if val_start >= len(data_train):
            break  # No more data to validate
        
        train_data = data_train.iloc[train_start:train_end]
        val_data = data_train.iloc[val_start:val_end]
        
        try:
            # Fit the SARIMA model
            model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
            results = model.fit(disp=False)
            
            # Make predictions on the validation set
            forecast = results.forecast(steps=len(val_data))
            
            # Calculate mean absolute error
            mae = mean_absolute_error(val_data, forecast)
            mae_values.append(mae)
            
        except:
            continue

    if mae_values:
        avg_mae = np.mean(mae_values)
        if avg_mae < best_mae:  # We want to minimize MAE
            best_mae = avg_mae
            best_model = results
            best_params = params

print(f'Best Model Parameters: {best_params}')
print(f'Best Model MAE: {best_mae}')

# Forecast from the best model on the test data
best_forecast_values = best_model.get_forecast(steps=len(data_test)).predicted_mean

# Calculate MAE for both models on the test dataset
original_mae = mean_absolute_error(data_test, predicted_mean)  # Replace with your original model's predictions
best_mae_test = mean_absolute_error(data_test, best_forecast_values)

print(f'Original Model MAE: {original_mae}')
print(f'Best Model MAE from Grid Search: {best_mae_test}')

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(data_test.index, data_test.values, label='Actual Data', color='green')
ax.plot(data_test.index, predicted_mean, label='Original Model Forecast', color='orange')  # Ensure `predicted_mean` is defined
ax.plot(data_test.index, best_forecast_values, label='Best Model Forecast', color='red')
plt.title('Forecast Comparison')
plt.legend()
plt.show()

plt.show()