#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:09:45 2024

@author: user
"""

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
from io import StringIO
import contextlib
import re
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

# pmdarima
import pmdarima
from pmdarima import ARIMA
from pmdarima import auto_arima

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

original_model = model.fit(disp=False)

# Print model summary
print(original_model.summary())

# plot diagnostics to evaluate model assumptions
original_model.plot_diagnostics(figsize=(10, 8))


# Predictions
predictions = original_model.get_forecast(steps=len(data_test))
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

# Train-validation-test data
# ======================================================================================
end_train = '2022-02-01'
end_val = '2023-10-01'
print(
    f"Train dates      : {ts.index.min()} --- {ts.loc[:end_train].index.max()}  "
    f"(n={len(ts.loc[:end_train])})"
)
print(
    f"Validation dates : {ts.loc[end_train:].index.min()} --- {ts.loc[:end_val].index.max()}  "
    f"(n={len(ts.loc[end_train:end_val])})"
)
print(
    f"Test dates       : {ts.loc[end_val:].index.min()} --- {ts.index.max()}  "
    f"(n={len(ts.loc[end_val:])})"
)

# Plot
# ======================================================================================
fig, ax = plt.subplots(figsize=(7, 3))
ts['log_diesel'].loc[:end_train].plot(ax=ax, label='train')
ts['log_diesel'].loc[end_train:end_val].plot(ax=ax, label='validation')
ts['log_diesel'].loc[end_val:].plot(ax=ax, label='test')
ax.set_title('Monthly fuel price in Spain')
ax.legend()
plt.show()

# Grid search based on backtesting
# ==============================================================================
forecaster = ForecasterSarimax(
                 regressor = Sarimax(
                                order   = (1, 1, 1), # Placeholder replaced in the grid search
                                maxiter = 500
                             )
             )

param_grid = {
    'order': [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)],
    'seasonal_order': [(0, 0, 0, 0), (0, 1, 0, 12), (1, 1, 1, 12), (1, 1, 0, 12)],
    'trend': [None, 'n', 'c']
}

results_grid = grid_search_sarimax(
                   forecaster            = forecaster,
                   y                     = ts['log_diesel'].loc[:end_val],
                   param_grid            = param_grid,
                   steps                 = 12,
                   refit                 = True,
                   metric                = 'mean_absolute_error',
                   initial_train_size    = len(data_train),
                   fixed_train_size      = False,
                   return_best           = False,
                   n_jobs                = 'auto',
                   suppress_warnings_fit = True,
                   verbose               = False,
                   show_progress         = True
               )
results_grid.head(5)

# Auto arima: selection based on AIC
# ==============================================================================
model = auto_arima(
            y                 = ts['log_diesel'].loc[:end_val],
            start_p           = 0,
            start_q           = 0,
            max_p             = 3,
            max_q             = 3,
            seasonal          = True,
            test              = 'adf',
            m                 = 12, # Seasonal period
            d                 = None, # The algorithm will determine 'd'
            D                 = None, # The algorithm will determine 'D'
            trace             = True,
            error_action      = 'ignore',
            suppress_warnings = True,
            stepwise          = True
        )

# Backtest predictions with the best model according to grid search
# ==============================================================================
forecaster = ForecasterSarimax(
                 regressor=Sarimax(order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), maxiter=500),
             )

metric_m1, predictions_m1 = backtesting_sarimax(
                                forecaster            = forecaster,
                                y                     = ts['log_diesel'],
                                initial_train_size    = len(ts['log_diesel'].loc[:end_val]),
                                steps                 = 12,
                                metric                = 'mean_absolute_error',
                                refit                 = True,
                                n_jobs                = "auto",
                                suppress_warnings_fit = True,
                                verbose               = False,
                                show_progress         = True
                            )

# Backtest predictions with the best model according to auto-arima
# ==============================================================================
forecaster = ForecasterSarimax(
                 regressor=Sarimax(order=(0, 1, 1), seasonal_order=(0, 1, 0, 12), maxiter=500),
             )

metric_m2, predictions_m2 = backtesting_sarimax(
                                forecaster            = forecaster,
                                y                     = ts['log_diesel'],
                                initial_train_size    = len(ts['log_diesel'].loc[:end_val]),
                                steps                 = 12,
                                metric                = 'mean_absolute_error',
                                refit                 = True,
                                n_jobs                = "auto",
                                suppress_warnings_fit = True,
                                verbose               = False,
                                show_progress         = True
                            )

# Compare predictions
# ==============================================================================
print("Metric (mean_absolute_error) for grid search model:")
print(metric_m1)
print("Metric (mean_absolute_error) for auto arima-model:")
print(metric_m2)

fig, ax = plt.subplots(figsize=(6, 3))
ts['log_diesel'].loc[end_val:].plot(ax=ax, label='test')
predictions_m1 = predictions_m1.rename(columns={'pred': 'grid search'})
predictions_m2 = predictions_m2.rename(columns={'pred': 'autoarima'})
predictions_m1.plot(ax=ax)
predictions_m2.plot(ax=ax)
ax.set_title('Backtest predictions with ARIMA model')
ax.legend()
plt.show()