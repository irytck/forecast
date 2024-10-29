#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:40:39 2024

@author: user
"""
# Libraries
# ======================================================================================
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
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
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
warnings.filterwarnings('once')

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

fig, ax = plt.subplots(figsize=(10, 6))
df['Gasoline (unit/Litre)'].plot(ax=ax, label='Gasoline')
df['Automotive diesel (unit/Litre)'].plot(ax=ax, label='Diesel')
df['Light fuel oil (unit/Litre)'].plot(ax=ax, label='Light Fuel')
plt.title('Monthly Fuel Price')
plt.xlabel('Date')
plt.ylabel('USD per Litre')
plt.legend()
plt.show()

# Time series
ts = df.drop(columns = ['Gasoline (unit/Litre)', 'Light fuel oil (unit/Litre)', 'Light fuel oil (unit/1000 litres)'])

'''
Time series models, like ARIMA or SARIMA, assume that the series is stationary. If the series is non-stationary, these models may provide inaccurate or unreliable forecasts. 
Stationarity simplifies the modeling process because it ensures that the underlying patterns in the data (mean, variance, and correlations) do not change over time.
The ADF (Augmented Dickey-Fuller) test and KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test are used to evaluate stationarity.
'''

warnings.filterwarnings("ignore")

ts_diff_1 = ts.diff().dropna()
ts_diff_2 =ts_diff_1.diff().dropna()

print('Test stationarity for original series')
print('-------------------------------------')
adfuller_result = adfuller(ts)
kpss_result = kpss(ts)
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

print('\nTest stationarity for differenced series (order=1)')
print('--------------------------------------------------')
adfuller_result = adfuller(ts_diff_1)
kpss_result = kpss(ts.diff().dropna())
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

print('\nTest stationarity for differenced series (order=2)')
print('--------------------------------------------------')
adfuller_result = adfuller(ts_diff_2)
kpss_result = kpss(ts.diff().diff().dropna())
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

warnings.filterwarnings("default")

# Plot series

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 5), sharex=True)
ts.plot(ax=axs[0], title='Original time series')
ts_diff_1.plot(ax=axs[1], title='Differenced order 1')
ts_diff_2.plot(ax=axs[2], title='Differenced order 2')
plt.show()

'''
The original series is non-stationary based on both ADF and KPSS tests.
After first differencing, the series becomes stationary, which is confirmed by both tests.
Second differencing is also stationary, but it's not necessary as the first differencing was enough to achieve stationarity.
Can proceed with model building using the first-differenced series, as it is stationary and suitable for time series models.
'''

# Autocorrelation Analysis

## Autocorrelation plot for original and differentiated series

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 4), sharex=True)
plot_acf(ts, ax=axs[0], lags=50, alpha=0.05)
axs[0].set_title('Autocorrelation original series')
plot_acf(ts_diff_1, ax=axs[1], lags=50, alpha=0.05)
axs[1].set_title('Autocorrelation differentiated series (order=1)')
plt.show()

'''
The gradually decaying ACF in original series indicates that the original series is likely non-stationary and has a trend or long-term dependency. 
The ACF in diferetiated serie showing a zero value at the first lag and oscillating values for subsequent lags indicates that the differencing has removed 
the long-term dependencies, achieving stationarity. The changing signs indicate that the series may have white noise characteristics, which is a good outcome for modeling.
'''

## Partial autocorrelation plot for original and differenced series
fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (6,4), sharex = True)
plot_pacf(ts, ax=axs[0], lags = 50,  alpha=0.05)
axs[0].set_title('Partial autocorrelation original series')
plot_pacf(ts_diff_1, ax=axs[1], lags = 50, alpha =0.05)
axs[1].set_title('Partial autocorrelation diferenced serie (order = 1')
plt.show()

'''
The sharp cutoff after lag 1 in the PACF indicates that the original series likely follows an AR(1) process, where the immediate past value is significantly correlated with the present value. 
This is consistent with the observation of non-stationarity due to the long-term dependency seen in the ACF.
Differenced Series: The cutoff at lag 0 suggests that the differencing has effectively removed the autocorrelation, indicating that the series is now stationary. 
Since the values do not correlate significantly beyond lag 0, this indicates that there is no longer a significant AR structure in the differenced series.

Given these observations, the following modeling approach is considered:
For the original series, an ARIMA(1, 0, 0) (or AR(1)) model might be appropriate since the PACF suggests significant correlation at lag 1.
For the differenced series, the absence of significant autocorrelation (cutoff at lag 0) may suggest that could fit a model with no AR component, potentially considering a 
moving average component (if the ACF indicates any behavior that warrants it).
'''

# Time series decomposition

res_decompose = seasonal_decompose(ts, model='additive', extrapolate_trend='freq')
res_descompose_diff_2 = seasonal_decompose(ts_diff_1, model='additive', extrapolate_trend='freq')

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9, 6), sharex=True)

res_decompose.observed.plot(ax=axs[0, 0])
axs[0, 0].set_title('Original series', fontsize=12)
res_decompose.trend.plot(ax=axs[1, 0])
axs[1, 0].set_title('Trend', fontsize=12)
res_decompose.seasonal.plot(ax=axs[2, 0])
axs[2, 0].set_title('Seasonal', fontsize=12)
res_decompose.resid.plot(ax=axs[3, 0])
axs[3, 0].set_title('Residuals', fontsize=12)
res_descompose_diff_2.observed.plot(ax=axs[0, 1])
axs[0, 1].set_title('Differenced series (order=1)', fontsize=12)
res_descompose_diff_2.trend.plot(ax=axs[1, 1])
axs[1, 1].set_title('Trend', fontsize=12)
res_descompose_diff_2.seasonal.plot(ax=axs[2, 1])
axs[2, 1].set_title('Seasonal', fontsize=12)
res_descompose_diff_2.resid.plot(ax=axs[3, 1])
axs[3, 1].set_title('Residuals', fontsize=12)
fig.suptitle('Time serie decomposition original series versus differenced series', fontsize=14)
fig.tight_layout()
plt.show()


# Split Train and Test
split_index = int(len(ts) * 0.8)

data_train = ts.iloc[:split_index]
data_test = ts.iloc[split_index:]

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

# First-order differentiation combined with seasonal differentiation
data_diff_1_12 = data_train.diff().diff(12).dropna()

warnings.filterwarnings("ignore")
adfuller_result = adfuller(data_diff_1_12)
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
kpss_result = kpss(data_diff_1_12)
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')
warnings.filterwarnings("default")

'''
La serie original es no estacionaria, pero tras una diferenciación de primer orden se vuelve estacionaria. 
Por lo tanto, una diferenciación de primer orden es suficiente para lograr estacionariedad en esta serie, 
lo que es ideal para el modelado de series temporales, como los modelos ARIMA.
'''

# Fit ARIMA model
arima_model = ARIMA(data_train, order=(1, 1, 1))
arima_fit = arima_model.fit()
print(arima_fit.summary())

# Prediction with ARIMA
arima_forecast = arima_fit.get_forecast(steps=len(data_test))
arima_forecast_mean = arima_forecast.predicted_mean
arima_forecast_mean.name = 'predictions_ARIMA'

# Fit SARIMA model
sarima_model = SARIMAX(data_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)
print(sarima_fit.summary())

# Prediction with SARIMA
sarima_forecast = sarima_fit.get_forecast(steps=len(data_test))
sarima_forecast_mean = sarima_forecast.predicted_mean
sarima_forecast_mean.name = 'predictions_SARIMA'

# Plotting the original series and forecasts
plt.figure(figsize=(15, 8))
plt.plot(data_train, label='Original Time Series', color='blue')
plt.plot(arima_forecast_mean, label='ARIMA Forecast', color='orange')
plt.plot(sarima_forecast_mean, label='SARIMA Forecast', color='green')
plt.plot(data_test, label='Actual Test Data', color='red', linestyle='--')  # Adding actual test data
plt.title('Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

