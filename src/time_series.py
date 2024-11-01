#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:40:39 2024

@author: user
"""
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')


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

# Step 1: Visualización y Exploración de Datos

fig, ax = plt.subplots(figsize=(10, 6))
df['Gasoline (unit/Litre)'].plot(ax=ax, label='Gasoline')
df['Automotive diesel (unit/Litre)'].plot(ax=ax, label='Diesel')
df['Light fuel oil (unit/Litre)'].plot(ax=ax, label='Light Fuel')
plt.title('Monthly Fuel Price')
plt.xlabel('Date')
plt.ylabel('USD per Litre')
plt.legend()
plt.show()

# Drop unneeded columns
ts = df.drop(columns = ['Gasoline (unit/Litre)', 'Light fuel oil (unit/Litre)', 'Light fuel oil (unit/1000 litres)'])

'''
Time series models, like ARIMA or SARIMA, assume that the series is stationary. If the series is non-stationary, these models may provide inaccurate or unreliable forecasts. 
Stationarity simplifies the modeling process because it ensures that the underlying patterns in the data (mean, variance, and correlations) do not change over time.
The ADF (Augmented Dickey-Fuller) test and KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test are used to evaluate stationarity.
'''
# Paso 2: Verificar Estacionariedad
print('Test stationarity for original series')
print('-------------------------------------')
adfuller_result = adfuller(ts)
kpss_result = kpss(ts)
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

# Paso 3: Diferenciación para Estacionariedad
ts_diff_1 = ts.diff().dropna()
ts_diff_2 =ts_diff_1.diff().dropna()

print('\nTest stationarity for differenced series (order=1)')
print('--------------------------------------------------')
adfuller_result = adfuller(ts_diff_1)
kpss_result = kpss(ts.diff().dropna())
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

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

'''
Original serie shows growing trend and probably seasonality. The original series is non-stationary based on both ADF (p value > 0.05) and KPSS tests (p value < 0.05).
After first differencing, the series becomes stationary, which is confirmed by both tests.
It's not necessary to deferentiate to second order as the first differencing was enough to achieve stationarity.
'''

# # Paso 4: Identificar los Parámetros p y q. Autocorrelation analysis

## ACF for original and differentiated series

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 4), sharex=True)
plot_acf(ts, ax=axs[0], lags=50, alpha=0.05)
axs[0].set_title('Autocorrelation original series')
plot_acf(ts_diff_1, ax=axs[1], lags=50, alpha=0.05)
axs[1].set_title('Autocorrelation differentiated series (order=1)')
plt.show()

'''
The gradually decaying ACF in original series indicates that the original series is likely non-stationary and has a trend or long-term dependency. 
The ACF in diferetiated serie showing first lag as significative, (slightly above confidence interval) nd the lag 2 is almost zero the q = 1 is cons¡dered. 
The following lags are close to cero (inside confidence o¡interval) and changing signs indicate that the series may have white noise characteristics, which is a good outcome for modeling.
'''

## PACF for original and differenced series
fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (6,4), sharex = True)
plot_pacf(ts, ax=axs[0], lags = 50,  alpha=0.05)
axs[0].set_title('Partial autocorrelation original series')
plot_pacf(ts_diff_1, ax=axs[1], lags = 50, alpha =0.05)
axs[1].set_title('Partial autocorrelation diferenced serie (order = 1')
plt.show()

'''
The PACF suggests significant correlation at lag 1, so p=1 is considered.

Given the observations, an ARIMA(1, 1, 1) model will be appropriate.

- **p = 1**: This indicates the order of the autoregressive (AR) part of the model. It means that the current value of the time series depends on the immediately previous value.

- **d = 1**: This represents the degree of differencing. The series has been differenced once to achieve stationarity, meaning we are using the changes in values rather than the original values.

- **q = 1**: This indicates the order of the moving average (MA) part of the model. It means that the current value of the time series is also influenced by the error (residual) from the previous period.

Mathematical Representation

The ARIMA(1, 1, 1) model can be mathematically represented as:

$y_t - y_{t-1} = \phi_1(y_{t-1} - y_{t-2}) + \theta_1\epsilon_{t-1} + \epsilon_t$

where:

- \( y_t \) is the current value of the time series.
- \( \phi_1 \) is the coefficient of the autoregressive part.
- \( \theta_1 \) is the coefficient of the moving average part.
- \( \epsilon_t \) is the error term at time \( t \).

The ARIMA(1, 1, 1) model effectively combines autoregressive and moving average components to model time series data that has been differenced once to ensure stationarity.
'''

# Split Data in Train and Test
split_index = int(len(ts) * 0.8)

data_train = ts.iloc[:split_index]
data_test = ts.iloc[split_index:]

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

# Fit ARIMA model
arima_model = ARIMA(data_train, order=(1, 1, 1))
arima_model = arima_model.fit()
print(arima_model.summary())

# Model Diagnostic
residuals = arima_model.resid
fig, ax = plt.subplots(2,1, figsize = (15,15))
sns.histplot(residuals, kde = True, ax = ax[0])
plot_acf(residuals, ax=ax[1], title = "Residuals ACF")
plt.show()

# Ljung-Box test for residuals autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print('\n Ljung-Box Test')
print('------------------------------------------------------')
print(lb_test)

'''The fact that most of the residuals are around zero indicates that the model captures the central tendency of the data reasonably well. 
Ideally, residuals should be normally distributed around zero, but the presence of skewness suggests that there may be aspects of the data 
that the model has not fully captured, potentially hinting at the need for further refinement or consideration of additional variables or transformations.
The ACF showing values close to zero and changing sign implies that there is little to no autocorrelation in the residuals. 
Residuals do not exhibit consistent patterns or correlations over time. This is a positive outcome, as it suggests that the residuals are behaving like white noise, 
indicating that the model has adequately captured the temporal dependencies in the data.
The p-value of 0.999751 is very high. This suggests there is no significant autocorrelation. 
This reinforces the conclusion drawn from the ACF plot that the residuals are behaving like white noise.

The diagnostics suggest that while the model captures the central tendencies of the data fairly well (most residuals are around zero), 
the presence of positive skewness indicates the need for attention, particularly regarding the positive outliers. 
This might suggest exploring different modeling techniques or transformations (e.g., log transformation) to address skewness.
The ACF and Ljung-Box test results imply that the model adequately captures the dependencies in the time series, as there is no significant autocorrelation in the residuals. 
However, further investigation into the nature of the outliers may still be warranted to improve model accuracy and robustness.
In summary, while the model appears to perform well overall, especially concerning autocorrelation, the skewness of the residuals highlights an area for potential improvement.
'''

# Prediction with ARIMA
arima_forecast = arima_model.get_forecast(len(data_test.index))
arima_forecast_mean = arima_forecast.predicted_mean
arima_forecast_mean.name = 'predictions_ARIMA'''


# First-order differentiation combined with seasonal differentiation
data_diff_1_12 = data_train.diff().diff(12).dropna()
adfuller_result = adfuller(data_diff_1_12)
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
kpss_result = kpss(data_diff_1_12)
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

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

