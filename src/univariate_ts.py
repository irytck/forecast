"""
Created on Wed Oct 23 12:40:39 2024

@author: user
"""
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
plt.style.use('seaborn-v0_8-darkgrid')

# pmdarima
from pmdarima import auto_arima

# statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# skforecast
from skforecast.plot import set_dark_theme
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from skforecast.model_selection_sarimax import grid_search_sarimax

# LSTM
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# Set pandas options to display all columns
pd.set_option('display.max_columns', None)
set_dark_theme()
# =============================================================================
# Load Data
# =============================================================================
data = pd.read_excel("/Users/User/projects/forecast/data/energy_prices_raw.xlsx")

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

# =============================================================================
# Step 1: EDA
# =============================================================================

# Plot Data
fig, ax = plt.subplots(figsize=(10, 6))
df['Gasoline (unit/Litre)'].plot(ax=ax, label='Gasoline')
df['Automotive diesel (unit/Litre)'].plot(ax=ax, label='Diesel')
df['Light fuel oil (unit/Litre)'].plot(ax=ax, label='Light Fuel')
plt.title('Monthly Fuel Price')
plt.xlabel('Date')
plt.ylabel('USD per Litre')
plt.legend()
plt.show()

# Drop unnecessary columns for analysis
data = df.drop(columns = ['Gasoline (unit/Litre)', 'Light fuel oil (unit/Litre)', 'Light fuel oil (unit/1000 litres)'])

# Original Time series Decomposition
res_decompose = seasonal_decompose(data, model='additive', extrapolate_trend='freq')

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

# Log transformation to stabilize variance
data['log_diesel']=np.log(data['Automotive diesel (unit/Litre)'])

# =============================================================================
# Step2: Stationarity Test
# =============================================================================
ts = data['log_diesel']
adfuller_result_ts = adfuller(ts)
kpss_result_ts = kpss(ts)

# First Differencing to achieve stationarity
ts_diff_1 = ts.diff().dropna()

adfuller_result_ts1 = adfuller(ts_diff_1)
kpss_result_ts1 = kpss(ts.diff().dropna())

print('Test stationarity for original series')
print('-------------------------------------')
print(f'ADF Statistic: {adfuller_result_ts[0]}, p-value: {adfuller_result_ts[1]}')
print(f'KPSS Statistic: {kpss_result_ts[0]}, p-value: {kpss_result_ts[1]}')

print('\nTest stationarity for differenced series (order=1)')
print('--------------------------------------------------')
print(f'ADF Statistic: {adfuller_result_ts1[0]}, p-value: {adfuller_result_ts1[1]}')
print(f'KPSS Statistic: {kpss_result_ts1[0]}, p-value: {kpss_result_ts1[1]}')

# Decompose differenced series to analyze trend, seasonality, and residuals
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

# =============================================================================
# Step 3: Autocorrelation analysis (ACF and PACF) to determine SARIMA parameters
# =============================================================================

## ACF for original and differentiated series
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 4), sharex=True)
plot_acf(data['Automotive diesel (unit/Litre)'], ax=axs[0], lags=50, alpha=0.05)
axs[0].set_title('Autocorrelation original series')
plot_acf(ts_diff_1, ax=axs[1], lags=50, alpha=0.05)
axs[1].set_title('Autocorrelation differentiated series (order=1)')
plt.show()

## PACF for original and differenced series
fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (6,4), sharex = True)
plot_pacf(data['Automotive diesel (unit/Litre)'], ax=axs[0], lags = 50,  alpha=0.05)
axs[0].set_title('Partial autocorrelation original series')
plot_pacf(ts_diff_1, ax=axs[1], lags = 50, alpha =0.05)
axs[1].set_title('Partial autocorrelation diferenced serie (order = 1')
plt.show()

# =============================================================================
# Step 4: Model Fit
# =============================================================================

# Split Data into Train and Test
split_index = int(len(ts) * 0.9)

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

# Fit SARIMA(1,1,1)(1,1,0,12) model
sarima_params = {
    'order': (1, 1, 0),       # (p, d, q)
    'seasonal_order': (0, 1, 1, 12)  # (P, D, Q, s)
}

# Initialize and fit SARIMA model
model = SARIMAX(data_train, order=sarima_params['order'], 
                seasonal_order=sarima_params['seasonal_order'])

original_model = model.fit(disp=False)

# =============================================================================
# Step 5: Diagnostic
# =============================================================================

# Model summary
print(original_model.summary())

# plot diagnostics to evaluate model assumptions
original_model.plot_diagnostics(figsize=(10, 8))

# =============================================================================
# Step 6: Predictions
# =============================================================================

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

# =============================================================================
# HYPERPARAMETERS OPTIMIZATION
# =============================================================================

# Step 1: Split in train validation and test
# =============================================================================
end_train = '2022-02-01'
end_val = '2023-10-01'
print(
    f"\nTrain dates      : {ts.index.min()} --- {ts.loc[:end_train].index.max()}  "
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

# Plot train, validation and test
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 3))
ts.loc[:end_train].plot(ax=ax, label='train')
ts.loc[end_train:end_val].plot(ax=ax, label='validation')
ts.loc[end_val:].plot(ax=ax, label='test')
ax.set_title('Monthly fuel price in Spain')
ax.legend()
plt.show()

# Grid search based on backtesting
# =============================================================================
forecaster = ForecasterSarimax(
                 regressor = Sarimax(
                                order   = (1, 1, 1), # Placeholder replaced in the grid search
                                maxiter = 500
                             )
             )

param_grid = {
    'order': [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)],
    'seasonal_order': [(0, 0, 0, 12), (1, 1, 0, 12), (1, 1, 1, 12), (0, 1, 0, 12)],
    'trend': [None, 'n', 'c']
}

results_grid = grid_search_sarimax(
                   forecaster            = forecaster,
                   y                     = ts.loc[:end_val],
                   param_grid            = param_grid,
                   steps                 = 12,
                   refit                 = True,
                   metric                = 'mean_absolute_error',
                   initial_train_size    = len(data_train),
                   fixed_train_size      = False,
                   return_best           = True,
                   n_jobs                = 'auto',
                   suppress_warnings_fit = True,
                   verbose               = False,
                   show_progress         = True
               )
print(results_grid[['order','seasonal_order', 'mean_absolute_error']].head(5))

# Auto arima: selection based on AIC
# =============================================================================
model = auto_arima(
            y                 = ts.loc[:end_val],
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
# =============================================================================
forecaster = ForecasterSarimax(
                 regressor=Sarimax(order=(2, 1, 1), seasonal_order=(0, 1, 0, 12), maxiter=500),
             )

metric_m1, predictions_m1 = backtesting_sarimax(
                                forecaster            = forecaster,
                                y                     = ts,
                                initial_train_size    = len(ts.loc[:end_val]),
                                steps                 = 12,
                                metric                = 'mean_absolute_error',
                                refit                 = True,
                                n_jobs                = "auto",
                                suppress_warnings_fit = True,
                                verbose               = False,
                                show_progress         = True
                            )

# Backtest predictions with the best model according to auto-arima
# =============================================================================
forecaster = ForecasterSarimax(
                 regressor=Sarimax(order=(0, 1, 1), seasonal_order=(0, 1, 0, 12), maxiter=500),
             )

metric_m2, predictions_m2 = backtesting_sarimax(
                                forecaster            = forecaster,
                                y                     = ts,
                                initial_train_size    = len(ts.loc[:end_val]),
                                steps                 = 12,
                                metric                = 'mean_absolute_error',
                                refit                 = True,
                                n_jobs                = "auto",
                                suppress_warnings_fit = True,
                                verbose               = False,
                                show_progress         = True
                            )

# Backtest predictions with the original model SARIMA(1,1,1)(1,1,0,12)
# =============================================================================
forecaster = ForecasterSarimax(
                 regressor=Sarimax(order=(1,1,1), seasonal_order=(1,1,0,12), maxiter=500),
    )
metric_m0, predictions_m0 = backtesting_sarimax(
    forecaster = forecaster, 
    y = ts, 
    initial_train_size = len(ts.loc[:end_val]),
    steps = 12, 
    metric = 'mean_absolute_error', 
    refit = True,
    n_jobs = "auto",
    suppress_warnings_fit = True,
    verbose               = False,
    show_progress         = True
    )


# Compare predictions
# =============================================================================
print("\nMEAN ABSOLUTE ERROR")
print("________________________________________________________________")
print("\nMetric (mean_absolute_error) for original model:")
print(metric_m0)
print("\nMetric (mean_absolute_error) for grid search model:")
print(metric_m1)
print("\nMetric (mean_absolute_error) for auto arima-model:")
print(metric_m2)

fig, ax = plt.subplots(figsize=(6, 3))

# Plot Actual data with Predictions
# =============================================================================
ts.loc[end_val:].plot(ax=ax, label='test', color='grey')

predictions_m0 = predictions_m0.rename(columns={'pred': 'original model'})
predictions_m1 = predictions_m1.rename(columns={'pred': 'grid search'})
predictions_m2 = predictions_m2.rename(columns={'pred': 'autoarima'})

# Plot predictions
predictions_m0['original model'].plot(ax=ax, label='original model', color='orange', linestyle='--')
predictions_m1['grid search'].plot(ax=ax, label='grid search', color='green', linestyle='--')
predictions_m2['autoarima'].plot(ax=ax, label='autoarima', color='blue', linestyle='--')

ax.set_title('Backtest predictions with ARIMA model')
ax.legend()
plt.show()

# =============================================================================
# RECURENT NEURAL NETWORK LSTM
# =============================================================================
# Convert to numpy array
ts = data['Automotive diesel (unit/Litre)']
ts = ts.values.reshape(-1, 1)

# LSTM uses sigmoid and tanh that are sensitive to magnitude sol values must be normalized
scaler = MinMaxScaler(feature_range=(0,1))
ts = scaler.fit_transform(ts)

# Split in train and test
train_size = int(len(ts)*0.7)
test_size = len(ts) - train_size
train, test = ts[0:train_size, :], ts[train_size:len(ts),:]

# Define function to convert an array of values into a dataset matrix

def to_sequences(data, seq_size=1):
    '''Creates a dataset where X is the value at a given time step (t, t-1, t-2, ...)
    and y is a value at the next time step (t+1)
    data: Serie temporal
    seq_size: Number of previous time steps to use as input variables to predict the next time period'''
    x = []
    y = []

    for i in range(len(data)-seq_size-1):
        #print(i)
        window = data[i:(i+seq_size), 0]
        x.append(window)
        y.append(data[i+seq_size, 0])
        
    return np.array(x),np.array(y)

seq_size = 7

trainX, trainY = to_sequences(train, seq_size=seq_size)
testX, testY = to_sequences(test, seq_size=seq_size)

print("Shape of training set: {}".format(trainX.shape))
print("Shape of test set: {}".format(testX.shape))

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print('Single LSTM with hidden Dense:')
model = Sequential()
model.add(LSTM(64, input_shape=(None, seq_size)))
model.add(Dense(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, 
#                        verbose=1, mode='auto', restore_best_weights=True)
model.summary()
print('Train:')

model.fit(trainX, trainY, validation_data=(testX, testY),
          verbose=2, epochs=100)


# make predictions

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions back to prescaled values
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(ts)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(ts)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(seq_size*2)+1:len(ts)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(ts), label='Baseline (Actual)')
plt.plot(trainPredictPlot, label='Train Predictions')
plt.plot(testPredictPlot, label='Test Predictions')
plt.legend()
plt.show()



