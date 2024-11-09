#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:29:41 2024

@author: user

Multivariate Time Series analysis using RNN (Recurent NEural Network) specifically LSTM(Long-short-term memory)
Data analysis for gasoline price monthly from 01/01/2015 to 01/09/2024

Data source for Crude oil WTI: https://finance.yahoo.com/quote/CL%3DF/history/?period1=1420070400&period2=1725148800&frequency=1mo

"""

# Import libraries
import yfinance as yf
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load Data

# Define the symbol and time period for Crude oil WTI prices
symbol = "CL=F"
start_date = "2015-01-01"
end_date = "2024-01-09"

# Fetch monthly data and select 'Adj Close' column
clf_ts = yf.download(symbol, start=start_date, end=end_date, interval="1mo")[('Adj Close', 'CL=F')]

# Rename column and set the frequency of the index to start-of-month ('MS')
clf_ts.name = 'CL=F'
clf_ts = clf_ts.asfreq('MS').sort_index()

# Check for missing values
missing_rows = clf_ts[clf_ts.isna()]
print(f'Number of rows with missing values: {clf_ts.isnull().mean()}')

# Interpolate missing values
clf_ts = clf_ts.interpolate()

# Verify if the index is complete
is_index_complete = clf_ts.index.equals(pd.date_range(start=clf_ts.index.min(), end=clf_ts.index.max(), freq='MS'))
print(f"Index complete: {is_index_complete}")


# Plot
plt.plot(clf_ts)
plt.show()

# Convert to numpy array
clf_ts = clf_ts.values.reshape(-1, 1)

# LSTM uses sigmoid and tanh that are sensitive to magnitude sol values must be normalized
scaler = MinMaxScaler(feature_range=(0,1))
clf_ts = scaler.fit_transform(clf_ts)

# Split in train and test
train_size = int(len(clf_ts)*0.7)
test_size = len(clf_ts) - train_size
train, test = clf_ts[0:train_size, :], clf_ts[train_size:len(clf_ts),:]

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
trainPredictPlot = np.empty_like(clf_ts)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(clf_ts)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(seq_size*2)+1:len(clf_ts)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(clf_ts), label='Baseline (Actual)')
plt.plot(trainPredictPlot, label='Train Predictions')
plt.plot(testPredictPlot, label='Test Predictions')
plt.legend()
plt.show()
