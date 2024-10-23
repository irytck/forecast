#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:40:39 2024

@author: user
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set pandas options to display all columns
pd.set_option('display.max_columns', None)

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
ts = pivoted_data.asfreq('MS').sort_index()

# Display ts
print(ts.head())

# Missing values
print(f'Number of rows with missing values: {ts.isnull().any(axis=1).mean()}')

# Verify that a temporary index is complete
start_date = ts.index.min()
end_date = ts.index.max()
date_range = pd.date_range(start=start_date, end=end_date, freq=ts.index.freq)
is_index_complete = (ts.index == date_range).all()
print(f"Index complete: {is_index_complete}")

# Convert Light Fuel Oil prices from per 1000 litres to per litre
ts['Light fuel oil (unit/Litre)'] = ts['Light fuel oil (unit/1000 litres)'] / 1000

fig, ax = plt.subplots(figsize=(10, 6))
ts['Gasoline (unit/Litre)'].plot(ax=ax, label='Gasoline')
ts['Automotive diesel (unit/Litre)'].plot(ax=ax, label='Diesel')
ts['Light fuel oil (unit/Litre)'].plot(ax=ax, label='Light Fuel')
plt.title('Monthly Gasoline Price')
plt.xlabel('Date')
plt.ylabel('USD per Litre')
plt.legend()
plt.show()
