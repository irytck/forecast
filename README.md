# Time Series Analysis of Diesel Fuel Prices in Spain (2015-2024)

## Project Overview

This project explores the diesel fuel price trends in Spain over the period from 2015 to 2024, focusing on seasonality and the impact of external shocks. By analyzing these trends, the project aims to provide insights into recurring patterns and the volatility caused by external factors such as oil supply disruptions, geopolitical events, and natural disasters. Such analysis can be invaluable for fuel-dependent industries—like airlines and logistics—by supporting more accurate budgeting and strategic pricing decisions.

## Objectives

- **Seasonality Analysis**: Identify and quantify seasonal trends in diesel fuel prices.
- **External Shock Analysis**: Assess and model the impact of significant external events on price volatility.
- **Forecasting Volatility**: Use advanced statistical models to anticipate future fluctuations in diesel prices, considering both seasonal and shock-based variations.

## Methodology

### Univariate Time Series Analysis

- **SARIMA Modeling**: A Seasonal Autoregressive Integrated Moving Average (SARIMA) model was applied to the diesel price data to decompose the time series into trend, seasonality, and noise components. This approach helps reveal recurring patterns that may influence price behavior in the long term.
- **Seasonal Decomposition**: By decomposing the series, we separate out seasonal fluctuations and analyze underlying trends.

### Multivariate Time Series Analysis

- **Exogenous Variables**: To enhance predictive accuracy, exogenous variables—such as global crude oil production and stock market indices—were incorporated. These external factors provide context for understanding diesel price volatility.
- **GARCH Models**: Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models were used to quantify and forecast volatility, especially in response to external shocks.

## Data Sources

- **Primary Data Source**: [IEA Monthly Oil Price Statistics](https://www.iea.org/data-and-statistics/data-product/monthly-oil-price-statistics-2), published by the International Energy Agency (IEA).
- **License**: Terms of Use for Non-CC Material as specified by the IEA.

## Installation

1. Clone this repository
2. Install the required Python packages

## Usage

1. **Data Preparation**: Load and preprocess the diesel fuel price data from IEA, along with any additional exogenous variables.
2. **Model Training**: Train the SARIMA and GARCH models using the preprocessed data.
3. **Analysis and Forecasting**: Run the models to perform analysis and forecast future prices and volatility, considering both seasonal patterns and external shocks.

## Results and Insights

The project provides the following outputs:
- Seasonal trends and their impact on diesel fuel prices in Spain.
- Quantified effects of external shocks on price volatility.
- Forecasted price and volatility trends to support strategic planning for fuel-dependent sectors.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
