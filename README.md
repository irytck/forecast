# Seasonality and External Shocks in Fuel Price Trends
**Problem:**
Fuel prices are often subject to seasonal demand fluctuations (e.g., higher prices in summer due to increased travel) and external shocks such as natural disasters, geopolitical conflicts, or pandemic-related disruptions. Analyzing these patterns can help stakeholders prepare for seasonal price changes and respond effectively to sudden price shocks.

**Solution:**
Time series analysis can:

Identify recurring seasonal patterns in fuel prices, such as price increases during travel-heavy periods (summer, holidays).
Detect and model the impact of external shocks (e.g., oil supply disruptions, hurricanes) on fuel price volatility.
Allow for more accurate budgeting and pricing strategies for fuel-dependent industries (e.g., airlines, logistics).

**Methodology:**

Use seasonal decomposition of time series (SARIMA or Fourier terms) to separate out seasonal trends and identify recurring patterns.
Apply GARCH models to quantify and forecast volatility caused by external shocks.
Conduct event analysis to measure the immediate and long-term impacts of specific shocks (e.g., COVID-19, OPEC decisions) on monthly fuel prices.

**Data Source:**

IEA, Monthly Oil Price Statistics, IEA, Paris https://www.iea.org/data-and-statistics/data-product/monthly-oil-price-statistics-2, Licence: Terms of Use for Non-CC Material