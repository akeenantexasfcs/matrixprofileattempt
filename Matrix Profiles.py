#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Required libraries
import yfinance as yf
import stumpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch historical data from Yahoo Finance
def get_data(ticker):
    """
    Downloads and returns the closing price data for a given stock ticker.

    Parameters:
    ticker (str): The stock symbol to fetch data for (e.g., 'QQQ').

    Returns:
    pandas.Series: A time series of closing prices.
    """
    # Download stock data from start date to the current date
    data = yf.download(ticker, start="2010-01-01")
    # Return only the 'Close' column, which contains the daily closing prices
    return data['Close']

# Define the ticker for which we want to analyze the historical data
ticker = 'QQQ'  # Nasdaq-100 ETF as an example

# Fetch the data using the get_data function
data = get_data(ticker)

# Plotting the closing prices
plt.figure(figsize=(10, 4))
plt.plot(data, label='Closing Prices')  # Plot the data
plt.title(f'Historical Closing Prices for {ticker}')  # Title of the plot
plt.xlabel('Date')  # X-axis label
plt.ylabel('Price')  # Y-axis label
plt.legend()  # Show legend to identify the plot
plt.show()  # Display the plot

# Optional: Trigger the analysis manually by setting 'Analyze' to True elsewhere in the notebook
if 'Analyze' in globals():
    # Set the window size for the motif analysis
    m = 50  # Smaller window sizes detect smaller patterns, larger windows detect broader trends

    # Calculate the matrix profile, which helps identify the smallest distance between any subsequence within the time series and all others
    mp = stumpy.stump(data, m)

    # Find the indices of the two best motifs (lowest values in the matrix profile signify the most similar or repetitive patterns)
    motifs_idx = np.argsort(mp[:, 0])[:2]

    # Plotting the detected motifs alongside the full data
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(data, label='Time Series Data')
    axs[0].set_title('Complete Time Series')
    axs[0].legend()

    for idx in motifs_idx:
        # Plot each motif. We add 'idx' to the x-values to align the motifs with their position in the full time series
        axs[1].plot(np.arange(idx, idx + m), data[idx:idx + m], label=f'Motif starting at {idx}')
    axs[1].set_title('Detected Motifs')
    axs[1].legend()

    plt.tight_layout()  # Adjust subplots to give some padding and prevent overlap
    plt.show()  # Display the plots


# In[ ]:




