#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import yfinance as yf
import stumpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch historical data from Yahoo Finance
def get_data(ticker):
    data = yf.download(ticker, start="2010-01-01")
    return data['Close']

# Streamlit app
def main():
    st.title("Stock Matrix Profile Analysis")

    # User input for ticker
    ticker = st.text_input("Enter stock ticker (e.g., QQQ):", "QQQ")

    if st.button("Analyze"):
        data = get_data(ticker)

        # Display closing prices
        st.subheader(f"Historical Closing Prices for {ticker}")
        st.line_chart(data)

        # Matrix Profile analysis
        m = 50  # Window size
        mp = stumpy.stump(data, m)
        motifs_idx = np.argsort(mp[:, 0])[:2]

        # Plot motifs
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(data, label='Time Series Data')
        axs[0].set_title('Complete Time Series')
        axs[0].legend()

        for idx in motifs_idx:
            axs[1].plot(np.arange(idx, idx + m), data[idx:idx + m], label=f'Motif starting at {idx}')
        axs[1].set_title('Detected Motifs')
        axs[1].legend()

        st.pyplot(fig)

if __name__ == "__main__":
    main()


# In[ ]:




