#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import stumpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data(ticker, start="2005-01-01", end=None):
    data = yf.download(ticker, start=start, end=end)
    return data['Close']

def main():
    st.title("Stock Matrix Profile Analysis")

    ticker = st.text_input("Enter stock ticker (e.g., QQQ):", "QQQ")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date for analysis", pd.to_datetime("2023-01-01"))
    with col2:
        end_date = st.date_input("End date for analysis", pd.to_datetime("2023-12-31"))

    if st.button("Analyze"):
        # Fetch data from 2005 to the end date
        data = get_data(ticker, start="2005-01-01", end=end_date)

        # Calculate window size (e.g., 30 days)
        window_size = 30

        # Extract the subsequence for the user-defined date range
        subsequence = data[start_date:end_date]

        if len(subsequence) < window_size:
            st.error(f"Selected date range is too short. Please select at least {window_size} days.")
            return

        # Calculate matrix profile
        mp = stumpy.stump(data, m=len(subsequence))

        # Find the top 3 closest matches
        top_matches_idx = np.argsort(mp[:, 0])[:3]

        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot full time series
        axs[0].plot(data.index, data, label='Full Time Series')
        axs[0].set_title('Complete Time Series')
        axs[0].legend()

        # Plot subsequence and matches
        axs[1].plot(subsequence.index, subsequence, label='Selected Window', color='red', linewidth=2)
        for idx in top_matches_idx:
            match_data = data.iloc[idx:idx+len(subsequence)]
            axs[1].plot(match_data.index, match_data, label=f'Match starting at {match_data.index[0].date()}')
        
        axs[1].set_title('Selected Window and Top Matches')
        axs[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # Display match details
        st.subheader("Top Matches Details")
        for idx in top_matches_idx:
            match_start = data.index[idx]
            match_end = data.index[idx + len(subsequence) - 1]
            st.write(f"Match period: {match_start.date()} to {match_end.date()}")

if __name__ == "__main__":
    main()

