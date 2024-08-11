#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import stumpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def get_data(ticker, start="2005-01-01", end=None):
    data = yf.download(ticker, start=start, end=end)
    return data['Close']

def main():
    st.title("Stock Matrix Profile Analysis - Motif Juxtaposition")

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

        # Extract the subsequence for the user-defined date range
        subsequence = data[start_date:end_date]

        if len(subsequence) < 5:  # Arbitrary minimum length
            st.error(f"Selected date range is too short. Please select a longer period.")
            return

        # Calculate matrix profile
        mp = stumpy.stump(data, m=len(subsequence))

        # Find the top 3 closest matches
        top_matches_idx = np.argsort(mp[:, 0])[:3]

        # Plotting
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=False)
        
        # Plot queried date range
        axs[0].plot(subsequence.index, subsequence, label='Queried Range', color='blue')
        axs[0].set_title(f'Queried Range: {start_date.date()} to {end_date.date()}')
        axs[0].legend()

        # Plot top 3 matches
        for i, idx in enumerate(top_matches_idx, start=1):
            match_data = data.iloc[idx:idx+len(subsequence)]
            axs[i].plot(match_data.index, match_data, label=f'Match {i}', color='red')
            axs[i].set_title(f'Match {i}: {match_data.index[0].date()} to {match_data.index[-1].date()}')
            axs[i].legend()

        # Format x-axis to show dates
        for ax in axs:
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        st.pyplot(fig)

        # Display match details
        st.subheader("Match Details")
        for i, idx in enumerate(top_matches_idx, start=1):
            match_start = data.index[idx]
            match_end = data.index[idx + len(subsequence) - 1]
            st.write(f"Match {i}: {match_start.date()} to {match_end.date()}")

        # Calculate and display correlation coefficients
        st.subheader("Correlation with Queried Range")
        for i, idx in enumerate(top_matches_idx, start=1):
            match_data = data.iloc[idx:idx+len(subsequence)]
            correlation = subsequence.corr(match_data)
            st.write(f"Match {i} correlation: {correlation:.4f}")

if __name__ == "__main__":
    main()

