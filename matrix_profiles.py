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
from scipy.spatial.distance import euclidean

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

        # Convert start_date and end_date to pandas Timestamp
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

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
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot queried date range
        ax.plot(subsequence.index, subsequence, label='Queried Range', color='blue', linestyle=':', linewidth=2)

        # Plot top 3 matches
        colors = ['red', 'green', 'orange']
        for i, idx in enumerate(top_matches_idx):
            match_data = data.iloc[idx:idx+len(subsequence)]
            # Align the match data with the queried range for easier comparison
            aligned_dates = pd.date_range(start=subsequence.index[0], periods=len(match_data), freq='D')
            ax.plot(aligned_dates, match_data.values, label=f'Match {i+1}', color=colors[i])

        ax.set_title(f'Queried Range and Top Matches - {ticker}')
        ax.legend()
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        st.pyplot(fig)

        # Display match details
        st.subheader("Match Details")
        for i, idx in enumerate(top_matches_idx, start=1):
            match_start = data.index[idx]
            match_end = data.index[idx + len(subsequence) - 1]
            st.write(f"Match {i}: {match_start.strftime('%Y-%m-%d')} to {match_end.strftime('%Y-%m-%d')}")

        # Calculate and display Euclidean distances
        st.subheader("Euclidean Distances from Queried Range")
        
        # Normalize the subsequence for fair comparison
        subsequence_normalized = (subsequence - subsequence.mean()) / subsequence.std()
        
        for i, idx in enumerate(top_matches_idx, start=1):
            match_data = data.iloc[idx:idx+len(subsequence)]
            # Normalize the match data
            match_data_normalized = (match_data - match_data.mean()) / match_data.std()
            # Calculate Euclidean distance
            distance = euclidean(subsequence_normalized, match_data_normalized)
            st.write(f"Match {i} distance: {distance:.4f}")

if __name__ == "__main__":
    main()

