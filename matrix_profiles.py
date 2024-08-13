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

def calculate_cumulative_change(series):
    return (series / series.iloc[0] - 1) * 100

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

        # Display match details first
        st.subheader("Match Details")
        st.write(f"Queried Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        match_details = []
        for i, idx in enumerate(top_matches_idx, start=1):
            match_start = data.index[idx]
            match_end = data.index[idx + len(subsequence) - 1]
            match_details.append((match_start, match_end))
            st.write(f"Match {i}: {match_start.strftime('%Y-%m-%d')} to {match_end.strftime('%Y-%m-%d')}")

        # Calculate and display Euclidean distances
        st.subheader("Euclidean Distances from Queried Range")
        
        subsequence_normalized = (subsequence - subsequence.mean()) / subsequence.std()
        
        for i, idx in enumerate(top_matches_idx, start=1):
            match_data = data.iloc[idx:idx+len(subsequence)]
            match_data = match_data[:len(subsequence)]  # Ensure same length
            match_data_normalized = (match_data - match_data.mean()) / match_data.std()
            distance = euclidean(subsequence_normalized, match_data_normalized)
            st.write(f"Match {i} distance: {distance:.4f}")

        # Add notification about x-axis
        st.info("On the first graph, the x-axis represents the date range for the queried data only.")

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot 1: Original time series
        date_range = pd.date_range(start=subsequence.index[0], periods=len(subsequence), freq='D')
        ax1.plot(date_range, subsequence, label='Queried Range', color='blue', linestyle=':', linewidth=2)

        colors = ['red', 'green', 'orange']
        for i, idx in enumerate(top_matches_idx):
            match_data = data.iloc[idx:idx+len(subsequence)]
            match_data = match_data[:len(subsequence)]  # Ensure same length
            match_start, match_end = match_details[i]
            label = f'Match {i+1}: {match_start.strftime("%Y-%m-%d")} to {match_end.strftime("%Y-%m-%d")}'
            ax1.plot(date_range, match_data.values, label=label, color=colors[i])

        ax1.set_title(f'Queried Range and Top Matches - {ticker}')
        ax1.legend(fontsize='x-small')
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Cumulative change
        subsequence_cum_change = calculate_cumulative_change(subsequence)
        ax2.plot(date_range, subsequence_cum_change, 
                 label='Queried Range', color='blue', linestyle=':', linewidth=2)

        for i, idx in enumerate(top_matches_idx):
            match_data = data.iloc[idx:idx+len(subsequence)]
            match_data = match_data[:len(subsequence)]  # Ensure same length
            match_cum_change = calculate_cumulative_change(match_data)
            match_start, match_end = match_details[i]
            label = f'Match {i+1}: {match_start.strftime("%Y-%m-%d")} to {match_end.strftime("%Y-%m-%d")}'
            ax2.plot(date_range, match_cum_change, label=label, color=colors[i])

        ax2.set_title(f'Cumulative Percent Change - {ticker}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Percent Change')
        ax2.legend(fontsize='x-small')
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()

