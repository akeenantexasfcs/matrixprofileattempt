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
        # Fetch data from 2005 to the end date plus 30 days
        data = get_data(ticker, start="2005-01-01", end=end_date + pd.Timedelta(days=30))

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

        # Plotting
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 24))  # Increased figure height for 3 subplots
        
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

        ax1.set_title(f'Queried Range and Top Matches - {ticker}', fontsize=14)
        ax1.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Cumulative change
        window_length = np.arange(len(subsequence))
        subsequence_cum_change = calculate_cumulative_change(subsequence)
        ax2.plot(window_length, subsequence_cum_change, 
                 label='Queried Range', color='blue', linestyle=':', linewidth=2)

        for i, idx in enumerate(top_matches_idx):
            match_data = data.iloc[idx:idx+len(subsequence)]
            match_data = match_data[:len(subsequence)]  # Ensure same length
            match_cum_change = calculate_cumulative_change(match_data)
            match_start, match_end = match_details[i]
            label = f'Match {i+1}: {match_start.strftime("%Y-%m-%d")} to {match_end.strftime("%Y-%m-%d")}'
            ax2.plot(window_length, match_cum_change, label=label, color=colors[i])

        ax2.set_title(f'Cumulative Percent Change - {ticker}', fontsize=14)
        ax2.set_xlabel('Window Length (Days)', fontsize=12)
        ax2.set_ylabel('Cumulative Percent Change', fontsize=12)
        ax2.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

        # Plot 3: The next 30 days
        ax3.set_title(f'The Next 30 Days - {ticker}', fontsize=14)
        ax3.set_xlabel('Window Length (Days)', fontsize=12)
        ax3.set_ylabel('Cumulative Percent Change', fontsize=12)

        # Plot queried range
        ax3.plot(window_length, subsequence_cum_change, 
                 label='Queried Range', color='blue', linestyle=':', linewidth=2)

        # Plot vertical bar at the end of window length
        ax3.axvline(x=len(window_length)-1, color='black', linestyle='--', label='End of Window')

        # Plot matches including next 30 days
        for i, idx in enumerate(top_matches_idx):
            match_data = data.iloc[idx:idx+len(subsequence)+30]
            match_cum_change = calculate_cumulative_change(match_data)
            match_start, match_end = match_details[i]
            label = f'Match {i+1}: {match_start.strftime("%Y-%m-%d")} to {match_end.strftime("%Y-%m-%d")}'
            
            # Plot up to the end of window length
            ax3.plot(window_length, match_cum_change[:len(window_length)], label=label, color=colors[i])
            
            # Plot the next 30 days
            if len(match_cum_change) > len(window_length):
                extra_days = len(match_cum_change) - len(window_length)
                ax3.plot(np.arange(len(window_length)-1, len(window_length)-1+extra_days), 
                         match_cum_change[len(window_length)-1:], color=colors[i], linestyle='--')

        # Add secondary x-axis for the days beyond the window length
        ax3_secondary = ax3.twiny()
        ax3_secondary.set_xlim(ax3.get_xlim())
        ax3_secondary.set_xticks(np.arange(len(window_length), len(window_length)+30, 10))
        ax3_secondary.set_xticklabels(['1', '10', '20', '30'])
        ax3_secondary.set_xlabel('Days Beyond Window', fontsize=10)

        ax3.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        st.pyplot(fig)

        # Add explanations for the graphs
        st.info("Graph 1: Shows the original time series for the queried range and matches.")
        st.info("Graph 2: Displays the cumulative percent change over the window length for the queried range and matches.")
        st.info("Graph 3: Illustrates the cumulative percent change for the queried range and matches, including performance for the next 30 days (if available) beyond the queried range.")

if __name__ == "__main__":
    main()

