#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
import stumpy
from scipy.spatial.distance import euclidean
import requests
from datetime import datetime
import yfinance as yf

# Retrieve the FRED API key from Streamlit secrets
FRED_API_KEY = st.secrets["FRED_API_KEY"]

def get_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)['Close']

def calculate_cumulative_change(data):
    return (data.pct_change() + 1).cumprod() - 1

def get_fred_data_with_preceding(api_key, series_id, start_date, end_date):
    # Try to get data within the specified range
    data = get_fred_data(api_key, series_id, start_date, end_date)
    
    if not data:
        # If no data in the range, fetch the earliest data point before the start date
        preceding_url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_end={start_date}&sort_order=desc&limit=1"
        response = requests.get(preceding_url)
        
        if response.status_code == 200:
            data = response.json()
            if 'observations' in data and len(data['observations']) > 0:
                obs = data['observations'][0]
                preceding_date = datetime.strptime(obs['date'], '%Y-%m-%d')
                preceding_value = float(obs['value'])
                return [(preceding_date, preceding_value)], preceding_date
    
    return data, None

def get_fred_data(api_key, series_id, start_date, end_date):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if 'observations' in data:
            return [(datetime.strptime(obs['date'], '%Y-%m-%d'), float(obs['value'])) for obs in data['observations'] if obs['value'] != '.']
    
    return None

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
        try:
            # Fetch stock data from 2005 to the end date plus 30 days
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

            # Display match details
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

            # Fetch FRED data
            federal_debt_data, debt_preceding_date = get_fred_data_with_preceding(FRED_API_KEY, "GFDEBTN", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            unrate_data, unrate_preceding_date = get_fred_data_with_preceding(FRED_API_KEY, "UNRATE", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            # Plotting
            fig = plt.figure(figsize=(15, 20))
            gs = GridSpec(4, 2, width_ratios=[3, 1], height_ratios=[1, 1, 1, 1], figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[2, :])
            ax4 = fig.add_subplot(gs[3, 0])
            ax5 = fig.add_subplot(gs[3, 1])
            
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
            ax1.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
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
            ax2.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

            # Plot 3: The next 30 days
            ax3.set_title(f'Cumulative Change: 30 Days Beyond Matched Motifs - {ticker}', fontsize=14)
            ax3.set_xlabel('Days Beyond Motif', fontsize=12)
            ax3.set_ylabel('Cumulative Percent Change', fontsize=12)

            # Plot matches for the next 30 days
            positive_returns = 0
            for i, idx in enumerate(top_matches_idx):
                match_end_idx = idx + len(subsequence)
                next_30_days = data.iloc[match_end_idx:match_end_idx+30]
                if len(next_30_days) > 0:
                    cum_change = calculate_cumulative_change(next_30_days)
                    match_start, match_end = match_details[i]
                    label = f'Match {i+1}: After {match_end.strftime("%Y-%m-%d")}'
                    ax3.plot(range(len(cum_change)), cum_change, label=label, color=colors[i])
                    if cum_change.iloc[-1] > 0:
                        positive_returns += 1

            ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
            ax3.set_xlim(0, 30)
            ax3.set_xticks(range(0, 31, 5))

            # Plot 4: National Debt
            if federal_debt_data:
                debt_df = pd.DataFrame(federal_debt_data, columns=['date', 'value']).set_index('date')
                ax4.plot(debt_df.index, debt_df['value'], label='National Debt')
                ax4.set_title('National Debt', fontsize=14)
                ax4.set_xlabel('Date', fontsize=12)
                ax4.set_ylabel('Debt ($ million)', fontsize=12)
                ax4.legend(fontsize=10)
                ax4.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Plot 5: Unemployment Rate
            if unrate_data:
                unrate_df = pd.DataFrame(unrate_data, columns=['date', 'value']).set_index('date')
                ax5.plot(unrate_df.index, unrate_df['value'], label='Unemployment Rate')
                ax5.set_title('Unemployment Rate', fontsize=14)
                ax5.set_xlabel('Date', fontsize=12)
                ax5.set_ylabel('Rate (%)', fontsize=12)
                ax5.legend(fontsize=10)
                ax5.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

            plt.tight_layout()
            st.pyplot(fig)

            # Display FRED data details
            st.subheader("Contextual Statistics")
            st.write(f"For the date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            if federal_debt_data:
                earliest_debt = federal_debt_data[0]
                st.write(f"**National Debt:** ${earliest_debt[1]:,.2f} million (Earliest data from {earliest_debt[0].strftime('%Y-%m-%d')})")
            else:
                st.error("Failed to retrieve federal debt data.")

            if unrate_data:
                earliest_unrate = unrate_data[0]
                st.write(f"**Unemployment Rate:** {earliest_unrate[1]:.2f}% (Earliest data from {earliest_unrate[0].strftime('%Y-%m-%d')})")
            else:
                st.error("Failed to retrieve unemployment rate data.")

            # Add summary table
            st.subheader("Summary of 30-Day Returns Beyond Matched Patterns")
            total_matches = len(top_matches_idx)
            positive_percentage = (positive_returns / total_matches) * 100 if total_matches > 0 else 0
            
            summary_data = {
                "Metric": ["Positive Returns", "Total Matches", "Percentage Positive"],
                "Value": [f"{positive_returns}/{total_matches}", f"{total_matches}", f"{positive_percentage:.2f}%"]
            }
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your input and try again.")

if __name__ == "__main__":
    main()

