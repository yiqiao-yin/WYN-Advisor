import streamlit as st
import yfinance as yf
import pandas as pd
from typing import List
import matplotlib.pyplot as plt

# Set up Title
st.set_page_config(page_title="WYN AI", page_icon=":robot_face:")
st.markdown(
    f"""
        <h1 style='text-align: center;'>W.Y.N. Artificial Intelligence😬</h1>
    """,
    unsafe_allow_html=True,
)

# Set up Sidebar
st.sidebar.title("Sidebar")
stocks = st.sidebar.text_input('Enter stocks (sep. by comma)', 'AAPL, MSFT, NVDA, TSLA')

# Functions
def download_stocks(tickers: List[str]) -> List[pd.DataFrame]:
    """
    Downloads stock data from Yahoo Finance.

    Args:
        tickers: A list of stock tickers.

    Returns:
        A list of Pandas DataFrames, one for each stock.
    """

    # Create a list of DataFrames.
    df_list = []

    # Iterate over the tickers.
    for ticker in tickers:
        # Download the stock data.
        df = yf.download(ticker)

        # Add the DataFrame to the list.
        df_list.append(df.tail(400))

    return df_list

stocks = stocks.split(', ')
list_of_stocks = download_stocks(stocks)
table = pd.DataFrame([list_of_stocks[j]['Close'] for j in range(len(list_of_stocks))]).transpose()
table.columns = stocks

returns = table.pct_change()
def _plot_returns():
    plt.figure(figsize=(14, 7))
    for c in returns.columns.values:
        plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
    plt.legend(loc='upper right', fontsize=12)
    plt.ylabel('daily returns')

st.pyplot(_plot_returns)