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

# `stocks` is a string of comma-separated stock symbols
stocks = stocks.split(', ')

# Get the list of stocks data using the `download_stocks` function
list_of_stocks = download_stocks(stocks)

# Create a DataFrame object from the closing prices of all stocks
table = pd.DataFrame([list_of_stocks[j]['Close'] for j in range(len(list_of_stocks))]).transpose()

# Set the column names to be the stocks symbols
table.columns = stocks


def _plot_returns() -> plt.Figure:
    """
    This function plots the daily returns of each stock contained in the DataFrame `table`.

    Returns:
        fig: A `Figure` instance representing the entire figure.
    """
    # Calculate the daily percentage change of all stocks using the `pct_change` method.
    returns = table.pct_change()

    # Plot each stock's daily returns on the same graph using a for loop and the `plot` method of pyplot object.
    fig, ax = plt.subplots(figsize=(14, 7))
    for c in returns.columns.values:
        ax.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)

    # Add legend and y-axis label to the plot.
    ax.legend(loc='upper right', fontsize=12)
    ax.set_ylabel('daily returns')
    
    return fig


return_figure = _plot_returns()
st.write(f"""
    Plot daily returns of the stocks selected: {stocks}
""")
st.pyplot(return_figure)