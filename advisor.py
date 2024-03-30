from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator


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
        df_list.append(df.tail(255 * 8))

    return df_list


def plot_mkt_cap(df: pd.DataFrame) -> px.treemap:
    """Takes in a DataFrame of stock information and plots market cap treemap

    Args:
    df: pandas DataFrame containing the following columns - ticker, sector, market_cap, colors, delta

    Returns:
    fig : Plotly express treemap figure object showing the market cap and color-coded
            according to the input "colors" column.
    """
    # Build and return the treemap figure
    fig = px.treemap(
        df,
        path=[px.Constant("all"), "sector", "ticker"],
        values="market_cap",
        color="colors",
        hover_data={"delta": ":.2p"},
    )
    return fig


def plot_returns(table: pd.DataFrame) -> plt.Figure:
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
        ax.plot(returns.index, returns[c], lw=3, alpha=0.8, label=c)

    # Add legend and y-axis label to the plot.
    ax.legend(loc="upper right", fontsize=12)
    ax.set_ylabel("daily returns")

    return fig


def portfolio_annualised_performance(
    weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray
) -> Tuple[float, float]:
    """
    Given the weights of the assets in the portfolio, their mean returns, and their covariance matrix,
    this function computes and returns the annualized performance of the portfolio in terms of its
    standard deviation (volatility) and expected returns.

    Args:
        weights (np.ndarray): The weights of the assets in the portfolio.
                              Each weight corresponds to the proportion of the investor's total
                              investment in the corresponding asset.

        mean_returns (np.ndarray): The mean (expected) returns of the assets.

        cov_matrix (np.ndarray): The covariance matrix of the asset returns. Each entry at the
                                 intersection of a row and a column represents the covariance
                                 between the returns of the asset corresponding to that row
                                 and the asset corresponding to that column.

    Returns:
        Tuple of portfolio volatility (standard deviation) and portfolio expected return, both annualized.
    """

    # Annualize portfolio returns by summing up the products of the mean returns and weights of each asset and then multiplying by 252
    # (number of trading days in a year)
    returns = np.sum(mean_returns * weights) * 252

    # Compute portfolio volatility (standard deviation) by dot multiplying the weights transpose and the dot product of covariance matrix
    # and weights. Then take the square root to get the standard deviation and multiply by square root of 252 to annualize it.
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    return std, returns


def random_portfolios(
    num_portfolios: int,
    num_weights: int,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate random portfolios and calculate their standard deviation, returns and Sharpe ratio.

    Args:
        num_portfolios (int): The number of random portfolios to generate.

        mean_returns (np.ndarray): The mean (expected) returns of the assets.

        cov_matrix (np.ndarray): The covariance matrix of the asset returns. Each entry at the
                                 intersection of a row and a column represents the covariance
                                 between the returns of the asset corresponding to that row
                                 and the asset corresponding to that column.

        risk_free_rate (float): The risk-free rate of return.

    Returns:
        Tuple of results and weights_record.

        results (np.ndarray): A 3D array with standard deviation, returns and Sharpe ratio of the portfolios.

        weights_record (List[np.ndarray]): A list with the weights of the assets in each portfolio.
    """
    # Initialize results array with zeros
    results = np.zeros((3, num_portfolios))

    # Initialize weights record list
    weights_record = []

    # Loop over the range of num_portfolios
    for i in np.arange(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_weights)

        # Normalize weights
        weights /= np.sum(weights)

        # Record weights
        weights_record.append(weights)

        # Calculate portfolio standard deviation and returns
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(
            weights, mean_returns, cov_matrix
        )

        # Store standard deviation, returns and Sharpe ratio in results
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev

    return results, weights_record


def display_simulated_ef_with_random(
    table: pd.DataFrame,
    mean_returns: List[float],
    cov_matrix: np.ndarray,
    num_portfolios: int,
    risk_free_rate: float,
) -> plt.Figure:
    """
    This function displays a simulated efficient frontier plot based on randomly generated portfolios with the specified parameters.

    Args:
    - mean_returns (List): A list of mean returns for each security or asset in the portfolio.
    - cov_matrix (ndarray): A covariance matrix for the securities or assets in the portfolio.
    - num_portfolios (int): The number of random portfolios to generate.
    - risk_free_rate (float): The risk-free rate of return.

    Returns:
    - fig (plt.Figure): A pyplot figure object
    """

    # Generate random portfolios using the specified parameters
    results, weights = random_portfolios(
        num_portfolios, len(mean_returns), mean_returns, cov_matrix, risk_free_rate
    )

    # Find the maximum Sharpe ratio portfolio and the portfolio with minimum volatility
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]

    # Create a DataFrame of the maximum Sharpe ratio allocation
    max_sharpe_allocation = pd.DataFrame(
        weights[max_sharpe_idx], index=table.columns, columns=["allocation"]
    )
    max_sharpe_allocation.allocation = [
        round(i * 100, 2) for i in max_sharpe_allocation.allocation
    ]
    max_sharpe_allocation = max_sharpe_allocation.T

    # Find index of the portfolio with minimum volatility
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]

    # Create a DataFrame of the minimum volatility allocation
    min_vol_allocation = pd.DataFrame(
        weights[min_vol_idx], index=table.columns, columns=["allocation"]
    )
    min_vol_allocation.allocation = [
        round(i * 100, 2) for i in min_vol_allocation.allocation
    ]
    min_vol_allocation = min_vol_allocation.T

    # Generate and plot the efficient frontier
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        results[0, :],
        results[1, :],
        c=results[2, :],
        cmap="YlGnBu",
        marker="o",
        s=10,
        alpha=0.3,
    )
    ax.scatter(sdp, rp, marker="*", color="r", s=500, label="Maximum Sharpe ratio")
    ax.scatter(
        sdp_min, rp_min, marker="*", color="g", s=500, label="Minimum volatility"
    )
    ax.set_title("Simulated Portfolio Optimization based on Efficient Frontier")
    ax.set_xlabel("Annual volatility")
    ax.set_ylabel("Annual returns")
    ax.legend(labelspacing=0.8)

    return fig, {
        "Annualised Return (efficient portfolio)": round(rp, 2),
        "Annualised Volatility (efficient portfolio)": round(sdp, 2),
        "Max Sharpe Allocation": max_sharpe_allocation,
        "Max Sharpe Allocation in Percentile": max_sharpe_allocation.div(
            max_sharpe_allocation.sum(axis=1), axis=0
        ),
        "Annualised Return (min variance portfolio)": round(rp_min, 2),
        "Annualised Volatility (min variance portfolio)": round(sdp_min, 2),
        "Min Volatility Allocation": min_vol_allocation,
        "Min Volatility Allocation in Percentile": min_vol_allocation.div(
            min_vol_allocation.sum(axis=1), axis=0
        ),
    }


def entry_strategy(
    start_date="2013-01-01",
    end_date="2019-12-6",
    tickers="AAPL",
    thresholds="10, 20, 30",
    buy_threshold=20,
    sell_threshold=80,
):
    rsi_threshold_1 = int(thresholds.split(",")[0])
    rsi_threshold_2 = int(thresholds.split(",")[1])
    rsi_threshold_3 = int(thresholds.split(",")[2])

    # Conditional Buy/Sell => Signals
    stock = yf.download(tickers, start_date, end_date)
    rsiData1 = RSIIndicator(stock["Close"], rsi_threshold_1, True)
    rsiData2 = RSIIndicator(stock["Close"], rsi_threshold_2, True)
    rsiData3 = RSIIndicator(stock["Close"], rsi_threshold_3, True)

    # Conditional Buy/Sell => Signals
    conditionalBuy1 = np.where(rsiData1.rsi() < buy_threshold, stock["Close"], np.nan)
    conditionalSell1 = np.where(rsiData1.rsi() > sell_threshold, stock["Close"], np.nan)
    conditionalBuy2 = np.where(rsiData2.rsi() < buy_threshold, stock["Close"], np.nan)
    conditionalSell2 = np.where(rsiData2.rsi() > sell_threshold, stock["Close"], np.nan)
    conditionalBuy3 = np.where(rsiData3.rsi() < buy_threshold, stock["Close"], np.nan)
    conditionalSell3 = np.where(rsiData3.rsi() > sell_threshold, stock["Close"], np.nan)

    # RSI Construction
    stock["RSI1"] = rsiData1.rsi()
    stock["RSI2"] = rsiData2.rsi()
    stock["RSI3"] = rsiData3.rsi()
    stock["RSI1_Buy"] = conditionalBuy1
    stock["RSI1_Sell"] = conditionalSell1
    stock["RSI2_Buy"] = conditionalBuy2
    stock["RSI2_Sell"] = conditionalSell2
    stock["RSI3_Buy"] = conditionalBuy3
    stock["RSI3_Sell"] = conditionalSell3

    strategy = "RSI"
    title = f"Close Price Buy/Sell Signals using WYN Entry Strategy"

    fig, axs = plt.subplots(2, sharex=True, figsize=(13, 9))

    if not stock["RSI1_Buy"].isnull().all():
        axs[0].scatter(
            stock.index,
            stock["RSI1_Buy"],
            color="green",
            label="Buy Signal 1",
            marker="^",
            alpha=1,
        )
    if not stock["RSI1_Sell"].isnull().all():
        axs[0].scatter(
            stock.index,
            stock["RSI1_Sell"],
            color="red",
            label="Sell Signal 1",
            marker="v",
            alpha=1,
        )
    axs[0].plot(stock["Close"], label="Close Price", color="blue", alpha=0.35)

    if not stock["RSI2_Buy"].isnull().all():
        axs[0].scatter(
            stock.index,
            stock["RSI2_Buy"],
            color="blue",
            label="Buy Signal 2",
            marker="^",
            alpha=1,
        )
    if not stock["RSI2_Sell"].isnull().all():
        axs[0].scatter(
            stock.index,
            stock["RSI2_Sell"],
            color="purple",
            label="Sell Signal 2",
            marker="v",
            alpha=1,
        )
    axs[0].plot(stock["Close"], label="Close Price", color="blue", alpha=0.35)

    if not stock["RSI3_Buy"].isnull().all():
        axs[0].scatter(
            stock.index,
            stock["RSI3_Buy"],
            color="cyan",
            label="Buy Signal 3",
            marker="^",
            alpha=1,
        )
    if not stock["RSI3_Sell"].isnull().all():
        axs[0].scatter(
            stock.index,
            stock["RSI3_Sell"],
            color="pink",
            label="Sell Signal 3",
            marker="v",
            alpha=1,
        )
    axs[0].plot(stock["Close"], label="Close Price", color="blue", alpha=0.35)

    # plt.xticks(rotation=45)
    axs[0].set_title(title)
    axs[0].set_xlabel("Date", fontsize=18)
    axs[0].set_ylabel("Close Price", fontsize=18)
    axs[0].legend(loc="upper left")
    axs[0].grid()

    axs[1].plot(stock["RSI1"], label="RSI", color="green")
    axs[1].plot(stock["RSI2"], label="RSI", color="blue")
    axs[1].plot(stock["RSI3"], label="RSI", color="red")

    return fig


def entry_strategy_plotly(
    start_date="2013-01-01",
    end_date="2019-12-6",
    tickers="AAPL",
    thresholds="10, 20, 30",
    buy_threshold=20,
    sell_threshold=80,
):
    rsi_threshold_1 = int(thresholds.split(",")[0])
    rsi_threshold_2 = int(thresholds.split(",")[1])
    rsi_threshold_3 = int(thresholds.split(",")[2])

    # Conditional Buy/Sell => Signals
    stock = yf.download(tickers, start_date, end_date)
    rsiData1 = RSIIndicator(stock["Close"], rsi_threshold_1, True)
    rsiData2 = RSIIndicator(stock["Close"], rsi_threshold_2, True)
    rsiData3 = RSIIndicator(stock["Close"], rsi_threshold_3, True)

    # Conditional Buy/Sell => Signals
    stock["RSI1_Buy"] = np.where(rsiData1.rsi() < buy_threshold, stock["Close"], np.nan)
    stock["RSI1_Sell"] = np.where(rsiData1.rsi() > sell_threshold, stock["Close"], np.nan)
    stock["RSI2_Buy"] = np.where(rsiData2.rsi() < buy_threshold, stock["Close"], np.nan)
    stock["RSI2_Sell"] = np.where(rsiData2.rsi() > sell_threshold, stock["Close"], np.nan)
    stock["RSI3_Buy"] = np.where(rsiData3.rsi() < buy_threshold, stock["Close"], np.nan)
    stock["RSI3_Sell"] = np.where(rsiData3.rsi() > sell_threshold, stock["Close"], np.nan)

    # Create a subplot figure with secondary Y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for close prices
    fig.add_trace(go.Scatter(x=stock.index, y=stock["Close"], name="Close Price", line=dict(color='blue', width=0.5)), secondary_y=False)

    # Add traces for buy and sell signals
    fig.add_trace(go.Scatter(x=stock.index, y=stock["RSI1_Buy"], mode='markers', name='Buy Signal (light)', marker=dict(color='green', size=6, symbol='triangle-up')), secondary_y=False)
    fig.add_trace(go.Scatter(x=stock.index, y=stock["RSI1_Sell"], mode='markers', name='Sell Signal (light)', marker=dict(color='red', size=6, symbol='triangle-down')), secondary_y=False)
    fig.add_trace(go.Scatter(x=stock.index, y=stock["RSI2_Buy"], mode='markers', name='Buy Signal (medium)', marker=dict(color='blue', size=6, symbol='triangle-up')), secondary_y=False)
    fig.add_trace(go.Scatter(x=stock.index, y=stock["RSI2_Sell"], mode='markers', name='Sell Signal (medium)', marker=dict(color='purple', size=6, symbol='triangle-down')), secondary_y=False)
    fig.add_trace(go.Scatter(x=stock.index, y=stock["RSI3_Buy"], mode='markers', name='Buy Signal (heavy)', marker=dict(color='cyan', size=6, symbol='triangle-up')), secondary_y=False)
    fig.add_trace(go.Scatter(x=stock.index, y=stock["RSI3_Sell"], mode='markers', name='Sell Signal (heavy)', marker=dict(color='pink', size=6, symbol='triangle-down')), secondary_y=False)

    # Add traces for RSI
    # fig.add_trace(go.Scatter(x=stock.index, y=rsiData1.rsi(), name="RSI 1", line=dict(color='green')), secondary_y=True)
    # fig.add_trace(go.Scatter(x=stock.index, y=rsiData2.rsi(), name="RSI 2", line=dict(color='blue')), secondary_y=True)
    # fig.add_trace(go.Scatter(x=stock.index, y=rsiData3.rsi(), name="RSI 3", line=dict(color='red')), secondary_y=True)

    # Set figure title, and axis titles
    fig.update_layout(title_text='Close Price Buy/Sell Signals using WYN Entry Strategy')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='<b>Close Price</b>', secondary_y=False)
    fig.update_yaxes(title_text='<b>RSI</b>', secondary_y=True)

    return fig


def get_stock_info(ticker: str) -> dict:
    # Get More Data:
    tck = yf.Ticker(ticker)
    ALL_DATA = {
        'get stock info': tck.info,
        'get historical market data': tck.history(period="max"),
        'show actions (dividends, splits)': tck.actions,
        'show dividends': tck.dividends,
        'show splits': tck.splits,
        'show financials': [tck.financials, tck.quarterly_financials],
        'show balance sheet': [tck.balance_sheet, tck.quarterly_balance_sheet],
        'show cashflow': [tck.cashflow, tck.quarterly_cashflow],
        # 'show earnings': [tck.earnings, tck.quarterly_earnings],
        # 'show sustainability': tck.sustainability,
        # 'show analysts recommendations': tck.recommendations,
        # 'show next event (earnings, etc)': tck.calendar
    }

    return ALL_DATA