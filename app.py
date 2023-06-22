from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# Set up Title
st.set_page_config(page_title="WYN AI", page_icon=":robot_face:")
st.markdown(
    f"""
        <h1 style='text-align: center;'>W.Y.N. Artificial Intelligence 🤖</h1>
    """,
    unsafe_allow_html=True,
)

# Set up Sidebar
st.sidebar.title("Sidebar")
stocks = st.sidebar.text_input(
    "Enter stocks (sep. by comma)",
    "AAPL, META, TSLA, AMZN, AMD, NVDA, TSM, MSFT, GOOGL, NFLX",
)
st.sidebar.write("All stocks held by Yiqiao Yin.")
st.sidebar.markdown(
    "© [Yiqiao Yin](https://www.y-yin.io/) | [LinkedIn](https://www.linkedin.com/in/yiqiaoyin/) | [YouTube](https://youtube.com/YiqiaoYin/)"
)


# Function: download stocks
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
stocks = stocks.split(", ")

# Get the list of stocks data using the `download_stocks` function
list_of_stocks = download_stocks(stocks)
st.success("Downloading latest stock data successfully!")

# Create a DataFrame object from the closing prices of all stocks
table = pd.DataFrame(
    [list_of_stocks[j]["Close"] for j in range(len(list_of_stocks))]
).transpose()

# Set the column names to be the stocks symbols
table.columns = stocks

# Get info
tickers = []
deltas = []
sectors = []
market_caps = []

for ticker in stocks:
    try:
        ## create Ticker object
        stock = yf.Ticker(ticker)
        tickers.append(ticker)

        ## download info
        info = stock.info

        ## download sector
        sectors.append(info["sector"])

        ## download daily stock prices for 2 days
        hist = stock.history("2d")

        ## calculate change in stock price (from a trading day ago)
        deltas.append((hist["Close"][1] - hist["Close"][0]) / hist["Close"][0])

        ## calculate market cap
        market_caps.append(info["sharesOutstanding"] * info["previousClose"])

        ## add print statement to ensure code is running
        print(f"downloaded {ticker}")
    except Exception as e:
        print(e)

# create dataframe for market cap
df_for_mkt_cap = pd.DataFrame(
    {
        "ticker": tickers,
        "sector": sectors,
        "delta": deltas,
        "market_cap": market_caps,
    }
)
color_bin = [-1, -0.02, -0.01, 0, 0.01, 0.02, 1]
df_for_mkt_cap["colors"] = pd.cut(
    df_for_mkt_cap["delta"],
    bins=color_bin,
    labels=["grey", "indianred", "lightpink", "lightgreen", "lime", "green"],
)


# Function: plot market cap heatmap
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


st.markdown(
    f"""
        <h4 style='text-align: left;'>Market Cap Heatmap</h4>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    r"""
    I trade large cap stocks first, so I visualize data using market cap heatmap.
    The philosophy comes from the famous [Fama-French 3 Factor](https://en.wikipedia.org/wiki/Fama%E2%80%93French_three-factor_model)
    model and the market cap is captured using the 2nd factor 'SMB'.
    """
)

# plot heatmap
fig_market_cap_heatmap = plot_mkt_cap(df=df_for_mkt_cap)
st.plotly_chart(fig_market_cap_heatmap)


# Function: plot returns
def plot_returns() -> plt.Figure:
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


st.markdown(
    f"""
        <h4 style='text-align: left;'>Time Series Plot of Daily Returns</h4>
    """,
    unsafe_allow_html=True,
)


return_figure = plot_returns()
st.write(
    f"""
    Plot daily returns of the stocks selected: {stocks}
"""
)
st.pyplot(return_figure)


# Function: annual performance
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


# Function: random portfolio
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


# Function: display simulated efficient frontier
def display_simulated_ef_with_random(
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
    ax.set_xlabel("annualised volatility")
    ax.set_ylabel("annualised returns")
    ax.legend(labelspacing=0.8)

    return fig, {
        "Annualised Return": round(rp, 2),
        "Annualised Volatility": round(sdp, 2),
        "Max Sharpe Allocation": max_sharpe_allocation,
        "Max Sharpe Allocation in Percentile": max_sharpe_allocation.div(
            max_sharpe_allocation.sum(axis=1), axis=0
        ),
        "Annualised Return": round(rp_min, 2),
        "Annualised Volatility": round(sdp_min, 2),
        "Min Volatility Allocation": min_vol_allocation,
        "Min Volatility Allocation in Percentile": min_vol_allocation.div(
            min_vol_allocation.sum(axis=1), axis=0
        ),
    }

st.markdown(
    f"""
        <h4 style='text-align: center;'>Modern Portfolio Theory</h4>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    r"""
    Among the large cap stocks, I trade a long run reversal strategy, and hence the visualization of returns from time-series plot and MPT.
    The philosophy comes from the famous [Carhart 4-Factor](https://en.wikipedia.org/wiki/Carhart_four-factor_model)
    model and the reversal strategy is captured using the 4th factor 'UMD'. If interested, one can 
    trace the algorithm proposed from this [paper](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1997.tb03808.x).
    """
)
st.warning("What is Modern Portolio Theory?")
st.markdown(r"""
    The efficient frontier is a concept in Modern Portfolio Theory. It is the set of optimal portfolios that offer the highest expected return for a defined level of risk or the lowest risk for a given level of expected return.

    Mathematically, the efficient frontier is the solution to the following optimization problem:

    Minimize:
    $$ \sigma_p = \sqrt{w^T\Sigma w} $$
    Subject to:
    $$ R_p = w^T \mu $$

    Where:

    - $w$ is a vector of portfolio weights.

    - $\Sigma$ is the covariance matrix of asset returns.

    - $\mu$ is the vector of expected asset returns.

    - $\sigma_p$ is the portfolio standard deviation (risk).

    - $R_p$ is the portfolio expected return.

    Here, $w^T$ denotes the transpose of $w$. The symbol $\sqrt{w^T\Sigma w}$ represents the standard deviation (volatility) of the portfolio returns, which is a measure of risk. The equation $R_p = w^T \mu$ states that the expected return of the portfolio should be equal to the portfolio weights times the expected returns of the individual assets.

    Note: This is the simplified version of the efficient frontier. In practice, one might consider additional constraints such as no short-selling (i.e., weights must be non-negative) or a requirement that all weights sum to one.
""")

returns = table.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# num_portfolios
num_portfolios = st.sidebar.select_slider(
    "Select total number of portfolios to similuate",
    value=5000,
    options=[10, 100, 1000, 5000, 10000],
)

# risk_free_rate
risk_free_rate = st.sidebar.select_slider(
    "Select simulated risk-free rate",
    value=0.01,
    options=[0.0001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
)

eff_front_figure, some_data = display_simulated_ef_with_random(
    mean_returns, cov_matrix, num_portfolios, risk_free_rate
)


st.markdown(
    f"""
        <h4 style='text-align: center;'>Efficient Portfolio:</h4>
    """,
    unsafe_allow_html=True,
)
st.write(f"Annualised Return: {some_data['Annualised Return']}")
st.write(f"Annualised Volatility: {some_data['Annualised Volatility']}")
st.write(f"Max Sharpe Allocation:")
st.table(some_data["Max Sharpe Allocation"])
st.write(f"Max Sharpe Allocation in Percentile:")
st.table(some_data["Max Sharpe Allocation in Percentile"])
st.markdown(
    f"""
        <h4 style='text-align: center;'>Min Variance Portfolio:</h4>
    """,
    unsafe_allow_html=True,
)
st.write(f"Annualised Return: {some_data['Annualised Return']}")
st.write(f"Annualised Volatility: {some_data['Annualised Volatility']}")
st.write(f"Min Volatility Allocation:")
st.table(some_data["Min Volatility Allocation"])
st.write(f"Min Volatility Allocation in Percentile:")
st.table(some_data["Min Volatility Allocation in Percentile"])
st.pyplot(eff_front_figure)


st.warning(
    "Note (1): The time of entry is a trade secret and decided by Mr. Yin based on experience."
)
st.warning(
    "Note (2): Though stocks are presented above, the weights decided by Mr. Yin is drastically different from the above allocation."
)
st.warning(
    "Note (3): The initial stock pool construction is also unreproducible. Mr. Yin mostly pick stocks from large cap brackets but occasionally break his own rules."
)
st.markdown(
    f"""
        <h6 style='text-align: left;'>Copyright © 2010-2023 Present Yiqiao Yin</h6>
    """,
    unsafe_allow_html=True,
)
