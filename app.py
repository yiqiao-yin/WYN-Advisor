from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator

from advisor import *
from advisor import entry_strategy_plotly, get_stock_info
from my_openai import call_chatcompletion

# Set the OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set up Title
st.set_page_config(page_title="WYN AI", page_icon="💹", layout="wide")

st.markdown(
    f"""
        <h1 style='text-align: center;'>W.Y.N. Artificial Intelligence 🤖</h1>
    """,
    unsafe_allow_html=True,
)

# Set up Sidebar
st.sidebar.title("Sidebar")
option = st.sidebar.selectbox(
    "Which strategy do you want to see?",
    ("--", "Portfolio Management", "Entry Strategy"),
)
# st.sidebar.write("You selected:", option)

# More sidebar
if option == "Portfolio Management":
    stocks = st.sidebar.text_input(
        "Enter stocks (sep. by comma and space, e.g. ', ')",
        "AAPL, META, TSLA, AMZN, AMD, NVDA, TSM, MSFT, GOOGL, NFLX",
    )
    # st.sidebar.write("Discalimer: The first 10 are held by Yiqiao Yin.")
    start_datetime = st.sidebar.date_input("Start date", datetime(2012, 1, 1))
    end_datetime = st.sidebar.date_input("End date", datetime.today())
    st.sidebar.write(
        "Range selected: from ",
        str(start_datetime).split(" ")[0],
        " to ",
        str(end_datetime).split(" ")[0],
    )
    num_portfolios = st.sidebar.select_slider(
        "Select total number of portfolios to similuate",
        value=5000,
        options=[10, 100, 200, 500, 1000, 2000, 5000, 8000, 10000],
    )
    risk_free_rate = st.sidebar.select_slider(
        "Select simulated risk-free rate",
        value=0.01,
        options=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05],
    )
    with st.sidebar:
        with st.form(key="my_form"):
            submit_button = st.form_submit_button(label="Submit!")
elif option == "Entry Strategy":
    start_datetime = st.sidebar.date_input("Start date", datetime(2010, 1, 1))
    end_datetime = st.sidebar.date_input("End date", datetime.today())
    this_stock = st.sidebar.text_input("Enter a ticker of a stock you like:", "AAPL")
    rsi_thresholds = st.sidebar.text_input(
        "Enter 3 integers for number of past days to construct RSI (sep. by comma and space):",
        "12, 26, 50",
    )
    thresholds_values = st.sidebar.slider(
        "Select a range of values to infer margin of error", 0.0, 100.0, (25.0, 75.0)
    )
    with st.sidebar:
        submit_button = st.form_submit_button(label="Submit!")
else:
    st.sidebar.write("Please pick an option from above.")

# Credits
st.sidebar.markdown(
    "© [Yiqiao Yin](https://www.y-yin.io/) | [LinkedIn](https://www.linkedin.com/in/yiqiaoyin/) | [YouTube](https://youtube.com/YiqiaoYin/)"
)


# Content starts here
if option == "Portfolio Management":
    if submit_button:
        col1, col2 = st.columns(2)

        # List `stocks` is a string of comma-separated stock symbols
        stocks = stocks.split(", ")

        with col1:
            # Get the list of stocks data using the `download_stocks` function
            list_of_stocks = download_stocks(stocks)
            st.sidebar.success("Downloading latest stock data successfully!")

            # Create a DataFrame object from the closing prices of all stocks
            table = pd.DataFrame(
                [list_of_stocks[j]["Close"] for j in range(len(list_of_stocks))]
            ).transpose()

            # Set the column names to be the stocks symbols
            table.columns = stocks

            # Filter by date range selected by user
            df = table
            new_index = [df.index[t].date() for t in range(len(df.index))]
            check1 = tuple(
                [new_index[t] >= start_datetime for t in range(len(new_index))]
            )
            check2 = tuple(
                [new_index[t] <= end_datetime for t in range(len(new_index))]
            )
            final_idx = [check1[t] and check2[t] for t in range(len(new_index))]
            filtered_df = df[final_idx]
            if filtered_df.shape[0] > 100:
                st.sidebar.success("Data filtered by date range selected by user.")
                table = filtered_df
            else:
                st.warning(
                    "Date range by user not valid, default range (past 2 years) is used."
                )
                table = table.tail(255 * 2)

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
                    deltas.append(
                        (hist["Close"][1] - hist["Close"][0]) / hist["Close"][0]
                    )

                    ## calculate market cap
                    market_caps.append(
                        info["sharesOutstanding"] * info["previousClose"]
                    )

                    ## add print statement to ensure code is running
                    print(f"downloaded {ticker}")
                except Exception as e:
                    print(e)

            # Create dataframe for market cap
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
                labels=["grey", "skyblue", "lightblue", "lightgreen", "lime", "black"],
            )

            # Start new section: Market Cap
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

            # Plot heatmap
            fig_market_cap_heatmap = plot_mkt_cap(df=df_for_mkt_cap)
            st.plotly_chart(fig_market_cap_heatmap, use_container_width=True)

            # Start new section: Time-series Plot
            st.markdown(
                f"""
                    <h4 style='text-align: left;'>Time Series Plot of Daily Returns</h4>
                """,
                unsafe_allow_html=True,
            )
            return_figure = plot_returns(table=table)
            st.write(
                f"""
                Plot daily returns of the stocks selected: {stocks}
            """
            )
            st.pyplot(return_figure, use_container_width=True)

            st.success("Efficient portfolio teacheds Mr. Yin what to buy. 💡")
            st.warning(
                "Bias (1): The time of entry is a trade secret and is completely decided by me based on personal experience. So far, I have not been able to replicate the timing strategy."
            )
            st.warning(
                "Bias (2): Though stocks are presented above, the weights decided by Mr. Yin is drastically different from the above allocation. The reason is I tend to hold the stocks that have gone up a lot and I don't like selling high growth positions for tax purposes."
            )
            st.warning(
                "Bias (3): The initial stock pool construction is also not reproducible. I mostly pick stocks from large cap brackets but occasionally break his own rules. Empirically speaking, I tend to construct my portfolio using 90-10 rule for equity and cash holdings. In the 90 percent, I have a 80-20 rule with 80 percent in large cap and 20 percent in small cap."
            )
            st.warning(
                "Bias (4): Typically asset pricing models tend to balance portfolios on a fixed time window, e.g. once a month etc.. I do not strictly follow this rule. My entry points are random because stock price is stochastic and bottoms usually appear stochasticly."
            )

        with col2:
            # Start new section: MPT
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
            st.warning("What is Efficient Frontier?")
            with st.expander("Expand/collapse for references:"):
                st.markdown(
                    r"""
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
                """
                )

            returns = table.pct_change()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            eff_front_figure, some_data = display_simulated_ef_with_random(
                table, mean_returns, cov_matrix, num_portfolios, risk_free_rate
            )

            # Start new section: Efficient Portfolio
            st.markdown(
                f"""
                    <h4 style='text-align: center;'>Efficient Portfolio:</h4>
                """,
                unsafe_allow_html=True,
            )
            st.write(
                f"Annualised <span style='color:green'>Return</span> (efficient portfolio): {some_data['Annualised Return (efficient portfolio)']}",
                unsafe_allow_html=True,
            )
            st.write(
                f"Annualised <span style='color:red'>Volatility</span> (efficient portfolio): {some_data['Annualised Volatility (efficient portfolio)']}",
                unsafe_allow_html=True,
            )
            sharpe_ratio_for_eff_port = np.round(
                some_data["Annualised Return (efficient portfolio)"]
                / some_data["Annualised Volatility (efficient portfolio)"],
                3,
            )
            st.write(
                f"Annualised <span style='color:blue'>Sharpe Ratio</span> (efficient portfolio): {sharpe_ratio_for_eff_port}",
                unsafe_allow_html=True,
            )
            lower_bound_eff_port = np.round(
                some_data["Annualised Return (efficient portfolio)"]
                - 2 * some_data["Annualised Volatility (efficient portfolio)"],
                3,
            )
            upper_bound_eff_port = np.round(
                some_data["Annualised Return (efficient portfolio)"]
                + 2 * some_data["Annualised Volatility (efficient portfolio)"],
                3,
            )
            st.write(
                f"Annualised <span style='color:orange'>return confidence Interval</span> (efficient portfolio): [{lower_bound_eff_port}, {upper_bound_eff_port}]",
                unsafe_allow_html=True,
            )
            # st.write(f"Max Sharpe Allocation:")
            # st.table(some_data["Max Sharpe Allocation"])
            st.write(f"Max Sharpe Allocation in Percentile:")
            st.table(some_data["Max Sharpe Allocation in Percentile"])

            # Start new section: Min Variance Portfolio
            st.markdown(
                f"""
                    <h4 style='text-align: center;'>Min Variance Portfolio:</h4>
                """,
                unsafe_allow_html=True,
            )
            st.write(
                f"Annualised <span style='color:green'>Return</span> (min. variance portfolio): {some_data['Annualised Return (min variance portfolio)']}",
                unsafe_allow_html=True,
            )
            st.write(
                f"Annualised <span style='color:red'>Volatility</span> (min. variance portfolio): {some_data['Annualised Volatility (min variance portfolio)']}",
                unsafe_allow_html=True,
            )
            sharpe_ratio_for_mv_port = np.round(
                some_data["Annualised Return (min variance portfolio)"]
                / some_data["Annualised Volatility (min variance portfolio)"],
                3,
            )
            st.write(
                f"Annualised <span style='color:blue'>Sharpe Ratio</span> (efficient portfolio): {sharpe_ratio_for_mv_port}",
                unsafe_allow_html=True,
            )
            lower_bound_mv_port = np.round(
                some_data["Annualised Return (min variance portfolio)"]
                - 2 * some_data["Annualised Volatility (min variance portfolio)"],
                3,
            )
            upper_bound_mv_port = np.round(
                some_data["Annualised Return (min variance portfolio)"]
                + 2 * some_data["Annualised Volatility (min variance portfolio)"],
                3,
            )
            st.write(
                f"Annualised <span style='color:orange'>return confidence Interval</span> (min variance portfolio): [{lower_bound_mv_port}, {upper_bound_mv_port}]",
                unsafe_allow_html=True,
            )
            # st.write(f"Min Volatility Allocation:")
            # st.table(some_data["Min Volatility Allocation"])
            st.write(f"Min Volatility Allocation in Percentile:")
            st.table(some_data["Min Volatility Allocation in Percentile"])
            st.pyplot(eff_front_figure)

    else:
        st.warning("Please select an option and click the submit button!")
elif option == "Entry Strategy":
    if submit_button:
        tab1, tab2, tab3 = st.tabs(
            ["Overview", "Buy/Sell Signal (Interactive)", "Basic Stock Info"]
        )

        # Get data
        all_info = get_stock_info(this_stock)

        with tab1:
            st.markdown(
                r"""
                The Relative Strength Index ([RSI](https://www.investopedia.com/terms/r/rsi.asp)) 
                is a momentum oscillator that determines the pace and variation of security prices.
                It is usually depicted graphically and oscillates on a scale of zero to 100.

                The RSI oscillates on a scale of zero to 100. Low RSI levels, below 30, generate 
                buy signals and indicate an oversold or undervalued condition. High RSI levels, 
                above 70, generate sell signals and suggest that a security is overbought or 
                overvalued. A reading of 50 denotes a neutral level or balance between bullish 
                and bearish positions.

                We allow users to select the number of days for RSI (this is one of the input 
                text area on the left sidebar). We also allow users to select the range to 
                measure margin of error, e.g. default values are (20, 80).
                """
            )
            # Start new section: Entry Strategy
            st.markdown(
                f"""
                    <h4 style='text-align: center;'>Entry Strategy:</h4>
                """,
                unsafe_allow_html=True,
            )
            st.write(f"Pick a stock you like and review entry strategy.")
            entry_plot = entry_strategy(
                start_date=str(start_datetime).split(" ")[0],
                end_date=str(end_datetime).split(" ")[0],
                tickers=this_stock,
                thresholds=rsi_thresholds,
                buy_threshold=thresholds_values[0],
                sell_threshold=thresholds_values[1],
            )
            st.pyplot(entry_plot)
            st.success("Entry Strategy teacheds Mr. Yin when to buy. 💡")

        with tab2:
            fig = entry_strategy_plotly(
                start_date=str(start_datetime).split(" ")[0],
                end_date=str(end_datetime).split(" ")[0],
                tickers=this_stock,
                thresholds=rsi_thresholds,
                buy_threshold=thresholds_values[0],
                sell_threshold=thresholds_values[1],
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        with tab3:
            st.warning(
                "Please be patient with us. Our generative AI robot is preparing the info for you!"
            )
            the_stock_info = all_info["get stock info"]
            assistant_prompt = {
                "role": "assistant",
                "content": "You are a helpful AI assistant for the user.",
            }
            result = []
            result.append(assistant_prompt)
            prompt = f"""
                Write a mardown file based on {the_stock_info}
            """
            user_prompt = {"role": "user", "content": prompt}
            result.append(user_prompt)
            try:
                response = call_chatcompletion(result)
                st.markdown(response)
            except:
                st.warning(
                    "OpenAI limit reached. We'll manually pull summarized information for you."
                )
                df = []
                for key, value in the_stock_info.items():
                    df.append([key, value])
                df = pd.DataFrame(df)
                st.table(df)
            st.success(
                "We used Yahoo Finance to acquire data and OpenAI's ChatGPT to create this file!"
            )

        st.warning(
            "Note (1): The entry strategy presented above simulates largely what Mr. Yin is executing, but the number of days and thresholds are not reproducible and these parameters are largely based on experience."
        )

        st.warning(
            "Note (2): Mr. Yin currently doesn't execute any exit strategy. Holding a stock is like marriage. Mr. Yin does not believe in short term profits and it certainly does not fulfil fiduciary by his experience. For starters, think about the tax you pay."
        )
    else:
        st.warning("Please select an option and click the submit button!")
else:
    st.warning("Please select an option and click the submit button!")


# Credit
def current_year():
    now = datetime.now()
    return now.year


# Example usage:
current_year = current_year()  # This will print the current year
st.markdown(
    f"""
        <h6 style='text-align: left;'>Copyright © 2010-{current_year} Present Yiqiao Yin</h6>
    """,
    unsafe_allow_html=True,
)
