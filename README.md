# WYN Advisor: Portfolio Optimization Streamlit App

<p>
    <img src="https://github.com/yiqiao-yin/WYN-Advisor/blob/main/docs/main.gif" width=800></img>
</p>

## Overview

This is a simple Streamlit app that allows users to generate an optimized portfolio allocation based on mean returns, covariance matrix and user inputs. The app calculates the efficient frontier and also identifies the portfolio with minimum variance.

This app can be seen as a basic first step towards a fully automated robo advisor. To access the app, please click the [Link](https://wyn-advisor.streamlit.app/)

## Installation

You must have Python 3.x installed on your machine.

1. Clone this repository to your local machine.
2. Open a terminal window and navigate to the root directory of this project.
3. Install required packages using pip:

   ```
   pip install -r requirements.txt
   ```

4. Run the app:

    ```
    streamlit run app.py
    ```

## Usage

Upon running the app, you will see an interface to input relevant parameters for generating the optimized portfolio:

- Stock returns (mean returns): A list of mean returns for each security or asset in the portfolio.
- Covariance Matrix: A covariance matrix for the securities or assets in the portfolio.
- Risk-free rate: The risk-free rate of return.
- Number of random portfolios: The number of random portfolios to generate.
- Starting weights (optional): A list of starting weights for all securities. If not provided, 
the app assigns equal weights to all securities by default.
- Stock List (optional): A list of names of all the securities. If not provided,
the app assigns names to securities as s1,s2..sn by default. 

Clicking on **Generate portfolio** button will create a chart displaying the simulated efficient frontier and the optimal portfolio allocation with the highest Sharpe ratio and another chart displaying portfolio allocations of both minimum variance portfolio  and the optimal portfolio allocation with the highest Sharpe ratio. 

The app also provides a data table showing:
- Expected annual returns
- Annual volatility
- Sharpe ratio 
- Portfolio weights for the maximum Sharpe ratio portfolio
- Portfolio weights for the minimum variance portfolio

## Credits

This app was developed by [Your Name] and was based on the following resources:

- [Simulating portfolio returns in Python](https://towardsdatascience.com/simulating-portfolio-returns-in-python-9f93b6e88dfe)
- [Efficient Frontier: Modern Portfolio Theory (MPT) and Python](https://towardsdatascience.com/efficient-frontier-modern-portfolio-theory-and-python-fb9508ff27e3)

If you encounter any issues or have suggestions for improvement, please contact [Your Email].