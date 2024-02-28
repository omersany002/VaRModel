"""
Calculating the VAR of SPDR Consumer Discretionary Portfolio
-------
Written by: Omer Sany Prakash
"""

# importing packages
import warnings
from scipy.optimize import minimize
from numba import jit
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# file path of consumer discretionary (XLY) fund summary
PATH = r'index-holdings-xly.xls'
df = pd.read_excel(PATH)
TICKERS = list(df[df.columns[0]])[1:]

# ignoring yfinance future warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

class OptPort:
    """A class that employs Optimal Portfolio Strategies"""
    def __init__(self, tickers, port_type):
        self.tickers = tickers
        self.port_type = port_type
        self.price_data =yf.Tickers(self.tickers).history('5y').Close

    def get_minimum_variance_portfolio(self):
        """Constructs a minimum variance portfolio"""
        num_stocks = len(self.tickers)
        returns = np.log(self.price_data / self.price_data.shift(1))
        cov_matrix = returns.cov()

        def calculate_portfolio_variance(weights, covariance_matrix):
            return np.dot(weights.T, np.dot(covariance_matrix, weights))

        args = cov_matrix
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (0.0, 1.0)
        bounds = tuple(bound for asset in range(num_stocks))
        result = minimize(calculate_portfolio_variance, num_stocks*[1./num_stocks,], \
            args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return pd.Series(result.x,index=self.tickers)

    def get_optimal_sharpe_ratio_portfolio(self):
        """Constructs an optimal Sharpe ratio portfolio"""
        num_stocks = len(self.tickers)
        returns = np.log(self.price_data / self.price_data.shift(1))
        avg_returns = returns.mean()
        cov_matrix = returns.cov()

        def calculate_sharpe_ratio(weights, avg_returns, covariance_matrix):
            portfolio_return = np.sum(avg_returns * weights)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_std_dev
            return -sharpe_ratio

        args = (avg_returns, cov_matrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (0.0, 1.0)
        bounds = tuple(bound for asset in range(num_stocks))
        result = minimize(calculate_sharpe_ratio, num_stocks*[1./num_stocks,], \
            args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return pd.Series(result.x, index=self.tickers)

    def portfolio(self):
        """"Returns the Choosen Portfolio"""
        if self.port_type[0] == 'M':
            return self.get_minimum_variance_portfolio()
        if self.port_type[0] == 'O':
            return self.get_optimal_sharpe_ratio_portfolio()
        weights = [1/len(self.tickers) for _ in range(len(self.tickers))]
        result = pd.Series(weights, index=self.tickers)
        return result

class VaRModel:
    """
    A class to represent a Value at Risk (VaR) model using Monte Carlo simulation.
    """
    def __init__(self, tickers, sims, port_val, days, conf_int,  seed, port_type='E'):
        """
        Constructs all the necessary attributes for the VaR_model object.
        """
        self.tickers = tickers
        self.simulations = sims
        self.portfolio_value = port_val
        self.days = days
        self.conf_int = conf_int
        self.port_type = port_type
        self.seed = seed
        self.mc_returns = self.monte_carlo()
        self.var = None

    def portfolio_summ(self):
        """Calculates the expected return and risk of the portfolio based on historical data."""
        price = yf.Tickers(self.tickers).history('180d').Close
        retn = np.log(price / price.shift(1))
        weights = OptPort(self.tickers, self.port_type).portfolio().values
        avg_retn = np.sum(retn.mean() * weights) * 252
        cov_matrix = retn.cov() * 252
        port_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return avg_retn, port_std_dev

    @staticmethod
    @jit(nopython = True)
    def portfolio_gain_loss(port_val, exp_ret, port_risk, days):
        """Calculates the potential gain or loss of the portfolio over a certain number of days."""
        z_score = np.random.normal(0,1)
        return port_val * exp_ret * days + port_val * \
            port_risk * z_score * np.sqrt(days)

    def monte_carlo(self):
        """
        Runs a Monte Carlo simulation to generate a distribution of potential portfolio returns.
        """
        np.random.seed(self.seed)
        p_return, p_risk= self.portfolio_summ()
        returns = []
        for _ in range(self.simulations):
            gain_loss = self.portfolio_gain_loss(self.portfolio_value, \
                p_return, p_risk, self.days)
            returns.append(gain_loss)
        return returns

    def risk_metrices(self):
        """Calculations of all the risk metrices"""
        self.var = -np.percentile(self.mc_returns, 100*(1-self.conf_int))

        # Tail Value-at-Risk (TVaR)
        tail_returns = [r for r in self.mc_returns if r < self.var]
        tvar = -np.mean(tail_returns)

        # Conditional Tail Expectation (CTE)
        alpha = 1 - self.conf_int
        sorted_returns = np.sort(self.mc_returns)
        index = int(alpha * len(sorted_returns))
        cte = -np.mean(sorted_returns[:index])

        print(f'Confidence Level: {self.conf_int*100}% | Window: {self.days} Day(s)')
        print(f'Value at Risk (VaR): {self.var:,.0f}$')
        print(f'Tail Value-at-Risk (TVaR): {tvar:,.0f}$')
        print(f'Conditional Tail Expectation (CTE): {cte:,.0f}$')

    def summary(self):
        """
        Prints the VaR at the specified confidence level and plots a 
        histogram of the simulated returns.
        """
        print("Risk Metrices:")
        print("-------------")
        self.risk_metrices()        # Printing VaR, TVaR and CTE
        print("-------------")
        plt.figure(figsize=(10,6))
        plt.hist(self.mc_returns, bins = 50)
        plt.xlabel('Portfolio Gain/Loss')
        plt.ylabel('Frequency')
        plt.title('VaR Model')

        # adding the var line to the histogram
        plt.axvline(-1 * self.var, color = 'r', linewidth=1, \
            label = f'VaR at {self.conf_int*100}% confidence level')
        plt.legend()
        plt.show()

# variables
SIMULATIONS = 1_000_000  # Number of simulations to run
PORTFOLIO_VALUE = 1_000_000  # Total value of the portfolio
DAYS = 1  # Time horizon for the VaR calculation (in days)
CONFIDENCE_INTERVAL = .95  # Confidence level for the VaR calculation
SEED = 1998

# Specifying portfolio type
# 'MINIMUM_VARIANCE' for minimum variance portfolio
# 'OPTIMAL_SHARPE' for portfolio with optimal Sharpe ratio
# Leave empty for equal weighted portfolio by default
PORT_TYPE = 'OPTIMAL_SHARPE'

# Creating an instance of the class and running the model
VaR = VaRModel(TICKERS, SIMULATIONS, PORTFOLIO_VALUE, DAYS, CONFIDENCE_INTERVAL, SEED, PORT_TYPE)
VaR.summary()
