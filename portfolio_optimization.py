"""Portfolio Optimization Script"""

# importing packages
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
