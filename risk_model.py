"""
Calculating the VAR of SPDR Consumer Discretionary Portfolio
-------
Written by: Omer Sany Prakash
"""

# importing packages
import warnings
from numba import jit
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from portfolio_optimization import OptPort

# ignoring yfinance future warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

class VaRModel:
    """
    A class to represent a Value at Risk (VaR) model using Monte Carlo simulation.
    """
    def __init__(self, tickers, args):
        """
        Constructs all the necessary attributes for the VaR_model object.
        """
        self.tickers = tickers
        self.risk_params = args['risk_model_params']
        self.conf_int = self.risk_params['confidence_interval']
        self.mc_returns = self.monte_carlo()
        self.var = None

    def portfolio_summ(self):
        """Calculates the expected return and risk of the portfolio based on historical data."""
        price = yf.Tickers(self.tickers).history('180d').Close
        retn = np.log(price / price.shift(1))
        weights = OptPort(self.tickers, self.risk_params['portfolio_type']).portfolio().values
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
        p_return, p_risk= self.portfolio_summ()
        returns = []
        for _ in range(self.risk_params['simulations']):
            gain_loss = self.portfolio_gain_loss(self.risk_params['investment_value'], \
                p_return, p_risk, self.risk_params['holding_day'])
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

        print(f'Confidence Level: {self.conf_int*100}% | Window: {self.risk_params['holding_day']} Day(s)')
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
