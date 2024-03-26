# Value at Risk (VaR) Model for Portfolio Risk Management

## Overview
This project implements a Value at Risk (VaR) model using Monte Carlo simulation to estimate the potential loss in the value of a portfolio over a specified time horizon and confidence interval. The model considers different portfolio allocation strategies, including minimum variance and optimal Sharpe ratio portfolios, to provide insights into portfolio risk management.

## Author
**Omer Sany Prakash**  
Graduate Student  
Oklahoma State University

## Installation
To run this project, you need to have Python installed along with the following libraries:

- scipy
- numba
- pandas
- yfinance
- numpy
- matplotlib

You can install these dependencies using pip:

```bash
pip install scipy numba pandas yfinance numpy matplotlib
```

## Usage
Clone the repository to your local machine.
Ensure you have the necessary Python dependencies installed (see Installation section).
Update the config.yaml file with your assumptions.
Execute the main file:
```python
python main.py
```

# Project Structure

- `index-holdings-xly.xls`: Excel file containing the consumer discretionary (XLY) fund summary.
- `var_model.py`: Python script implementing the VaR model and portfolio optimization classes.
- `README.md`: Markdown file providing an overview of the project.

## Detailed Explanation

### Portfolio Optimization
- The script begins by importing necessary packages and defining classes for portfolio optimization and VaR modeling.
- `OptPort` class provides methods to construct minimum variance and optimal Sharpe ratio portfolios using historical stock price data retrieved from Yahoo Finance (`yfinance`).
- `VaRModel` class represents the VaR model, which utilizes Monte Carlo simulation to generate a distribution of potential portfolio returns and estimate the VaR.

### Monte Carlo Simulation
- Monte Carlo simulation is employed to model the future behavior of the portfolio by generating multiple random scenarios.
- The simulation calculates potential gains or losses of the portfolio based on historical return data and portfolio characteristics such as expected return and risk.

### Risk Metrics Calculation
- The model calculates various risk metrics including VaR, Tail Value-at-Risk (TVaR), and Conditional Tail Expectation (CTE) to assess portfolio risk.
- VaR is estimated as the specified percentile of the simulated portfolio returns distribution.
- TVaR represents the average loss exceeding the VaR.
- CTE indicates the average loss in the tail of the distribution beyond VaR.

### Visualization
- The script visualizes the simulated portfolio returns distribution using a histogram and highlights the VaR level for the specified confidence interval.

