import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr

# Import data
def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

# Portfolio Performance
def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    returns = np.sum(meanReturns*weights)*Time
    std = np.sqrt( np.dot(weights.T, np.dot(covMatrix, weights)) ) * np.sqrt(Time)
    return returns, std

stockList = ['VALE']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=360)

returns, meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)
returns = returns.dropna()

weights = np.random.random(len(returns.columns))
weights /= np.sum(weights)

returns['portfolio'] = returns.dot(weights)

def historicalVaR(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the percentile of the distribution at the given alpha confidence level
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)

    # A passed user-defined-function will be passed a Series for evaluation.
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha=alpha)

    else:
        raise TypeError("Expected returns to be dataframe or series")

    def historicalCVaR(returns, alpha=5):
        """
        Read in a pandas dataframe of returns / a pandas series of returns
        Output the CVaR for dataframe / series
        """
        if isinstance(returns, pd.Series):
            belowVaR = returns <= historicalVaR(returns, alpha=alpha)
            return returns[belowVaR].mean()

        # A passed user-defined-function will be passed a Series for evaluation.
        elif isinstance(returns, pd.DataFrame):
            return returns.aggregate(historicalCVaR, alpha=alpha)

        else:
            raise TypeError("Expected returns to be dataframe or series")

Time = 1

VaR = -historicalVaR(returns['portfolio'], alpha = 5)*np.sqrt(Time)
CVaR = -historicalVaR(returns['portfolio'],alpha=5)*np.sqrt(Time)
pRet, pStd = portfolioPerformance(weights,meanReturns, covMatrix, Time)

InitialInvestment= 1000#ПЕРЕМЕННАЯ
print('Ожидаемая доходность портфеля: ',round(InitialInvestment*pRet,2))
print('Значение риска 95 CI: ',round(InitialInvestment*VaR,2))
print('Условный 95CI: ',round(InitialInvestment*CVaR,2))