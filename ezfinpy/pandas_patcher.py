import pandas as pd
from pandas import Series, DataFrame

from .simulator import SimResult, simulate_without_rebalance, simulate_with_rebalance

# simulation

def simulate(
    prices,
    rebalance: bool,
    weights: str | dict = "ew",
    rebal_freq: str = "1M",
    ) -> SimResult:
    """
    Run the simulation.
    -----
    parameters:
        rebalance: bool (required)
            If 'True', the simulation will rebalance itself.

        weights: str or dict, default 'ew'
            If dict, than it is the weight for each asset (number between 0 and 1),
            The sum can't be different than one.
            Example:
                asset_weights = {
                    'asset1': 0.3,
                    'asset2': 0.2,
                    'asset3': 0.5,
                }
            If 'ew', it runs the simulation using equal weight for all assets (1 / number of assets).

        rebal_freq: str, default '1M'
            Rebalance frequency. Has the same valid inputs as pandas.DataFrame.resample() function.
    """
    if rebalance:
        return simulate_with_rebalance(
            prices=prices, rebal_weights=weights, rebal_freq=rebal_freq
        )
    else:
        return simulate_without_rebalance(
            prices=prices, starting_weights=weights
        )

# utils

def base(data, base=1):
    return data.div(data.iloc[0]).mul(base)

def to_prices(returns):
    return returns.add(1).cumprod()

def include(_self, data: Series | DataFrame | list):
    if isinstance(data, list):
        for s in data:
            if isinstance(s, (Series, DataFrame)):
                _self = pd.concat([_self, s], axis=1)
            else:
                raise ValueError("Invalid types inside list")

    elif isinstance(data, (Series, DataFrame)):
        _self = pd.concat([_self, data], axis=1)
    else:
        raise ValueError("Invalid arg type.")
    return _self

def exclude(_self, names: str | list | Series | DataFrame):
    return _self.drop(columns=names)

def total_return(prices):
    return prices.iloc[-1]/prices.iloc[0] - 1

def drawdown(prices):
    return prices.div(prices.cummax()) - 1

def volatility(prices, days: int = 252):
    return prices.pct_change().std() * days**0.5

def rolling_volatility(prices, days: int = 252):
    return prices.pct_change().rolling(days).std() * 252**0.5

def cagr(prices, days_on_year: int = 252):
    return (prices.iloc[-1]/prices.iloc[0])**(252/len(prices)) - 1

def sharpe(prices, risk_free: float):
    """
    risk_free:
        CAGR of the risk free rate
    """
    return (cagr(prices) - risk_free)/volatility(prices)


# modifiers

def ezplot():
    pass

def ezprint():
    pass

def ezstats():
    pass

def ezresample(data, freq):
    all_dates = data.index
    dates = pd.DataFrame(index=all_dates)
    dates['correct_date'] = dates.index
    dates = dates.resample(freq).last()
    resampled_data = data.loc[dates['correct_date'].values].copy()
    return resampled_data

funcs = [
    simulate,
    base,
    to_prices,
    include,
    exclude,
    total_return,
    drawdown,
    volatility,
    rolling_volatility,
    cagr,
    sharpe,
    ezplot,
    ezprint,
    ezresample,
]
for pandas_type in [Series, DataFrame]:
    for func in funcs:
        setattr(pandas_type, func.__name__, func)

