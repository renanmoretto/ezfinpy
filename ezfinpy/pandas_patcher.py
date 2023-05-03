import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

from .simulator import simulate

# utils

def rebase(data, base=1):
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

def ezplot(
        data: Series | DataFrame,
        drawdowns: bool = True,
        rolling_volatility: bool = True,
        returns: bool = False,
        **kw,
    ) -> None:

    prices = data.to_prices() if returns else data.copy()

    plot_info = [{'dataframe': prices, 'ylabel': 'Prices'}]

    if drawdowns:
        drawdowns_data = prices.drawdown()
        plot_info.append({'dataframe': drawdowns_data, 'ylabel': 'Drawdowns (%)'})

    if rolling_volatility:
        rolling_volatility_data = prices.rolling_volatility()
        plot_info.append({'dataframe': rolling_volatility_data, 'ylabel': 'Rolling Volatility (%)'})

    fig, axes = plt.subplots(nrows=len(plot_info), ncols=1, **kw)

    for i, plot in enumerate(plot_info):
        row = i
        ax = axes[row]
        ax.set_ylabel(plot['ylabel'])
        ax.grid()
        
        if row==0:
            ax.plot(plot['dataframe'].index, plot['dataframe'].values, linewidth=2, label=prices.columns)
            ax.legend(prices.columns.to_list())
        else:
            ax.plot(plot['dataframe'].index, plot['dataframe'].values, linewidth=2)

    title = ''
    fig.suptitle(title, fontsize=16)

    plt.show()

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


def patch():
    funcs = [
        simulate,
        rebase,
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

