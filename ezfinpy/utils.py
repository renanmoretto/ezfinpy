import pandas as pd
from pandas import DatetimeIndex, DataFrame

def format_prices(prices):
    prices.index = pd.to_datetime(prices.index)
    prices[prices.columns] = prices[prices.columns].astype(float)
    prices.name = "prices"
    prices.index.name = "date"
    return prices.copy()

def get_sim_rebal_dates(prices: DataFrame, rebal_freq: str):
    if not isinstance(prices.index, DatetimeIndex):
        raise ValueError("Please change 'prices.index' to DatetimeIndex")
    elif not isinstance(rebal_freq, str):
        raise ValueError("Invalid type for 'rebal_freq")
    rebal_dates = prices.ezresample(rebal_freq).index
    return rebal_dates