import pandas as pd
from pandas import DatetimeIndex, Series, DataFrame

def format_prices(prices):
    prices.index = pd.to_datetime(prices.index)
    prices[prices.columns] = prices[prices.columns].astype(float)
    prices.name = "prices"
    prices.index.name = "date"
    return prices.copy()


def get_rebalance_dates(all_dates, freq):
    _rebal_dates = (
        pd.DataFrame(index=all_dates, columns=["0"]).resample(freq).last().index
    )
    aux = pd.DataFrame(index=_rebal_dates, columns=[0]).fillna(1)
    aux_daily = pd.concat([aux, pd.DataFrame(index=all_dates)], axis=1).shift(1)
    aux_daily.iloc[0] = 1  # start
    rebal_dates = aux_daily.dropna().index.to_list()
    return rebal_dates

def get_sim_dates(prices: DataFrame, rebal_freq: str):
    if not isinstance(prices.index, DatetimeIndex):
        raise ValueError(f"Please change 'prices.index' to DatetimeIndex")
    if not isinstance(rebal_freq, str):
        raise ValueError(f"Invalid type for 'rebal_freq")
    all_dates = prices.index
    rebal_dates = prices.ezfreq(rebal_freq).index
    return all_dates, rebal_dates