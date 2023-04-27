import pandas as pd
import numpy as np

from dataclasses import dataclass
from pandas import DataFrame, Series

from .utils import get_rebalance_dates

@dataclass
class SimResult:
    prices: DataFrame | Series
    values: DataFrame | Series
    exposure: DataFrame | Series
    result: DataFrame | Series
    all_dates: list
    rebal_dates: list

def validate_weights(weights):
    if isinstance(weights, dict):
        if len(weights) == 0:
            raise ValueError("Dict for the weights has zero length.")

        if sum(weights.values()) > 1:
            raise ValueError("Dict for the weights has sum more than 1.")

        if sum(weights.values()) < 1:
            raise ValueError("Dict for the weights has sum less than 1.")

    if isinstance(weights, str) and weights not in ["ew"]:
        raise ValueError(
            f"""Not a valid string value for weights parameter: {weights}.
                         If it isn't a dict, it has to be 'ew' (equal weight)"""
        )


def simulate_without_rebalance(prices, start_weights: dict | str = "ew") -> SimResult:
    validate_weights(start_weights)

    if start_weights == "ew":
        values = prices.pct_change().fillna(0).add(1).cumprod()
        sim_result = values.sum(axis=1).pct_change().fillna(0).add(1).cumprod()

    if isinstance(start_weights, dict):
        tickers = prices.columns.to_list()

        normalized_prices = prices.pct_change().fillna(0).add(1).cumprod()

        weighted_values = pd.DataFrame()
        for ticker in tickers:
            weighted_values[ticker] = normalized_prices[ticker].mul(
                start_weights[ticker]
            )
        values = weighted_values.sum(axis=1)
        sim_result = values.pct_change().fillna(0).add(1).cumprod()

    sim_result.index.name = "date"
    sim_result.columns = ["sim"]

    total_value = values.sum(axis=1)
    exposure = values.apply(lambda row: row / total_value.loc[row.name], axis=1)

    return SimResult(
        prices = prices,
        values = values,
        exposure = exposure,
        result = sim_result,
        all_dates = sim_result.index.to_list(),
        rebal_dates = [],
    )


def simulate_with_rebalance(
    prices: DataFrame,
    rebal_weights: dict | str,
    rebal_freq: str = "1M",
) -> SimResult:
    validate_weights(rebal_weights)

    tickers = prices.columns.to_list()
    n_tickers = len(prices.columns)
    returns = prices.pct_change().fillna(0)
    all_dates = prices.index.to_list()
    rebal_dates = get_rebalance_dates(all_dates, rebal_freq)

    if rebal_weights == "ew":
        weights = pd.Series(index=tickers, dtype=np.float64).fillna(1 / n_tickers).to_dict()

    if isinstance(rebal_weights, dict):
        weights = rebal_weights.copy()

    values = pd.DataFrame()
    for i, date in enumerate(all_dates):
        for ticker in tickers:
            if i == 0:
                values.loc[date, ticker] = 1
            else:
                if date not in rebal_dates:
                    values.loc[date, ticker] = values.iloc[i - 1][ticker] * (
                        1 + returns.loc[date, ticker]
                    )
                else:
                    values.loc[date, ticker] = (
                        total_value.iloc[i - 1] * weights[ticker]
                    ) * (1 + returns.loc[date, ticker])
        total_value = values.sum(axis=1)

    exposure = values.apply(lambda row: row / total_value.loc[row.name], axis=1)

    sim_result = pd.DataFrame(total_value).pct_change().fillna(0).add(1).cumprod()
    sim_result.columns = ["sim"]

    return SimResult(
        prices = prices,
        values = values,
        exposure = exposure,
        result = sim_result,
        all_dates = all_dates,
        rebal_dates = rebal_dates,
    )