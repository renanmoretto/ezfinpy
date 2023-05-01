import pandas as pd
import numpy as np

from dataclasses import dataclass
from pandas import DataFrame, Series

from .utils import get_sim_rebal_dates

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


def simulate_without_rebalance(prices, starting_weights: dict | str = "ew") -> SimResult:
    validate_weights(starting_weights)

    if starting_weights == "ew":
        values = prices.pct_change().fillna(0).add(1).cumprod()
        sim_result = values.sum(axis=1).pct_change().fillna(0).add(1).cumprod()

    if isinstance(starting_weights, dict):
        tickers = prices.columns.to_list()

        normalized_prices = prices.pct_change().fillna(0).add(1).cumprod()

        weighted_values = pd.DataFrame()
        for ticker in tickers:
            weighted_values[ticker] = normalized_prices[ticker].mul(
                starting_weights[ticker]
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
    rebal_weights: dict | str = 'ew',
    rebal_freq: str = "1M",
) -> SimResult:
    validate_weights(rebal_weights)

    tickers = prices.columns.to_list()
    n_tickers = len(prices.columns)

    if rebal_weights == "ew":
        weights = pd.Series(index=tickers, dtype=np.float64).fillna(1 / n_tickers).to_dict()
    if isinstance(rebal_weights, dict):
        weights = rebal_weights.copy()

    returns_array = prices.pct_change().fillna(0).values

    all_dates_array = prices.index.values

    rebal_dates = get_sim_rebal_dates(prices, rebal_freq)
    rebal_dates_array = np.insert(rebal_dates.values, 0, all_dates_array[0]) # first day

    values = np.zeros((len(all_dates_array), len(tickers)))
    total_value = []
    for i, date in enumerate(all_dates_array):
        for j, ticker in enumerate(tickers):
            if i == 0:
                values[i, j] = 1 * weights.get(ticker)
            else:
                if all_dates_array[i-1] in rebal_dates_array:
                    values[i, j] = (total_value[i-1] * weights.get(ticker)) * (1 + returns_array[i, j])
                else:
                    values[i, j] = values[i-1, j] * (1 + returns_array[i, j])
        total_value.append(values[i,:].sum())

    exposure = values / np.array(total_value)[:, np.newaxis]

    values = pd.DataFrame(values, index=all_dates_array, columns=tickers)
    exposure = pd.DataFrame(exposure, index=all_dates_array, columns=tickers)
    sim_result = pd.DataFrame(total_value, index=all_dates_array, columns=['sim'])

    return SimResult(
        prices = prices,
        values = values,
        exposure = exposure,
        result = sim_result,
        all_dates = prices.index,
        rebal_dates = rebal_dates,
    )

def cython_simulate_with_rebalance(
    prices: DataFrame,
    rebal_weights: dict | str = 'ew',
    rebal_freq: str = "1M",
) -> SimResult:
    validate_weights(rebal_weights)

    tickers = prices.columns.to_list()
    n_tickers = len(prices.columns)

    if rebal_weights == "ew":
        weights = pd.Series(index=tickers, dtype=np.float64).fillna(1 / n_tickers).to_dict()
    if isinstance(rebal_weights, dict):
        weights = rebal_weights.copy()

    returns_array = prices.pct_change().fillna(0).values

    all_dates_array = prices.index.values

    rebal_dates = get_sim_rebal_dates(prices, rebal_freq)
    rebal_dates_array = np.insert(rebal_dates.values, 0, all_dates_array[0]) # first day

    values = np.zeros((len(all_dates_array), len(tickers)))
    total_value = []
    for i, date in enumerate(all_dates_array):
        for j, ticker in enumerate(tickers):
            if i == 0:
                values[i, j] = 1 * weights.get(ticker)
            else:
                if all_dates_array[i-1] in rebal_dates_array:
                    values[i, j] = (total_value[i-1] * weights.get(ticker)) * (1 + returns_array[i, j])
                else:
                    values[i, j] = values[i-1, j] * (1 + returns_array[i, j])
        total_value.append(values[i,:].sum())

    exposure = values / np.array(total_value)[:, np.newaxis]

    values = pd.DataFrame(values, index=all_dates_array, columns=tickers)
    exposure = pd.DataFrame(exposure, index=all_dates_array, columns=tickers)
    sim_result = pd.DataFrame(total_value, index=all_dates_array, columns=['sim'])

    return SimResult(
        prices = prices,
        values = values,
        exposure = exposure,
        result = sim_result,
        all_dates = prices.index,
        rebal_dates = rebal_dates,
    )