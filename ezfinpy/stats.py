import pandas as pd
from tabulate import tabulate
from pandas import Series, DataFrame

class SingleStats:
    """
    
    """
    def __init__(self, prices: Series, risk_free: float = 0.0):
        if not isinstance(prices, Series):
            raise ValueError("Can't create a SingleStats using a Dataframe. Please pass prices as Series.")

        self.prices = prices
        self.risk_free = risk_free

        self.name = prices.name
        self.start = prices.index[0]
        self.end = prices.index[-1]

        if self.prices.index.to_series().diff().min() == pd.Timedelta('1 days'):
            self.best_day = self.prices.pct_change().max()
            self.worst_day = self.prices.pct_change().max()
        else:
            self.best_day = None
            self.worst_day = None

        self.best_month = self.prices.ezresample('1M').pct_change().max()
        self.worst_month = self.prices.ezresample('1M').pct_change().min()

        self.best_year = self.prices.ezresample('1Y').pct_change().max()
        self.worst_year = self.prices.ezresample('1Y').pct_change().min()

        self.stats = {
            'name': self.name,
            'start': self.start,
            'end': self.end,
            'total_return': self.prices.total_return(),
            'cagr': self.prices.cagr(),
            'max_drawdown': self.prices.drawdown().min(),
            'vol': self.prices.volatility(),
            'sharpe': self.prices.sharpe(risk_free=self.risk_free),
            'best_day': self.best_day,
            'worst_day': self.worst_day,
            'best_month': self.best_month,
            'worst_month': self.worst_month,
            'best_year': self.best_year,
            'worst_year': self.worst_year,
        }

        self.tabulated_stats = [
            ['Name', self.name],
            ['Start', self.start.strftime('%Y-%m-%d')],
            ['End', self.end.strftime('%Y-%m-%d')],
            ['Total Return', f"{self.stats['total_return']:,.2%}"],
            ['CAGR', f"{self.stats['cagr']:,.2%}"],
            ['Max Drawdown', f"{self.stats['max_drawdown']:,.2%}"],
            ['Volatility (ann.)', f"{self.stats['vol']:,.2%}"],
            ['Sharpe', f"{self.stats['sharpe']:,.2f}"],
            ['Best Day', f"{self.stats['best_day']:,.2%}"],
            ['Worst Day', f"{self.stats['worst_day']:,.2%}"],
            ['Best Month', f"{self.stats['best_month']:,.2%}"],
            ['Worst Month', f"{self.stats['worst_month']:,.2%}"],
            ['Best Year', f"{self.stats['best_year']:,.2%}"],
            ['Worst Year', f"{self.stats['worst_year']:,.2%}"],
        ]
    
    def print_stats(self):
        print(tabulate(self.tabulated_stats, headers='firstrow'))

class GroupStats:
    def __init__(self, prices: DataFrame, risk_free: float = 0.0):
        if not isinstance(prices, DataFrame):
            raise ValueError("Please pass prices as DataFrame.")
        if len(prices.columns) < 2:
            raise ValueError("Number of securities not valid. Length of 'prices.columns' has to be bigger than 1.")
        
        self.prices = prices
        self.risk_free = risk_free

        self.group_list = [SingleStats(prices=prices[asset], risk_free=risk_free) for asset in prices.columns]

        self.group_stats = {}
        for single_stat in self.group_list:
            self.group_stats.update({single_stat.name: single_stat.stats})


        # generate tabulated table for the group
        tabulated_table = []
        for i, stat in enumerate(self.group_list):
            if i == 0:
                for stat_tabulated_row in stat.tabulated_stats:
                    row_key = stat_tabulated_row[0]
                    value = stat_tabulated_row[1]
                    tabulated_table.append([row_key, value])
            else:
                for row, value in zip(tabulated_table, stat.tabulated_stats):
                    row.append(value[1])            
        self.tabulated_stats = tabulated_table

    def print_stats(self):
        print(tabulate(self.tabulated_stats, headers='firstrow'))