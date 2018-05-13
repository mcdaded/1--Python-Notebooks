"""

# Source data is based on this ADVFN site
https://uk.advfn.com/p.php?pid=financials&symbol=AAPL&btn=quarterly_reports
https://uk.advfn.com/p.php?pid=financials&symbol=AAPL&btn=quarterly_reports&istart_date=0

@author: dmcdade
@created: 2018-05-02 21:04

"""

import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame, to_datetime
import requests
from bs4 import BeautifulSoup
from io import StringIO


class ADVFN:
    """
    Class for interacting and extracting stock fundamental data.
    """
    urlbase = 'https://uk.advfn.com/p.php?pid=financials'
    fundamental_type = ['INDICATORS', 'INCOME STATEMENT', 'INCOME STATEMENT (YEAR-TO-DATE)',
                        'BALANCE SHEET', 'ASSETS', 'EQUITY & LIABILITIES', 'CASH-FLOW STATEMENT',
                        'OPERATING ACTIVITIES', 'INVESTING ACTIVITIES', 'FINANCING ACTIVITIES',
                        'NET CASH FLOW', 'RATIOS CALCULATIONS', 'PROFIT MARGINS', 'NORMALIZED RATIOS',
                        'SOLVENCY RATIOS', 'EFFICIENCY RATIOS', 'ACTIVITY RATIOS', 'LIQUIDITY RATIOS',
                        'CAPITAL STRUCTURE RATIOS', 'PROFITABILITY', 'AGAINST THE INDUSTRY RATIOS']

    def __init__(self):
        pass

    def _get_format_url(self, symbol, start_idx):
        url = self.urlbase + '&symbol={symbol}&btn=quarterly_reports'.format(symbol=symbol)
        url += '&istart_date={start_idx}'.format(start_idx=start_idx)
        return url

    def _get_soup_table(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml')
        tables = soup.findAll('table')
        # TODO: figure out better way than index
        soup_data_table = tables[7].find('table')
        return soup_data_table

    def _get_format_data_df(self, soup_data_table):
        df = pd.read_html(StringIO(str(soup_data_table)))[0]
        is_empty = df[1].isnull() & df[2].isnull() & df[3].isnull() & df[4].isnull() & df[5].isnull()
        empty_df = df[is_empty].copy()
        df = df[~is_empty].copy()
        empty_df = empty_df[empty_df[0].isin(self.fundamental_type)].copy().reset_index()
        quarters = df[df[0] == 'quarter end date'].values[0]
        quarters[0] = 'fundamental'
        df.columns = quarters
        df['category'] = None
        for idx in df.index:
            category = empty_df[empty_df['index'] < idx].tail(1)[0].values[0]
            df.loc[idx, 'category'] = category
        df = pd.melt(df, id_vars=['category', 'fundamental'], value_vars=quarters[1:])
        df.columns = ['category', 'fundamental', 'quarter', 'val']
        return df[~df['quarter'].isnull()]

    def get_stock_fundamentals(self, symbol, start_idx=0, last_df=DataFrame()):
        columns = ['quarter', 'symbol', 'category', 'fundamental', 'val']
        stock_data = pd.DataFrame(columns=columns)
        # check that the fundamentals exist
        url = self._get_format_url(symbol, start_idx)
        table = self._get_soup_table(url)
        # check if data was returned and not the same as previously processed records to continue
        if table:
            stock_data = self._get_format_data_df(table)
            if not stock_data.equals(last_df):
                # Get the other fundamentals
                df = self.get_stock_fundamentals(symbol, start_idx+1, stock_data)
                stock_data = stock_data.append(df, ignore_index=True).drop_duplicates()
        stock_data['symbol'] = symbol
        return stock_data[columns]


class StockFundamentals(ADVFN):

    def __init__(self):
        ADVFN.__init__(self)


if __name__ == '__main__':
    stocks = ADVFN()
    aapl = stocks.get_stock_fundamentals('AAPL')
    aapl.shape
