"""

@author: dmcdade
@created: 2018-05-02 21:28

"""

import os
import sys
import pandas as pd
from pandas import DateOffset, read_sql_query, DataFrame, to_datetime
from sqlalchemy import create_engine
import uuid
from config import config
from stock_list import get_sp500_ticker_df
from stock_fundamentals import StockFundamentals


def controller(engine):
    period = pd.to_datetime('now')
    data = [uuid.uuid4(), period]
    current = DataFrame(data=[data], columns=['pk', 'created_at'])
    try:
        controller = read_sql_query("SELECT * FROM stock_controller", engine)
        controller['created_at'] = controller['created_at'].apply(to_datetime)
        max_created_at = controller['created_at'].max()
    except:
        max_created_at = pd.to_datetime('2001-01-01')
    should_run = period > max_created_at + DateOffset(days=7)
    return current, should_run


def execute_query(engine, sql_query=''):
    with engine.begin() as connection:
        result = connection.execute(sql_query)
    return result


def main(engine):
    stock_fundamentals = StockFundamentals()
    sp500 = get_sp500_ticker_df()
    sp500.to_sql('sp500', engine, index=True, if_exists='replace')
    execute_query(engine, 'DROP TABLE stock_fundamentals')
    for ticker in sp500['ticker_symbol'].values:
        print(ticker, sp500[sp500['ticker_symbol'] == ticker].index)
        fundamental = stock_fundamentals.get_stock_fundamentals(ticker)
        fundamental.to_sql('stock_fundamentals', engine, index=False, if_exists='append')


if __name__ == '__main__':
    config = config[os.getenv('FLASK_CONFIG') or 'default']
    engine = create_engine(config.SQLALCHEMY_DATABASE_URI)
    current, should_run = controller(engine)
    if should_run:
        main(engine)
        current['updated_at'] = to_datetime('now')
        current.to_sql('stock_controller', engine, if_exists='append')
