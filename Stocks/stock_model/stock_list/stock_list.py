"""
Process to fetch list of S&P 500 stocks

@author: dmcdade
@created: 2018-05-02 at 20:46

"""

from bs4 import BeautifulSoup
import pickle
import requests
import pandas as pd


def save_sp500_tickers(tickers):
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers, f)


def get_sp500_tickers():
    url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = str(row.findAll('td')[0].text)
        tickers.append(ticker)
    # save_sp500_tickers(tickers)
    return tickers


def get_sp500_ticker_df():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data = pd.read_html(url)
    columns = data[0].ix[0]
    tabledf = data[0].ix[1:]
    tabledf.columns = [c.replace(' ', '_').lower() for c in columns]
    return tabledf


if __name__ == '__main__':
    tickers = get_sp500_ticker_df()
    print(tickers)
