"""
INPUT NEEDED:
    stocks (list): 
        list of ticker to be displayed in the table of the first page


OUTPUT:
    table (dict):
        table for the first page
    price_data (dict):
        price_data of selected ticker for chart
    volatility_data (dict):
        volatility of selected ticker for chart
"""

from . import data
import pandas as pd
import numpy as np

## GATHER DATA FOR TABLE: 1-year price data
def initialize(stock_list, period):

    prices = data.fetch_data(stock_list, period=period)
    prices = prices[stock_list]
    symbols = np.array(prices.columns)
    current_prices = np.array(prices.iloc[-1,:])
    high_52Week = np.array(prices.max())
    low_52Week = np.array(prices.min())
    change_in_price = np.array(prices.iloc[-1,:] - prices.iloc[0,:])
    percent_change_in_price = np.array((prices.iloc[-1,:] / prices.iloc[0,:] - 1)*100)

    table_dict = {'SYMBOLS':symbols, 'PRICE':current_prices, 'HIGH':high_52Week, 'LOW':low_52Week, 'CHANGE':change_in_price, 'PCHANGE':percent_change_in_price}

    ## ADDING CHART DATA

    prices_dict = prices.reset_index().rename(columns={'Date':'DATE'})
    prices_dict = prices_dict.to_dict(orient='list')

    log_returns = np.log(prices / prices.shift(1))[1:]
    volatility_dict = log_returns.reset_index().rename(columns={'Date':'DATE'})
    volatility_dict = volatility_dict.to_dict(orient='list')
    
    return table_dict, prices_dict, volatility_dict