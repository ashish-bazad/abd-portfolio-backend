import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(stock_list, start_date=None, end_date=None, period=None, dividend=False):
    
    """Returns stock_price_df, stock_dividend_df"""
    
    if type(stock_list)!=list:
        stock_list = [stock_list]

    if len(stock_list) > 1:
        tickers = yf.Tickers(stock_list)
        
        if start_date is not None:
            historical_data = tickers.history(start=start_date, end=end_date)
        if period is not None:
            historical_data = tickers.history(period=period)
        
        if dividend:

            hist_price = historical_data.Close
            hist_price = hist_price.bfill(axis=0)
            hist_price = hist_price.ffill(axis=0)     
            
            hist_dividend = historical_data.Dividends
            hist_dividend = hist_dividend.bfill(axis=0)
            hist_dividend = hist_dividend.ffill(axis=0)

            return hist_price.dropna(), hist_dividend.dropna()
        else:

            hist_price = historical_data.Close 
            hist_price = hist_price.bfill(axis=0)
            hist_price = hist_price.ffill(axis=0)

            return hist_price.dropna()
    
    elif len(stock_list) == 1:
        tickers = yf.Ticker(stock_list[0])
        
        if start_date is not None:
            historical_data = tickers.history(start=start_date, end=end_date)
        if period is not None:
            historical_data = tickers.history(period=period)
        
        if dividend:    
            hist_price = historical_data.Close
            hist_price.index = hist_price.index.tz_localize(None)     
            hist_price = hist_price.bfill(axis=0)
            hist_price = hist_price.ffill(axis=0)

            hist_dividend = historical_data.Dividends
            hist_dividend.index = hist_dividend.index.tz_localize(None)
            hist_dividend = hist_dividend.bfill(axis=0)
            hist_dividend = hist_dividend.ffill(axis=0)

            return pd.DataFrame(hist_price).rename(columns={'Close':stock_list[0]}).dropna(), pd.DataFrame(hist_dividend).rename(columns={'Close':stock_list[0]}).dropna()
        else:
            hist_price = historical_data.Close
            hist_price.index = hist_price.index.tz_localize(None) 
            hist_price = hist_price.bfill(axis=0)
            hist_price = hist_price.ffill(axis=0)

            return pd.DataFrame(hist_price).rename(columns={'Close':stock_list[0]}).dropna()
