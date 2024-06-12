import numpy as np
import pandas as pd
import scipy.stats as stats
from . import data, bucket as basket

def calc_beta(portfolio_return: pd.core.series.Series, market_return: pd.core.series.Series):
    
    # Find the intersection of indices
    index = np.intersect1d(market_return.index, portfolio_return.index)
    
    # Filter the log returns to only include the intersected indices
    market_return = market_return[index]
    portfolio_return = portfolio_return[index]

    # Calculate the covariance and variance
    cov = np.cov(market_return, portfolio_return)[0, 1]
    var = market_return.var()

    # Calculate beta
    beta = cov / var
    
    return beta

def VaR(sorted_array, confidence_level):
    index = round((1 - confidence_level)*len(sorted_array))
    
    return sorted_array[index]

def CVaR(sorted_array, confidence_level):
    index = round((1 - confidence_level)*len(sorted_array))
    
    avg_ret = np.mean(sorted_array[:index])
    
    return avg_ret

def initialize(stocks_list, start_date, end_date, benchmark_ticker, market_ticker, initial_amount, num_simulations, portfolio_minimum_weights, portfolio_maximum_weights, basket_minimum_weights, basket_maximum_weights):
    """
    Inputs:
        stocks_list: list of stocks
        start_date: start date of historical data
        end_date: end date of historical data
        benchmark_ticker: ticker of the chosen benchmark
        market_ticker: ticker of chosen market for the portfolio, used for beta calculation
        initial_amount: initially invested amount
        num_simulations: number of simulations to be performed
        portfolio_minimum_weights: minimum weight limits for all individual stocks
        portfolio_maximum_weights: minimum weight limits for all individual stocks
        basket_minimum_weights: minimum weight limits for all the baskets
        basket_maximum_weights: maximum weight limits for all the baskets
    """

    INITIAL_AMT_INVESTED = initial_amount
    NUM_SIMULATIONS = num_simulations

    ## FETCH DATA
    stock_prices, stock_dividend = data.fetch_data(stocks_list, start_date=start_date, end_date=end_date, dividend=True)

    ## CALCULATE LOG RETURNS
    log_returns = np.log(stock_prices / stock_prices.shift(1))[1:]
    log_returns_down_only = log_returns.applymap(lambda x: 0 if x>0 else x)

    ANNUAL_TRADING_DAYS = 252
    NUM_TRADING_DAYS = len(stock_prices)

    portfolio_weights = []
    basket_weights = []
    annual_portfolio_returns = []
    portfolio_risks = []

    for i in range(NUM_SIMULATIONS):
        
        # STEP-I: GENERATING RANDOM WEIGHTS
        w, basket_w = basket.get_adjusted_weights(
                                            portfolio_minimum_weights=portfolio_minimum_weights,
                                            portfolio_maximum_weights=portfolio_maximum_weights,
                                            basket_minimum_weights=basket_minimum_weights,
                                            basket_maximum_weights=basket_maximum_weights
        )
        
        portfolio_weights.append(w)
        basket_weights.append(basket_w)
        
        w = np.expand_dims(w, axis=0)
        
        # STEP-II: CALCULATING RETURNS
        p_returns = (log_returns.mean() @ w.T)[0]
        annual_portfolio_returns.append(p_returns * ANNUAL_TRADING_DAYS) # annualized portfolio return
        
        # STEP-III: CALCULATING RISKS
        p_risks = np.sqrt(w @ log_returns.cov() @ w.T)[0][0]
        portfolio_risks.append(p_risks * np.sqrt(ANNUAL_TRADING_DAYS)) # annualized portfolio risk

    portfolio_weights = np.array(portfolio_weights)
    basket_weights = np.array(basket_weights)
    annual_portfolio_returns = np.exp(annual_portfolio_returns) - 1
    portfolio_risks = np.array(portfolio_risks)

    ## SHARPE AND OPTIMISED WEIGHTS

    # risk-free rate of 10-yr bonds
    risk_free_rate = 0.07 # annual rate

    sharpe = (annual_portfolio_returns - risk_free_rate) / portfolio_risks

    ## OPTIMISED WEIGHTS
    index = np.argmax(sharpe)
    optimised_weights = portfolio_weights[index]

    ## NUM OF SHARES AND PORTFOLIO VALUE
    num_of_shares = INITIAL_AMT_INVESTED*optimised_weights/stock_prices.iloc[0,:]

    portfolio_value = np.sum(stock_prices*num_of_shares, axis=1)
    capital_gain = (portfolio_value[-1] / portfolio_value[0] - 1)*100
    portfolio_dividend = np.sum(stock_dividend*num_of_shares, axis=1)
    dividend_yield = 100*np.sum(portfolio_dividend) / INITIAL_AMT_INVESTED

    portfolio_log_returns = np.log(portfolio_value / portfolio_value.shift(1))[1:]

    ## MARKET
    market_prices = data.fetch_data(market_ticker, start_date=start_date, end_date=end_date)
    market_log_returns = np.log(market_prices / market_prices.shift(1))[1:]

    market_shares_bought = INITIAL_AMT_INVESTED / market_prices.iloc[0,:]
    market_value = np.sum(market_shares_bought * market_prices, axis=1)

    market_gain = (market_value[-1] / market_value[0] - 1)*100

    ## SHARPE
    sharpe_ratio = sharpe[index]

    ## SORTINO
    temp_w = np.expand_dims(optimised_weights, axis=0)
    sortino_risk = np.sqrt(temp_w @ log_returns_down_only.cov()*ANNUAL_TRADING_DAYS @ temp_w.T)[0][0]
    sortino_ratio = (annual_portfolio_returns[index] - risk_free_rate) / sortino_risk

    ## TREYNOR
    beta = calc_beta(portfolio_log_returns, market_log_returns.iloc[:,0])
    treynor_ratio = (annual_portfolio_returns[index] - risk_free_rate) / beta

    ## JENSON'S ALPHA
    annual_market_return = np.exp(market_log_returns.mean()*ANNUAL_TRADING_DAYS) - 1
    CAPM_expected_return = risk_free_rate + beta*(annual_market_return - risk_free_rate)
    jenson_alpha = annual_portfolio_returns[index] - CAPM_expected_return



    ## VaR CVaR calculation

    sorted_returns = portfolio_value.pct_change()[1:]
    sorted_returns = sorted_returns.sort_values()

    ## Value-at-Risk 
    var_at_90 = portfolio_value[-1]*VaR(sorted_returns, 0.90)
    var_at_95 = portfolio_value[-1]*VaR(sorted_returns, 0.95)
    var_at_99 = portfolio_value[-1]*VaR(sorted_returns, 0.99)

    ## Conditional Value-at-Risk
    cvar_at_90 = portfolio_value[-1]*CVaR(sorted_returns, 0.90)
    cvar_at_95 = portfolio_value[-1]*CVaR(sorted_returns, 0.95)
    cvar_at_99 = portfolio_value[-1]*CVaR(sorted_returns, 0.99)

    ## VaR from Monte Carlo Method
    probab = np.random.rand(NUM_SIMULATIONS)
    simulated_returns = stats.norm.ppf(probab, loc=sorted_returns.mean(), scale=sorted_returns.std())

    simulated_sorted_returns = np.sort(simulated_returns)
    sim_var_at_90 = portfolio_value[-1]*CVaR(simulated_sorted_returns, 0.90)
    sim_var_at_95 = portfolio_value[-1]*CVaR(simulated_sorted_returns, 0.95)
    sim_var_at_99 = portfolio_value[-1]*CVaR(simulated_sorted_returns, 0.99)



    ## BENCHMARK COMPARISON

    benchmark_prices = data.fetch_data(benchmark_ticker, start_date=start_date, end_date=end_date)

    benchmark_shares = INITIAL_AMT_INVESTED / benchmark_prices.iloc[0,:]
    benchmark_value = np.sum(benchmark_shares * benchmark_prices, axis=1)
    benchmark_log_returns = np.log(benchmark_value / benchmark_value.shift(1))[1:]

    benchmark_gain = (benchmark_value[-1] / benchmark_value[0] - 1)*100

    # Find the intersection of indices
    index_intersect = np.intersect1d(benchmark_log_returns.index, portfolio_log_returns.index)

    # Filter the log returns to only include the intersected indices
    bm_p_log_difference = portfolio_log_returns[index_intersect] - benchmark_log_returns[index_intersect]

    tracking_error_percent = bm_p_log_difference.std()*np.sqrt(ANNUAL_TRADING_DAYS)*100

    information_ratio = (capital_gain - benchmark_gain)*(ANNUAL_TRADING_DAYS / NUM_TRADING_DAYS) / tracking_error_percent



    ## CORRELATION MATRIX

    correlation_matrix = log_returns.corr()

    result_dict = {}
    result_dict['p_capital_gain'] = capital_gain
    result_dict['p_dividend_yield'] = dividend_yield
    result_dict['p_market_gain'] = market_gain
    result_dict['optimised_weights'] = optimised_weights
    result_dict['tickers_list'] = stocks_list
    result_dict['p_portfolio_returns'] = annual_portfolio_returns[index] * 100
    result_dict['portfolio_std'] = portfolio_risks[index]
    result_dict['portfolio_beta'] = beta
    result_dict['sharpe'] = sharpe_ratio
    result_dict['treynor'] = treynor_ratio
    result_dict['sortino'] = sortino_ratio
    result_dict['jenson'] = jenson_alpha
    result_dict['var'] = {'ninety_p':var_at_90, 'ninety_five_p':var_at_95, 'ninety_nine_p':var_at_99}
    result_dict['cvar'] = {'ninety_p':cvar_at_90, 'ninety_five_p':cvar_at_95, 'ninety_nine_p':cvar_at_99}
    result_dict['var_monte_carlo'] = {'ninety_p':sim_var_at_90, 'ninety_five_p':sim_var_at_95, 'ninety_nine_p':sim_var_at_99}
    result_dict['portfolio_value'] = portfolio_value
    result_dict['portfolio_dividend'] = portfolio_dividend
    result_dict['var_monte_carlo_simulated_returns'] = simulated_returns
    result_dict['p_benchmark_returns'] = benchmark_gain
    result_dict['p_tracking_error'] = tracking_error_percent
    result_dict['information_ratio'] = information_ratio
    result_dict['benchmark_value'] = benchmark_value
    result_dict['correlation_matrix'] = correlation_matrix
    result_dict['date'] = portfolio_value.reset_index()['Date']
    result_dict['capital_gain_per'] = (stock_prices.iloc[-1,:] / stock_prices.iloc[0,:] - 1)*100

    return result_dict

def equal_weighting(stocks_list, start_date, end_date, benchmark_ticker, market_ticker, initial_amount):
    """
    Inputs:
        stocks_list: list of stocks
        start_date: start date of historical data
        end_date: end date of historical data
        benchmark_ticker: ticker of the chosen benchmark
        market_ticker: ticker of chosen market for the portfolio, used for beta calculation
        initial_amount: initially invested amount
    """
    INITIAL_AMT_INVESTED = initial_amount
    NUM_SIMULATIONS = 1000
    ## FETCH DATA
    stock_prices, stock_dividend = data.fetch_data(stocks_list, start_date=start_date, end_date=end_date, dividend=True)

    ## CALCULATE LOG RETURNS
    log_returns = np.log(stock_prices / stock_prices.shift(1))[1:]
    log_returns_down_only = log_returns.applymap(lambda x: 0 if x>0 else x)

    ANNUAL_TRADING_DAYS = 252
    NUM_TRADING_DAYS = len(stock_prices)

    ## EQUAL WEIGHTING
    w = np.ones(stock_prices.shape[1])/stock_prices.shape[1] # shape: (10,)
    w = np.expand_dims(w, axis=0) # shape: (1,10)

    annual_portfolio_returns = np.exp(log_returns.mean() @ w.T * ANNUAL_TRADING_DAYS)[0] - 1
    portfolio_risk = np.sqrt(w @ log_returns.cov() @ w.T * ANNUAL_TRADING_DAYS).iloc[0,0]
    sortino_risk = np.sqrt(w @ log_returns_down_only.cov() @ w.T * ANNUAL_TRADING_DAYS).iloc[0,0]

    weights = w.flatten()
    risk_free_rate = 0.07
    sharpe_ratio = (annual_portfolio_returns - risk_free_rate) / portfolio_risk
    sortino_ratio = (annual_portfolio_returns - risk_free_rate) / sortino_risk

    num_of_shares = INITIAL_AMT_INVESTED*weights/stock_prices.iloc[0,:]

    portfolio_value = np.sum(stock_prices*num_of_shares, axis=1)
    capital_gain = (portfolio_value[-1] / portfolio_value[0] - 1)*100
    portfolio_dividend = np.sum(stock_dividend*num_of_shares, axis=1)
    dividend_yield = 100*np.sum(portfolio_dividend) / INITIAL_AMT_INVESTED

    portfolio_log_returns = np.log(portfolio_value / portfolio_value.shift(1))[1:]

    ## MARKET
    market_prices = data.fetch_data(market_ticker, start_date=start_date, end_date=end_date)
    market_log_returns = np.log(market_prices / market_prices.shift(1))[1:]

    market_shares_bought = INITIAL_AMT_INVESTED / market_prices.iloc[0,:]
    market_value = np.sum(market_shares_bought * market_prices, axis=1)

    market_gain = (market_value[-1] / market_value[0] - 1)*100

    beta = calc_beta(portfolio_log_returns, market_log_returns.iloc[:,0])

    treynor_ratio = (annual_portfolio_returns - risk_free_rate) / beta

    annual_market_return = np.exp(market_log_returns.mean() * ANNUAL_TRADING_DAYS) - 1
    CAPM_expected_return = risk_free_rate + beta*(annual_market_return - risk_free_rate)
    jenson_alpha = annual_portfolio_returns - CAPM_expected_return

    ## VaR CVaR calculation

    sorted_returns = portfolio_value.pct_change()[1:]
    sorted_returns = sorted_returns.sort_values()

    ## Value-at-Risk 
    var_at_90 = portfolio_value[-1]*VaR(sorted_returns, 0.90)
    var_at_95 = portfolio_value[-1]*VaR(sorted_returns, 0.95)
    var_at_99 = portfolio_value[-1]*VaR(sorted_returns, 0.99)

    ## Conditional Value-at-Risk
    cvar_at_90 = portfolio_value[-1]*CVaR(sorted_returns, 0.90)
    cvar_at_95 = portfolio_value[-1]*CVaR(sorted_returns, 0.95)
    cvar_at_99 = portfolio_value[-1]*CVaR(sorted_returns, 0.99)

    probab = np.random.rand(NUM_SIMULATIONS)
    simulated_returns = stats.norm.ppf(probab, loc=sorted_returns.mean(), scale=sorted_returns.std())

    simulated_sorted_returns = np.sort(simulated_returns)
    sim_var_at_90 = portfolio_value[-1]*CVaR(simulated_sorted_returns, 0.90)
    sim_var_at_95 = portfolio_value[-1]*CVaR(simulated_sorted_returns, 0.95)
    sim_var_at_99 = portfolio_value[-1]*CVaR(simulated_sorted_returns, 0.99)

    ## BENCHMARK COMPARISON

    benchmark_prices = data.fetch_data(benchmark_ticker, start_date=start_date, end_date=end_date)

    benchmark_shares = INITIAL_AMT_INVESTED / benchmark_prices.iloc[0,:]
    benchmark_value = np.sum(benchmark_shares * benchmark_prices, axis=1)
    benchmark_log_returns = np.log(benchmark_value / benchmark_value.shift(1))[1:]

    benchmark_gain = (benchmark_value[-1] / benchmark_value[0] - 1)*100

    # Find the intersection of indices
    index_intersect = np.intersect1d(benchmark_log_returns.index, portfolio_log_returns.index)

    # Filter the log returns to only include the intersected indices
    bm_p_log_difference = portfolio_log_returns[index_intersect] - benchmark_log_returns[index_intersect]

    tracking_error_percent = bm_p_log_difference.std()*np.sqrt(ANNUAL_TRADING_DAYS)*100

    information_ratio = (capital_gain - benchmark_gain)*(ANNUAL_TRADING_DAYS / NUM_TRADING_DAYS) / tracking_error_percent

    ## CORRELATION MATRIX
    correlation_matrix = log_returns.corr()

    result_dict = {}
    result_dict['p_capital_gain'] = capital_gain
    result_dict['p_dividend_yield'] = dividend_yield
    result_dict['p_market_gain'] = market_gain
    result_dict['optimised_weights'] = weights
    result_dict['tickers_list'] = stocks_list
    result_dict['p_portfolio_returns'] = annual_portfolio_returns * 100
    result_dict['portfolio_std'] = portfolio_risk
    result_dict['portfolio_beta'] = beta
    result_dict['sharpe'] = sharpe_ratio
    result_dict['treynor'] = treynor_ratio
    result_dict['sortino'] = sortino_ratio
    result_dict['jenson'] = jenson_alpha
    result_dict['var'] = {'ninety_p':var_at_90, 'ninety_five_p':var_at_95, 'ninety_nine_p':var_at_99}
    result_dict['cvar'] = {'ninety_p':cvar_at_90, 'ninety_five_p':cvar_at_95, 'ninety_nine_p':cvar_at_99}
    result_dict['var_monte_carlo'] = {'ninety_p':sim_var_at_90, 'ninety_five_p':sim_var_at_95, 'ninety_nine_p':sim_var_at_99}
    result_dict['portfolio_value'] = portfolio_value
    result_dict['portfolio_dividend'] = portfolio_dividend
    result_dict['var_monte_carlo_simulated_returns'] = simulated_returns
    result_dict['p_benchmark_returns'] = benchmark_gain
    result_dict['p_tracking_error'] = tracking_error_percent
    result_dict['information_ratio'] = information_ratio
    result_dict['benchmark_value'] = benchmark_value
    result_dict['correlation_matrix'] = correlation_matrix
    result_dict['date'] = portfolio_value.reset_index()['Date']
    result_dict['capital_gain_per'] = (stock_prices.iloc[-1,:] / stock_prices.iloc[0,:] - 1)*100
    
    return result_dict

def risk_parity(stocks_list, start_date, end_date, benchmark_ticker, market_ticker, initial_amount):
    """
    Inputs:
        stocks_list: list of stocks
        start_date: start date of historical data
        end_date: end date of historical data
        benchmark_ticker: ticker of the chosen benchmark
        market_ticker: ticker of chosen market for the portfolio, used for beta calculation
        initial_amount: initially invested amount
    """
    INITIAL_AMT_INVESTED = initial_amount
    NUM_SIMULATIONS = 1000
    ## FETCH DATA
    stock_prices, stock_dividend = data.fetch_data(stocks_list, start_date=start_date, end_date=end_date, dividend=True)

    ## CALCULATE LOG RETURNS
    log_returns = np.log(stock_prices / stock_prices.shift(1))[1:]
    log_returns_down_only = log_returns.applymap(lambda x: 0 if x>0 else x)

    ANNUAL_TRADING_DAYS = 252
    NUM_TRADING_DAYS = len(stock_prices)

    ## Equal Weighting
    w = 1 / np.array(stock_prices.std()) # shape: (10,)
    w /= w.sum()

    if w.sum() != 1.0:
        w /= w.sum() 
    w = np.expand_dims(w, axis=0) # shape: (1,10)

    annual_portfolio_returns = np.exp(log_returns.mean() @ w.T * ANNUAL_TRADING_DAYS)[0] - 1
    portfolio_risk = np.sqrt(w @ log_returns.cov() @ w.T * ANNUAL_TRADING_DAYS).iloc[0,0]
    sortino_risk = np.sqrt(w @ log_returns_down_only.cov() @ w.T * ANNUAL_TRADING_DAYS).iloc[0,0]

    weights = w.flatten()
    risk_free_rate = 0.07
    sharpe_ratio = (annual_portfolio_returns - risk_free_rate) / portfolio_risk
    sortino_ratio = (annual_portfolio_returns - risk_free_rate) / sortino_risk

    num_of_shares = INITIAL_AMT_INVESTED*weights/stock_prices.iloc[0,:]

    portfolio_value = np.sum(stock_prices*num_of_shares, axis=1)
    capital_gain = (portfolio_value[-1] / portfolio_value[0] - 1)*100
    portfolio_dividend = np.sum(stock_dividend*num_of_shares, axis=1)
    dividend_yield = 100*np.sum(portfolio_dividend) / INITIAL_AMT_INVESTED

    portfolio_log_returns = np.log(portfolio_value / portfolio_value.shift(1))[1:]

    ## MARKET
    market_prices = data.fetch_data(market_ticker, start_date=start_date, end_date=end_date)
    market_log_returns = np.log(market_prices / market_prices.shift(1))[1:]

    market_shares_bought = INITIAL_AMT_INVESTED / market_prices.iloc[0,:]
    market_value = np.sum(market_shares_bought * market_prices, axis=1)

    market_gain = (market_value[-1] / market_value[0] - 1)*100

    beta = calc_beta(portfolio_log_returns, market_log_returns.iloc[:,0])

    treynor_ratio = (annual_portfolio_returns - risk_free_rate) / beta

    annual_market_return = np.exp(market_log_returns.mean() * ANNUAL_TRADING_DAYS) - 1
    CAPM_expected_return = risk_free_rate + beta*(annual_market_return - risk_free_rate)
    jenson_alpha = annual_portfolio_returns - CAPM_expected_return

    ## VaR CVaR calculation

    sorted_returns = portfolio_value.pct_change()[1:]
    sorted_returns = sorted_returns.sort_values()

    ## Value-at-Risk 
    var_at_90 = portfolio_value[-1]*VaR(sorted_returns, 0.90)
    var_at_95 = portfolio_value[-1]*VaR(sorted_returns, 0.95)
    var_at_99 = portfolio_value[-1]*VaR(sorted_returns, 0.99)

    ## Conditional Value-at-Risk
    cvar_at_90 = portfolio_value[-1]*CVaR(sorted_returns, 0.90)
    cvar_at_95 = portfolio_value[-1]*CVaR(sorted_returns, 0.95)
    cvar_at_99 = portfolio_value[-1]*CVaR(sorted_returns, 0.99)

    probab = np.random.rand(NUM_SIMULATIONS)
    simulated_returns = stats.norm.ppf(probab, loc=sorted_returns.mean(), scale=sorted_returns.std())

    simulated_sorted_returns = np.sort(simulated_returns)
    sim_var_at_90 = portfolio_value[-1]*CVaR(simulated_sorted_returns, 0.90)
    sim_var_at_95 = portfolio_value[-1]*CVaR(simulated_sorted_returns, 0.95)
    sim_var_at_99 = portfolio_value[-1]*CVaR(simulated_sorted_returns, 0.99)

    ## BENCHMARK COMPARISON

    benchmark_prices = data.fetch_data(benchmark_ticker, start_date=start_date, end_date=end_date)

    benchmark_shares = INITIAL_AMT_INVESTED / benchmark_prices.iloc[0,:]
    benchmark_value = np.sum(benchmark_shares * benchmark_prices, axis=1)
    benchmark_log_returns = np.log(benchmark_value / benchmark_value.shift(1))[1:]

    benchmark_gain = (benchmark_value[-1] / benchmark_value[0] - 1)*100

    # Find the intersection of indices
    index_intersect = np.intersect1d(benchmark_log_returns.index, portfolio_log_returns.index)

    # Filter the log returns to only include the intersected indices
    bm_p_log_difference = portfolio_log_returns[index_intersect] - benchmark_log_returns[index_intersect]

    tracking_error_percent = bm_p_log_difference.std()*np.sqrt(ANNUAL_TRADING_DAYS)*100

    information_ratio = (capital_gain - benchmark_gain)*(ANNUAL_TRADING_DAYS / NUM_TRADING_DAYS) / tracking_error_percent

    ## CORRELATION MATRIX
    correlation_matrix = log_returns.corr()

    result_dict = {}
    result_dict['p_capital_gain'] = capital_gain
    result_dict['p_dividend_yield'] = dividend_yield
    result_dict['p_market_gain'] = market_gain
    result_dict['optimised_weights'] = weights
    result_dict['tickers_list'] = stocks_list
    result_dict['p_portfolio_returns'] = annual_portfolio_returns * 100
    result_dict['portfolio_std'] = portfolio_risk
    result_dict['portfolio_beta'] = beta
    result_dict['sharpe'] = sharpe_ratio
    result_dict['treynor'] = treynor_ratio
    result_dict['sortino'] = sortino_ratio
    result_dict['jenson'] = jenson_alpha
    result_dict['var'] = {'ninety_p':var_at_90, 'ninety_five_p':var_at_95, 'ninety_nine_p':var_at_99}
    result_dict['cvar'] = {'ninety_p':cvar_at_90, 'ninety_five_p':cvar_at_95, 'ninety_nine_p':cvar_at_99}
    result_dict['var_monte_carlo'] = {'ninety_p':sim_var_at_90, 'ninety_five_p':sim_var_at_95, 'ninety_nine_p':sim_var_at_99}
    result_dict['portfolio_value'] = portfolio_value
    result_dict['portfolio_dividend'] = portfolio_dividend
    result_dict['var_monte_carlo_simulated_returns'] = simulated_returns
    result_dict['p_benchmark_returns'] = benchmark_gain
    result_dict['p_tracking_error'] = tracking_error_percent
    result_dict['information_ratio'] = information_ratio
    result_dict['benchmark_value'] = benchmark_value
    result_dict['correlation_matrix'] = correlation_matrix
    result_dict['date'] = portfolio_value.reset_index()['Date']
    result_dict['capital_gain_per'] = (stock_prices.iloc[-1,:] / stock_prices.iloc[0,:] - 1)*100

    return result_dict