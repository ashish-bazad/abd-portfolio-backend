import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

## PLOT STOCK PRICES
#1 single stock

def plot_price(price_data, dividend_data=None, stock_name=None, figure=None, percent_scale=False, initial_price_ref=False):
    
    """Plotly graph, make sure you use plotly elements"""
    
    if figure is None:
        fig = go.Figure()
    else:
        fig = figure
    
    if stock_name is None:
        stock = price_data.name
    else:
        stock = stock_name
    
    if percent_scale:
        price_data = 100*(price_data/price_data[0] - 1)
    
    ## PLOTTING THE PRICES
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data, mode='lines', name=stock))
    
    if dividend_data is not None:
        ## PLOTTING THE DIVIDEND ROLLOUT
        dividend_rollout_days = dividend_data[dividend_data != 0].index
        price_when_dividend = price_data.loc[dividend_rollout_days]
        dividend_amounts = dividend_data.loc[dividend_rollout_days]
        
        # Constructing hover text without date
        hover_text = [f"Dividend Amount: {amount:.2f}" for amount in dividend_amounts]
        
        fig.add_trace(go.Scatter(x=dividend_rollout_days, y=price_when_dividend*1.05, 
                                 mode='markers', marker=dict(symbol='circle', size=10), 
                                 showlegend=False, name="{} Dividend".format(stock),
                                 hovertext=hover_text))
    if initial_price_ref:
        fig.add_trace(go.Scatter(x=price_data.index, 
                                 y=np.ones(len(price_data))*price_data.iloc[0], 
                                 mode='lines', line=dict(width=2,dash='dash'), showlegend=False))

    
    fig.update_layout(
        title='Stock Price with Dividend Rollout',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        hovermode="x unified"
        #hoverlabel=dict(bgcolor='white', font=dict(color='black'))
    )
    
    return fig

#2 plot multiple stock_price

def plot_multi_price(df_price, df_dividend=None, figure=None, percent_scale=False):
    
    """input dataframes"""
    
    if figure is None:
        fig = go.Figure()
    else:
        fig = figure
    
    if df_dividend is not None:
        if percent_scale:
            for stock in df_price.columns:
                fig = plot_price(df_price[stock], df_dividend[stock], figure=fig, percent_scale=True)
        else:
            for stock in df_price.columns:
                fig = plot_price(df_price[stock], df_dividend[stock], figure=fig)

    else:
        if percent_scale:
            for stock in df_price.columns:
                fig = plot_price(df_price[stock], figure=fig, percent_scale=True)
        else:
            for stock in df_price.columns:
                fig = plot_price(df_price[stock], figure=fig)
    if percent_scale:
        fig.add_trace(go.Scatter(x=df_price.index, 
                             y=np.zeros(len(df_price)), 
                             mode='lines', line=dict(width=2,dash='dash'), showlegend=False))

    return fig