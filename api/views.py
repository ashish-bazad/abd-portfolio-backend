import pandas as pd
import numpy as np
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .tools import first_page, second_page
from .tools.equities import *
from .tools.commodities import *
from .tools.REIT import *
from .tools.t_notes import *
from .tools.crypto import *
from .tools.nifty50 import *

@api_view(['GET'])
def search_equity(request):
    to_search = str(request.GET.get('search', "")).upper()
    result = {}
    for key, value in equity_tickers.items():
        if key.startswith(to_search) or (value.lower()).__contains__(to_search.lower()):
            result[key] = value
    return Response({"tickers": result}, status=status.HTTP_200_OK)

@api_view(['GET'])
def search_REIT(request):
    to_search = str(request.GET.get('search', "")).upper()
    result = {}
    for key, value in reit_tickers.items():
        if key.startswith(to_search) or (value.lower()).__contains__(to_search.lower()):
            result[key] = value
    return Response({"tickers": result}, status=status.HTTP_200_OK)

@api_view(['GET'])
def search_t_notes(request):
    to_search = str(request.GET.get('search', "")).upper()
    result = {}
    for key, value in t_notes_tickers.items():
        if key.startswith(to_search) or (value.lower()).__contains__(to_search.lower()):
            result[key] = value
    return Response({"tickers": result}, status=status.HTTP_200_OK)

@api_view(['GET'])
def search_crypto(request):
    to_search = str(request.GET.get('search', "")).upper()
    result = {}
    for key, value in crypto_tickers.items():
        if key.startswith(to_search) or (value.lower()).__contains__(to_search.lower()):
            result[key] = value
    return Response({"tickers": result}, status=status.HTTP_200_OK)

@api_view(['GET'])
def search_commodities(request):
    to_search = str(request.GET.get('search', "")).upper()
    result = {}
    for key, value in commodities_tickers.items():
        if key.startswith(to_search) or (value.lower()).__contains__(to_search.lower()):
            result[key] = value
    return Response({"tickers": result}, status=status.HTTP_200_OK)

@api_view(['GET'])
def get_equity_tickers(request):
    return Response({"tickers": nifty50_tickers}, status=status.HTTP_200_OK)

@api_view(['GET'])
def get_commodities_tickers(request):
    return Response({"tickers": commodities_tickers}, status=status.HTTP_200_OK)

@api_view(['GET'])
def get_REIT_tickers(request):
    return Response({"tickers": reit_tickers}, status=status.HTTP_200_OK)

@api_view(['GET'])
def get_t_notes_tickers(request):
    return Response({"tickers": t_notes_tickers}, status=status.HTTP_200_OK)

@api_view(['GET'])
def get_crypto_tickers(request):
    return Response({"tickers": crypto_tickers}, status=status.HTTP_200_OK)

@api_view(['GET'])
def get_tickers_data(request):
    tickers_list = request.GET.get('tickers', None).split(',')
    print(tickers_list)
    period = str(request.GET.get('period', '1y'))
    table_data, price_data, volatility_data = {}, {}, {}
    try:
        table_data, price_data, volatility_data = first_page.initialize(tickers_list,  period=period)
    except Exception as e:
        print(e)
    return Response({"table_data": table_data, "price_data": price_data, "volatility_data": volatility_data}, status=status.HTTP_200_OK)

@api_view(['POST'])
def analyze_data(request):
    data = request.data
    start_date = data.get('start_date', None)
    end_date = data.get('end_date', None)
    equity_bucket = data.get('equity_bucket', None)
    commodities_bucket = data.get('commodities_bucket', None)
    reit_bucket = data.get('reit_bucket', None)
    t_notes_bucket = data.get('t_notes_bucket', None)
    crypto_bucket = data.get('crypto_bucket', None)
    buckets_min_weights = data.get('buckets_min_weights', None)
    buckets_max_weights = data.get('buckets_max_weights', None)
    benchmark_ticker = data.get('benchmark_ticker', None)
    market_ticker = data.get('market_ticker', None)
    number_of_simulations = int(data.get('number_of_simulations', None))
    initial_amount = int(data.get('initial_amount', None))

    tickers_list = []
    portfolio_minimum_weights = []
    portfolio_maximum_weights = []
    buckets_minimum_weights = [i / 100 for i in buckets_min_weights if i != None]
    buckets_maximum_weights = [buckets_max_weights[i] / 100 for i in range(len(buckets_max_weights)) if buckets_min_weights[i] != None]
    if len(equity_bucket) > 0:
        tmp_min = []
        tmp_max = []
        for i in equity_bucket:
            tickers_list.append(i[0][0])
            tmp_min.append(i[1] / 100)
            tmp_max.append(i[2] / 100)
        portfolio_minimum_weights.append(tmp_min)
        portfolio_maximum_weights.append(tmp_max)

    if len(commodities_bucket) > 0:
        tmp_min = []
        tmp_max = []
        for i in commodities_bucket:
            tickers_list.append(i[0][0])
            tmp_min.append(i[1] / 100)
            tmp_max.append(i[2] / 100)
        portfolio_minimum_weights.append(tmp_min)
        portfolio_maximum_weights.append(tmp_max)

    if len(reit_bucket) > 0:
        tmp_min = []
        tmp_max = []
        for i in reit_bucket:
            tickers_list.append(i[0][0])
            tmp_min.append(i[1] / 100)
            tmp_max.append(i[2] / 100)
        portfolio_minimum_weights.append(tmp_min)
        portfolio_maximum_weights.append(tmp_max)

    if len(t_notes_bucket) > 0:
        tmp_min = []
        tmp_max = []
        for i in t_notes_bucket:
            tickers_list.append(i[0][0])
            tmp_min.append(i[1] / 100)
            tmp_max.append(i[2] / 100)
        portfolio_minimum_weights.append(tmp_min)
        portfolio_maximum_weights.append(tmp_max)

    if len(crypto_bucket) > 0:
        tmp_min = []
        tmp_max = []
        for i in crypto_bucket:
            tickers_list.append(i[0][0])
            tmp_min.append(i[1] / 100)
            tmp_max.append(i[2] / 100)
        portfolio_minimum_weights.append(tmp_min)
        portfolio_maximum_weights.append(tmp_max)

    result_dict = second_page.initialize(tickers_list, start_date, end_date, benchmark_ticker, market_ticker, initial_amount, number_of_simulations, portfolio_minimum_weights, portfolio_maximum_weights, buckets_minimum_weights, buckets_maximum_weights)
    result_dict2 = second_page.equal_weighting(tickers_list, start_date, end_date, benchmark_ticker, market_ticker, initial_amount)
    result_dict3 = second_page.risk_parity(tickers_list, start_date, end_date, benchmark_ticker, market_ticker, initial_amount)
    
    return Response({"results_0": result_dict, "results_1": result_dict2, "results_2": result_dict3}, status=status.HTTP_200_OK)