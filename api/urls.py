from django.urls import path
from .views import *

urlpatterns = [
    path('tickers_data/', get_tickers_data, name='get_tickers_data'),
    path('tickers_equity/', get_equity_tickers, name='get_equity_tickers'),
    path('tickers_commodities/', get_commodities_tickers, name='get_commodities_tickers'),
    path('tickers_reit/', get_REIT_tickers, name='get_REIT_tickers'),
    path('tickers_t_notes/', get_t_notes_tickers, name='get_t_notes_tickers'),
    path('tickers_crypto/', get_crypto_tickers, name='get_crypto_tickers'),
    path('search_t_notes/', search_t_notes, name='search_t_notes'),
    path('search_crypto/', search_crypto, name='search_crypto'),
    path('search_commodities/', search_commodities, name='search_commodities'),
    path('search_reit/', search_REIT, name='search_REIT'),
    path('search_equity/', search_equity, name='search_equity'),
    path('data_analysis/', analyze_data, name = 'data_analysis'),

]
