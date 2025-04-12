# trading/admin.py

from django.contrib import admin
from .models import TradingSession, TradingAction, PortfolioHistory


@admin.register(TradingSession)
class TradingSessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'stock_symbol', 'mode', 'initial_capital', 'created_at')
    list_filter = ('mode', 'created_at')
    search_fields = ('stock_symbol', 'session_id')
    readonly_fields = ('session_id', 'created_at')


@admin.register(TradingAction)
class TradingActionAdmin(admin.ModelAdmin):
    list_display = ('session', 'step', 'action', 'price', 'timestamp')
    list_filter = ('action', 'timestamp')
    search_fields = ('session__stock_symbol', 'action')


@admin.register(PortfolioHistory)
class PortfolioHistoryAdmin(admin.ModelAdmin):
    list_display = ('session', 'value', 'timestamp')
    list_filter = ('timestamp',)
    search_fields = ('session__stock_symbol',)