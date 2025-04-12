# trading/serializers.py

from rest_framework import serializers
from .models import TradingSession, TradingAction, PortfolioHistory

class TradingRequestSerializer(serializers.Serializer):
    mode = serializers.ChoiceField(choices=['testing', 'intraday', 'daily'])
    intraday = serializers.CharField(required=False, allow_null=True)
    stock_symbol = serializers.CharField(max_length=15)
    num_days = serializers.IntegerField()
    initial_capital = serializers.FloatField()
    news_query = serializers.CharField(required=False, allow_null=True)
    model_name = serializers.CharField(required=False, allow_null=True)
    
    def validate_stock_symbol(self, value):
        if not value.replace('.', '').isalpha():
            raise serializers.ValidationError("Stock symbol must contain only letters and dots")
        return value.upper()

class TradingResponseSerializer(serializers.Serializer):
    session_id = serializers.UUIDField()
    message = serializers.CharField()

class TradingSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = TradingSession
        fields = '__all__'

class TradingActionSerializer(serializers.ModelSerializer):
    class Meta:
        model = TradingAction
        fields = '__all__'

class PortfolioHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = PortfolioHistory
        fields = '__all__'