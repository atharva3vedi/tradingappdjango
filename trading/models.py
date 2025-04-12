# trading/models.py

from django.db import models
import uuid

class TradingSession(models.Model):
    SESSION_MODES = [
        ('testing', 'Testing'),
        ('intraday', 'Intraday'),
        ('daily', 'Daily'),
    ]
    
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    stock_symbol = models.CharField(max_length=15)
    initial_capital = models.FloatField()
    mode = models.CharField(max_length=10, choices=SESSION_MODES)
    num_days = models.IntegerField()
    news_query = models.CharField(max_length=255, blank=True, null=True)
    model_name = models.CharField(max_length=255, blank=True, null=True)
    intraday = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.stock_symbol} - {self.session_id}"

class TradingAction(models.Model):
    session = models.ForeignKey(TradingSession, on_delete=models.CASCADE, related_name='actions')
    step = models.IntegerField()
    price = models.FloatField()
    action = models.CharField(max_length=10)
    portfolio_value = models.FloatField()
    timestamp = models.DateTimeField()
    
    class Meta:
        ordering = ['step']
        
    def __str__(self):
        return f"{self.session.stock_symbol} - Step {self.step} - {self.action}"

class PortfolioHistory(models.Model):
    session = models.ForeignKey(TradingSession, on_delete=models.CASCADE, related_name='portfolio_history')
    value = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']
        
    def __str__(self):
        return f"{self.session.stock_symbol} - {self.timestamp} - {self.value}"