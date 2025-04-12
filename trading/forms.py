# trading/forms.py
# Django forms for validating trading requests

from django import forms
from django.core.validators import RegexValidator, MinValueValidator

class TradingRequestForm(forms.Form):
    MODE_CHOICES = [
        ('testing', 'Testing'),
        ('intraday', 'Intraday'),
        ('daily', 'Daily')
    ]
    
    mode = forms.ChoiceField(choices=MODE_CHOICES)
    intraday = forms.CharField(required=False)
    stock_symbol = forms.CharField(
        max_length=15,
        validators=[RegexValidator(r'^[A-Z\.]{1,15}$', 'Enter a valid stock symbol (capital letters only)')]
    )
    num_days = forms.IntegerField(validators=[MinValueValidator(1)])
    initial_capital = forms.FloatField(validators=[MinValueValidator(100)])
    news_query = forms.CharField(required=False)
    model_name = forms.CharField(required=False)