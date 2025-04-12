# trading/views.py
# Django views for handling trading requests

import uuid
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import os
import asyncio
import threading
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError

from .session_manager import get_session_manager
from .trading_session import TradingSession
from .forms import TradingRequestForm
from .models import TradingSession as TradingSessionModel

@csrf_exempt
@require_http_methods(["POST"])
def start_trading(request):
    """Start a new trading session"""
    try:
        # Parse the request data
        data = json.loads(request.body)
        form = TradingRequestForm(data)
        
        if not form.is_valid():
            return JsonResponse({'error': form.errors}, status=400)
        
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Get form data
        stock_symbol = form.cleaned_data['stock_symbol']
        initial_capital = form.cleaned_data['initial_capital']
        mode = form.cleaned_data['mode']
        num_days = form.cleaned_data['num_days']
        news_query = form.cleaned_data.get('news_query')
        model_name = form.cleaned_data.get('model_name')
        intraday_str = form.cleaned_data.get('intraday')
        
        # Convert the string field for intraday to a boolean
        intraday_bool = True if intraday_str and intraday_str.lower() == "yes" else False
        
        # Create a new trading session
        session = TradingSession(
            session_id=session_id,
            symbol=stock_symbol,
            initial_capital=initial_capital,
            mode=mode,
            num_days=num_days,
            news_query=news_query,
            model_name=model_name,
            intraday=intraday_bool
        )
        
        # Create and save the database record
        db_session = TradingSessionModel.objects.create(
            session_id=session_id,
            stock_symbol=stock_symbol,
            initial_capital=initial_capital,
            mode=mode,
            num_days=num_days,
            news_query=news_query,
            model_name=model_name,
            intraday=intraday_bool
        )
        
        # Add the session to the session manager
        session_manager = get_session_manager()
        session_manager.add_session(session)
        
        # Start the trading session in a background thread
        thread = threading.Thread(target=session.run)
        thread.start()
        
        # Store the thread in the session for later reference
        session.thread = thread
        
        return JsonResponse({
            'session_id': session_id,
            'message': 'Trading session started'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def stop_trading(request, session_id):
    """Stop a trading session"""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)
        
        if not session:
            return JsonResponse({'error': 'Session not found'}, status=404)
        
        session.stop()
        return JsonResponse({'message': 'Session stopped'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_history(request, session_id):
    """Get the history of a trading session"""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)
        
        if not session:
            return JsonResponse({'error': 'Session not found'}, status=404)
        
        return JsonResponse({
            'actions': session.get_actions(),
            'portfolio_history': session.get_portfolio_history()
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_session_status(request, session_id):
    """Get the status of a trading session"""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            return JsonResponse({'error': 'Session not found'}, status=404)

        # Check if the thread is still alive
        thread_alive = hasattr(session, 'thread') and session.thread.is_alive()
        
        # Force a fresh portfolio update
        portfolio_value = session.current_capital + sum(qty * session.prices[-1] for qty, _ in session.inventory)

        return JsonResponse({
            'is_running': session._is_running and thread_alive,
            'current_step': len(session.portfolio_history),
            'total_steps': len(session.prices) - 1 if session.prices else 0,
            'current_capital': portfolio_value,
            'inventory': session.inventory,
            'thread_alive': thread_alive
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
