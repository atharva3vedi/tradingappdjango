# trading/tests.py

from django.test import TestCase
from django.urls import reverse
import json
from uuid import UUID
from .models import TradingSession
from .trading_session import TradingSession as TradingSessionImplementation
from .session_manager import get_session_manager


class TradingViewsTestCase(TestCase):
    def test_start_trading_view(self):
        """Test that the start_trading view correctly creates a new session"""
        test_data = {
            "mode": "testing",
            "stock_symbol": "AAPL",
            "num_days": 30,
            "initial_capital": 10000.0,
            "intraday": "no",
            "news_query": "Apple Inc"
        }
        
        response = self.client.post(
            reverse('start_trading'),
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('session_id', data)
        
        # Try to validate the UUID
        try:
            UUID(data['session_id'])
            is_valid_uuid = True
        except ValueError:
            is_valid_uuid = False
        
        self.assertTrue(is_valid_uuid)
        self.assertIn('message', data)
        self.assertEqual(data['message'], 'Trading session started')


class TradingSessionTestCase(TestCase):
    def test_session_initialization(self):
        """Test that a trading session initializes correctly"""
        session = TradingSessionImplementation(
            session_id='test-session',
            symbol='AAPL',
            initial_capital=10000.0,
            mode='testing',
            num_days=10,
            intraday=False
        )
        
        self.assertEqual(session.session_id, 'test-session')
        self.assertEqual(session.symbol, 'AAPL')
        self.assertEqual(session.initial_capital, 10000.0)
        self.assertEqual(session.mode, 'testing')
        self.assertEqual(session.current_capital, 10000.0)
        self.assertEqual(len(session.inventory), 0)


class SessionManagerTestCase(TestCase):
    def test_session_manager(self):
        """Test that the session manager correctly stores and retrieves sessions"""
        session_manager = get_session_manager()
        
        # Create a test session
        session = TradingSessionImplementation(
            session_id='test-manager-session',
            symbol='MSFT',
            initial_capital=5000.0,
            mode='testing',
            num_days=5,
            intraday=False
        )
        
        # Add session to manager
        session_manager.add_session(session)
        
        # Retrieve session from manager
        retrieved_session = session_manager.get_session('test-manager-session')
        
        self.assertEqual(retrieved_session.session_id, 'test-manager-session')
        self.assertEqual(retrieved_session.symbol, 'MSFT')
        self.assertEqual(retrieved_session.initial_capital, 5000.0)