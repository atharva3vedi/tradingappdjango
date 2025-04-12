# trading/trading_session.py

import os
import time
import datetime
import math
import random
import numpy as np
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from threading import Lock
import tensorflow as tf
import json
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from sklearn.preprocessing import StandardScaler
from .utils import (
    getStateEnhanced,
    action_to_str,
    get_sentiment_score,
    fetch_stock_data,
    determine_lot_size,
    calculate_sharpe_ratio
)
from .models import TradingSession as TradingSessionModel, TradingAction, PortfolioHistory
from django.utils import timezone
from .model_manager import model_manager

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")

from keras.models import Sequential, load_model
from keras.layers import Dense, Lambda, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.saving import register_keras_serializable

@register_keras_serializable()
def multiply_by_2(x):
    return 2 * x

class Agent:
    def __init__(self, state_size, model=None, model_type="improved", is_eval=True):
        self.state_size = state_size
        self.action_size = 3
        self.memory = []
        self.inventory = []
        self.is_eval = is_eval
        self.model_type = model_type
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.batch_size = 32

        if model is not None:
            self.model = model
        else:
            self.model = self._build_improved_model()

    def _build_improved_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(self.action_size, activation='tanh')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
        return model

    def act(self, state):
        q_vals = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size), q_vals.tolist()
        return int(np.argmax(q_vals)), q_vals.tolist()

    def expReplay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([exp[0] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        
        current_qs = self.model.predict(states, verbose=0)
        future_qs = self.model.predict(next_states, verbose=0)
        
        x, y = [], []
        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward if done else reward + self.gamma * np.amax(future_qs[index])
            current_q = current_qs[index]
            current_q[action] = target
            x.append(state)
            y.append(current_q)
            
        self.model.fit(np.array(x), np.array(y), batch_size=batch_size, verbose=0)

class TradingSession:
    def __init__(self, session_id: str, symbol: str, initial_capital: float, mode: str,
                 num_days: int = 30, news_query: str = None, model_name: str = None, intraday: bool = False):
        self.session_id = session_id
        # Add .NS suffix if not present for Indian stocks
        self.symbol = symbol if symbol.endswith('.NS') else f"{symbol}.NS"
        self.initial_capital = initial_capital
        self.mode = mode
        self.num_days = num_days
        self.news_query = news_query
        self.model_name = model_name
        self.intraday = intraday
        self._is_running = False
        self.logs = []
        self.history_logs = []
        self.portfolio_history = []
        self.lock = Lock()
        self.inventory = []
        self.current_capital = initial_capital
        self.channel_layer = get_channel_layer()

        # Determine the fetch mode
        fetch_mode = "intraday" if (mode == "testing" and intraday) or mode == "intraday" else "daily"
        try:
            self.prices, self.dates = fetch_stock_data(self.symbol, fetch_mode, num_days=self.num_days)
            if not self.prices:
                raise ValueError(f"No stock data available for symbol {self.symbol}. Please check if the symbol is correct.")
        except Exception as e:
            raise ValueError(f"Error fetching data for {self.symbol}: {str(e)}")
            
        # Set the state size
        self.window_size = 10
        self.state_size = (self.window_size - 1) + 7
        
        # Get model and scaler from model manager
        self.model = model_manager.get_model(self.symbol)
        self.scaler = model_manager.get_scaler(self.symbol)
        
        # Initialize Agent with the pre-trained model
        self.agent = Agent(state_size=self.state_size, model=self.model)
        
        # Add metric tracking
        self.portfolio_history = []
        self.actions = []
        self.returns = []
        self.volatilities = []
        self.sharpe_ratios = []

    def run(self):
        """Updated trading loop with proper state handling"""
        self._is_running = True
        l = len(self.prices) - 1
        
        self._log_progress("Session started", 0, "system")

        # Initialize progress tracking
        global_step = 0
        N_updates = math.log(self.agent.epsilon_min/self.agent.epsilon) / math.log(self.agent.epsilon_decay)
        update_interval = int(round((l * 1) / N_updates))  # Assume 1 episode for testing

        for t in range(l):
            if not self._is_running:
                break

            self._log_progress(f"Processing step {t}/{l}", t, "step_start")

            # Proper date handling for sentiment
            current_date = self.dates[t].strftime('%Y-%m-%d')
            sentiment = 0.5  # Default neutral sentiment
            
            if self.intraday:
                # Intraday mode: update sentiment daily
                if t == 0 or (self.dates[t].date() != self.dates[t-1].date()):
                    sentiment = get_sentiment_score(f"{self.news_query} {current_date}")
            else:
                # Daily mode: update every 64 steps
                if t % 64 == 0:
                    sentiment = get_sentiment_score(f"{self.news_query} {current_date}")

            # Get proper state
            state, indicators = getStateEnhanced(self.prices, t, self.window_size, sentiment)
            state = self.scaler.transform(state.reshape(1, -1))[0]
            
            # Agent decision
            action, q_vals = self.agent.act(state)
            print(f"Step {t}: Action={action_to_str(action)}, Q-values={q_vals}")
            
            # Execute trade
            profit = 0
            if action == 1:  # Buy
                lot_size = determine_lot_size(self.current_capital, self.prices[t])
                if self.current_capital >= self.prices[t] * lot_size:
                    self.current_capital -= self.prices[t] * lot_size
                    self.inventory.append((lot_size, self.prices[t]))
                    print(f"Step {t}: Bought {lot_size} shares at {self.prices[t]}, new capital={self.current_capital}")
            elif action == 2 and self.inventory:  # Sell
                qty, purchase_price = self.inventory.pop(0)
                profit = (self.prices[t] - purchase_price) * qty
                self.current_capital += self.prices[t] * qty
                print(f"Step {t}: Sold {qty} shares at {self.prices[t]}, profit={profit}, new capital={self.current_capital}")
                
                # Calculate metrics
                profit_percentage = ((self.prices[t] - purchase_price) / purchase_price) * 100
                position_returns = [profit_percentage]  # Simplified for example
                vol = indicators[4]
                sharpe = calculate_sharpe_ratio(np.array(position_returns))
                
                self.returns.append(profit_percentage)
                self.volatilities.append(vol)
                self.sharpe_ratios.append(sharpe)

            # Store portfolio state
            held_value = sum(qty * self.prices[t] for qty, _ in self.inventory)
            portfolio_value = self.current_capital + held_value
            self.portfolio_history.append(portfolio_value)
            
            # Experience replay (for training mode)
            if self.mode == "training":
                next_state, _ = getStateEnhanced(self.prices, t+1, self.window_size, sentiment)
                next_state = self.scaler.transform(next_state.reshape(1, -1))[0]
                done = (t == l - 1)
                self.agent.memory.append((state, action, profit, next_state, done))
                
                if len(self.agent.memory) > self.agent.batch_size:
                    self.agent.expReplay(self.agent.batch_size)
                
                # Epsilon decay
                global_step += 1
                if global_step % update_interval == 0:
                    self.agent.epsilon = max(self.agent.epsilon * self.agent.epsilon_decay, 
                                           self.agent.epsilon_min)

            # WebSocket logging and DB storage
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "step": t,
                "price": self.prices[t],
                "action": action_to_str(action),
                "q_values": q_vals,
                "portfolio_value": round(portfolio_value, 2),
                "capital": round(self.current_capital, 2),
                "shares_held": sum(qty for qty, _ in self.inventory),
                "indicators": {
                    "sma": round(indicators[0], 2),
                    "ema": round(indicators[1], 2),
                    "rsi": round(indicators[2], 2),
                    "macd": round(indicators[3], 2),
                    "volatility": round(indicators[4], 4),
                    "sentiment": round(sentiment, 4)
                }
            }
            
            # Store the action in the database
            TradingAction.objects.create(
                session_id=self.session_id,
                step=t,
                price=self.prices[t],
                action=action_to_str(action),
                portfolio_value=portfolio_value,
                timestamp=timezone.now()
            )
            
            # Store portfolio value
            PortfolioHistory.objects.create(
                session_id=self.session_id,
                value=portfolio_value,
                timestamp=timezone.now()
            )
            
            with self.lock:
                self.history_logs.append(log_entry)
                self.actions.append(log_entry)
            
            self._log_progress(f"Action taken: {action_to_str(action)}", t, "action")
            
        # Final reporting
        self._generate_report()
    
    def _log_progress(self, message: str, step: int, log_type: str):
        log_entry = {
            "type": log_type,
            "step": step,
            "message": message,
            "timestamp": datetime.datetime.now().isoformat()
        }
        with self.lock:
            self.logs.append(log_entry)
            
        # Send WebSocket message
        try:
            channel_name = f"session_{self.session_id}"
            async_to_sync(self.channel_layer.group_send)(
                channel_name,
                {
                    "type": "trading_message",
                    "message": log_entry
                }
            )
        except Exception as e:
            print(f"Error sending WebSocket message: {e}")

    def _generate_report(self):
        """Creates final report with metrics"""
        # Use non-interactive backend for Matplotlib
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        plt.figure(figsize=(15,5))
        plt.plot(self.portfolio_history, label='Portfolio Value')
        plt.title(f"Trading Session {self.session_id} Results")
        plt.legend()
        image_path = f"session_{self.session_id}.png"
        plt.savefig(image_path)
        plt.close()
        
        # Update session with additional info
        session_model = TradingSessionModel.objects.get(session_id=self.session_id)
        session_model.save()
        
    def stop(self):
        self._is_running = False

    def new_logs_available(self) -> bool:
        with self.lock:
            return len(self.logs) > 0

    def get_latest_log(self):
        with self.lock:
            if self.logs:
                return self.logs.pop(0)
            return {}

    def get_actions(self):
        with self.lock:
            return list(self.history_logs)

    def get_portfolio_history(self):
        with self.lock:
            return list(self.portfolio_history)

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def add_session(self, session: TradingSession):
        self.sessions[session.session_id] = session

    def get_session(self, session_id: str) -> TradingSession:
        if session_id not in self.sessions:
            from django.http import Http404
            raise Http404("Session not found")
        return self.sessions[session_id]

# Global session manager
session_manager = SessionManager()