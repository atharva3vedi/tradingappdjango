# trading/utils.py
# Django-compatible version of the utils.py file

import os
import random
import numpy as np
import datetime
import yfinance as yf
import talib
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
from transformers import PegasusTokenizer, TFPegasusForConditionalGeneration, pipeline
import warnings
import json
from django.conf import settings
import joblib

# --- Stock Data & Technical Indicators Functions ---

def fetch_stock_data(symbol: str, mode: str, num_days: int = 30):
    """
    Fetch historical stock data using yfinance.
    For mode 'intraday', a 1-minute interval is used; for 'daily', daily data is fetched.
    """
    try:
        if mode == "intraday":
            # For intraday, we need to use a shorter period and 1m interval
            data = yf.download(symbol, period="1d", interval="1m")
        else:
            # For daily data, use the specified number of days
            data = yf.download(symbol, period=f"{num_days}d", interval="1d")
        
        if len(data) < 10:  # Minimum data check
            raise ValueError("Insufficient data points")
            
        # Use squeeze() to ensure we are working with a Series
        close_prices = data["Close"].squeeze().tolist()
        dates = data.index.tolist()
        
        if not close_prices:
            raise ValueError(f"No data available for symbol {symbol}")
            
        return close_prices, dates
    except Exception as e:
        print(f"Data fetch error for {symbol}: {str(e)}")
        return [], []


def determine_lot_size(capital: float, current_price: float, risk_percentage: float = 0.01) -> int:
    return max(1, int((capital * risk_percentage) / current_price))

def calculate_volatility(prices, window=20):
    if len(prices) < window:
        return 0.0
    returns = np.diff(prices[-window:]) / prices[-window:-1]
    return np.std(returns) * np.sqrt(252)

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    if np.std(returns) == 0:
        return 0.0
    return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

def getIndicators(prices, t, sma_period=20, ema_period=20, rsi_period=14,
                  macd_fast=12, macd_slow=26, macd_signal=9):
    prices_array = np.array(prices[:t+1])
    if len(prices_array) >= sma_period:
        sma_series = talib.SMA(prices_array, timeperiod=sma_period)
        sma = float(sma_series[-1])
    else:
        sma = float(np.mean(prices_array))
    if len(prices_array) >= ema_period:
        ema_series = talib.EMA(prices_array, timeperiod=ema_period)
        ema = float(ema_series[-1])
    else:
        ema = float(np.mean(prices_array))
    if len(prices_array) >= rsi_period:
        rsi_series = talib.RSI(prices_array, timeperiod=rsi_period)
        rsi = float(rsi_series[-1])
        if np.isnan(rsi):
            rsi = 50.0
    else:
        rsi = 50.0
    if len(prices_array) >= macd_slow:
        macd, _, _ = talib.MACD(prices_array,
                                fastperiod=macd_fast,
                                slowperiod=macd_slow,
                                signalperiod=macd_signal)
        macd_val = float(macd[-1])
        if np.isnan(macd_val):
            macd_val = 0.0
    else:
        macd_val = 0.0
    return sma, ema, rsi, macd_val

def getStateEnhanced(prices, t, window_size, current_sentiment=0.0):
    """
    Computes state from price data – matches original CODE.py implementation.
    The state vector is composed of:
    - (window_size - 1) price differences, and
    - 7 indicators: SMA, EMA, RSI, MACD, volatility, trend, and current_sentiment.
    """
    if t - window_size + 1 < 0:
        pad = [prices[0]] * (window_size - t - 1)
        block = pad + prices[0:t+1]
    else:
        block = prices[t - window_size + 1:t + 1]
        
    # Compute price differences only (do not include returns)
    price_diffs = [block[i+1] - block[i] for i in range(len(block)-1)]
    
    # Compute technical indicators
    sma, ema, rsi, macd_val = getIndicators(prices, t)
    volatility = calculate_volatility(prices[:t+1])
    trend = (block[-1] - block[0]) / block[0] if block[0] != 0 else 0
    
    # Build state array: price_diffs (9 elements if window_size is 10) plus 7 indicators = 16 features
    state = np.array(price_diffs + [sma, ema, rsi, macd_val, volatility, trend, current_sentiment])
    
    # Pack indicators (optional) as: (sma, ema, rsi, macd_val, volatility, price_diffs, trend, current_sentiment)
    indicators = (sma, ema, rsi, macd_val, volatility, price_diffs, trend, current_sentiment)
    return state, indicators


def action_to_str(action: int) -> str:
    mapping = {0: "Hold", 1: "Buy", 2: "Sell"}
    return mapping.get(action, "Unknown")

# --- Sentiment Analysis Functions ---

sentiment_cache = {}

def setup_nlp_environment():
    """Set up NLP environment with CPU-only isolation for transformers."""
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        print("Loading NLP models on CPU only...")
        tokenizer = PegasusTokenizer.from_pretrained("human-centered-summarization/financial-summarization-pegasus")
        model = TFPegasusForConditionalGeneration.from_pretrained("human-centered-summarization/financial-summarization-pegasus")
        sentiment_pipeline = pipeline("sentiment-analysis", framework="tf")
        success = True
    except Exception as e:
        print(f"Error loading NLP models: {e}")
        tokenizer, model, sentiment_pipeline = None, None, None
        success = False
    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
    return tokenizer, model, sentiment_pipeline, success

tokenizer, model, sentiment_pipeline, nlp_success = setup_nlp_environment()

def get_yahoo_news_links(query, max_articles=5):
    """Get Yahoo Finance news links with error handling."""
    search_url = f'https://news.search.yahoo.com/search?p={query}'
    try:
        r = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        articles = soup.find_all('h4', class_='s-title')[:max_articles]
        urls = [a.find('a')['href'] for a in articles if a.find('a')]
        return urls
    except Exception as e:
        print(f"Error fetching news links: {e}")
        return []

def scrape_articles(urls):
    """Scrape articles with error handling."""
    articles = []
    for url in urls:
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            words = text.split(' ')[:350]  # Limit to 350 words
            articles.append(' '.join(words))
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    return articles

def summarize_articles(articles):
    """Summarize articles using NLP models; fallback to extraction if needed."""
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    summaries = []
    try:
        if not nlp_success or tokenizer is None or model is None:
            summaries = [' '.join(article.split()[:50]) for article in articles]
        else:
            for article in articles:
                try:
                    input_ids = tokenizer.encode(article, return_tensors="tf")
                    output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
                    summary = tokenizer.decode(output[0], skip_special_tokens=True)
                    summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing article: {e}")
                    summaries.append(' '.join(article.split()[:50]))
    except Exception as e:
        print(f"Error in summarization: {e}")
        summaries = [' '.join(article.split()[:50]) for article in articles]
    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
    return summaries

def simple_sentiment_analysis(text):
    """Simple rule-based sentiment analysis."""
    positive_words = ['increase', 'growth', 'profit', 'gain', 'positive', 'up', 'higher', 'bull', 'bullish']
    negative_words = ['decrease', 'loss', 'down', 'negative', 'lower', 'bear', 'bearish', 'crash']
    text = text.lower()
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    total = positive_count + negative_count
    if total == 0:
        return {"score": 0.5}
    score = positive_count / total
    return {"score": score}

def analyze_sentiment(summaries):
    """Analyze sentiment using NLP pipeline; fallback to simple method if needed."""
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        if not nlp_success or sentiment_pipeline is None:
            results = [simple_sentiment_analysis(summary) for summary in summaries]
        else:
            results = sentiment_pipeline(summaries)
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        results = [simple_sentiment_analysis(summary) for summary in summaries]
    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
    return results

def get_sentiment_score(query: str) -> float:
    """
    Complete sentiment analysis implementation.
    This function uses NLP models to summarize news articles and analyze sentiment.
    """
    if query in sentiment_cache:
        print(f"Using cached sentiment for '{query}'")
        return sentiment_cache[query]

    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    try:
        urls = get_yahoo_news_links(query)
        if not urls:
            print(f"No news links found for '{query}', returning neutral sentiment")
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
            return 0.5  # Neutral sentiment

        articles = scrape_articles(urls)
        if not articles:
            print(f"No articles scraped for '{query}', returning neutral sentiment")
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
            return 0.5

        summaries = summarize_articles(articles)
        sentiments = analyze_sentiment(summaries)

        if sentiments:
            avg_score = float(np.mean([s["score"] for s in sentiments if "score" in s]))
            sentiment_cache[query] = avg_score
            print(f"Sentiment score for '{query}': {avg_score:.4f}")
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
            return avg_score

        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
        return 0.5
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
        return 0.5  # Return neutral sentiment on error

# Django-specific utility functions for session management
def save_session_data(session_id, data):
    """Save session data to a JSON file"""
    session_dir = os.path.join(settings.MEDIA_ROOT, 'trading_sessions')
    os.makedirs(session_dir, exist_ok=True)
    
    file_path = os.path.join(session_dir, f"{session_id}.json")
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_session_data(session_id):
    """Load session data from a JSON file"""
    file_path = os.path.join(settings.MEDIA_ROOT, 'trading_sessions', f"{session_id}.json")
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None