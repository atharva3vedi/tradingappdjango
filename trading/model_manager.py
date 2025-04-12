import os
import joblib
import yfinance as yf
from keras.models import load_model
from .utils import fetch_stock_data, getStateEnhanced
import numpy as np
from sklearn.preprocessing import StandardScaler
from django.core.exceptions import ValidationError
import logging
from tqdm import tqdm
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.scalers_dir = os.path.join(os.path.dirname(__file__), 'scalers')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.scalers_dir, exist_ok=True)
        
        # Top 5 Indian stocks from different sectors
        self.categories = {
            'tech': ['TCS.NS', 'INFY.NS'],
            'finance': ['HDFCBANK.NS'],
            'healthcare': ['SUNPHARMA.NS'],
            'energy': ['RELIANCE.NS']
        }
        
        # Load existing models and scalers
        self.models = {}
        self.scalers = {}
        self._load_existing_models()
    
    def validate_stock_symbol(self, symbol):
        """Validate if a stock symbol exists and has data"""
        try:
            # Check if we can fetch data for this symbol
            data = yf.download(symbol, period='1d')
            if len(data) == 0:
                raise ValidationError(f"No data available for symbol {symbol}")
            return symbol.upper()
        except Exception as e:
            raise ValidationError(f"Invalid stock symbol {symbol}: {str(e)}")
    
    def get_category(self, symbol):
        """Determine which category a stock symbol belongs to"""
        symbol = symbol.upper()
        for category, symbols in self.categories.items():
            if symbol in symbols:
                return category
        return 'general'
    
    def get_model_confidence(self, symbol):
        """Calculate confidence score for a stock symbol"""
        category = self.get_category(symbol)
        if category == 'general':
            return 0.5  # Lower confidence for unknown stocks
        return 1.0  # High confidence for known stocks
    
    def get_trading_recommendation(self, symbol, price_data):
        """Get trading recommendation with confidence score"""
        model = self.get_model(symbol)
        category = self.get_category(symbol)
        confidence = self.get_model_confidence(symbol)
        
        # Get the recommendation
        state, _ = getStateEnhanced(price_data, len(price_data)-1, 10, 0.5)
        state = self.scalers[category].transform(state.reshape(1, -1))[0]
        
        q_values = model.predict(state[np.newaxis, :], verbose=0)[0]
        action = int(np.argmax(q_values))
        
        return {
            'action': action,
            'confidence': confidence,
            'q_values': q_values.tolist(),
            'category': category
        }
    
    def _load_existing_models(self):
        """Load existing models and scalers from disk"""
        for category in list(self.categories.keys()) + ['general']:
            model_path = os.path.join(self.models_dir, f'{category}_model.h5')
            scaler_path = os.path.join(self.scalers_dir, f'{category}_scaler.pkl')
            
            if os.path.exists(model_path):
                try:
                    # Try loading with standard loss functions
                    self.models[category] = load_model(
                        model_path,
                        custom_objects={
                            'MeanSquaredError': MeanSquaredError,
                            'MeanAbsoluteError': MeanAbsoluteError,
                            'multiply_by_2': multiply_by_2
                        }
                    )
                    logger.info(f"Loaded {category} model from {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load {category} model: {str(e)}")
                    # If loading fails, we'll create a new model
                    self.models[category] = None
            
            if os.path.exists(scaler_path):
                try:
                    self.scalers[category] = joblib.load(scaler_path)
                    logger.info(f"Loaded {category} scaler from {scaler_path}")
                except Exception as e:
                    logger.warning(f"Could not load {category} scaler: {str(e)}")
                    self.scalers[category] = None
    
    def update_models(self):
        """Update all models with latest data"""
        logger.info("Starting model update process")
        
        # Update category-specific models
        for category in self.categories:
            logger.info(f"Updating {category} model")
            self._train_new_model(category)
        
        # Update general model with all data
        logger.info("Updating general model")
        self._train_new_model('general')
        
        logger.info("Model update process completed")
    
    def _train_new_model(self, category):
        """Train a new model for a category"""
        from .trading_session import Agent
        from tqdm import tqdm
        
        logger.info(f"Starting training for {category} model")
        
        # Get training data
        training_data = []
        if category == 'general':
            # For general model, use data from all categories
            for cat_symbols in self.categories.values():
                for symbol in tqdm(cat_symbols, desc=f"Fetching data for {category}"):
                    prices, _ = fetch_stock_data(symbol, 'daily', num_days=365)
                    if prices:
                        training_data.extend(prices)
        else:
            # For category-specific models, use only that category's data
            for symbol in tqdm(self.categories[category], desc=f"Fetching data for {category}"):
                prices, _ = fetch_stock_data(symbol, 'daily', num_days=365)
                if prices:
                    training_data.extend(prices)
        
        if not training_data:
            logger.warning(f"No training data available for category {category}")
            return
        
        logger.info(f"Training data collected: {len(training_data)} points")
        
        # Create and train the model
        window_size = 10
        state_size = (window_size - 1) + 7
        agent = Agent(state_size=state_size, model_type="improved")
        
        # Prepare training data in batches
        batch_size = min(agent.batch_size, len(training_data) - window_size)
        num_batches = (len(training_data) - window_size) // batch_size
        
        logger.info(f"Starting model training with {num_batches} batches of size {batch_size}")
        
        # Train the model with progress bar
        for batch_idx in tqdm(range(num_batches), desc=f"Training {category} model"):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            for i in range(start_idx, end_idx):
                state, _ = getStateEnhanced(training_data, i, window_size, 0.5)
                next_state, _ = getStateEnhanced(training_data, i+1, window_size, 0.5)
                
                # Calculate reward based on price movement
                reward = (training_data[i+1] - training_data[i]) / training_data[i]
                
                agent.memory.append((state, 0, reward, next_state, False))
                
                if len(agent.memory) > agent.batch_size:
                    agent.expReplay(agent.batch_size)
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                logger.info(f"Training progress: {batch_idx}/{num_batches} batches completed")
        
        # Save the model with custom objects
        model_path = os.path.join(self.models_dir, f'{category}_model.h5')
        try:
            agent.model.save(
                model_path,
                save_format='h5',
                include_optimizer=True
            )
            self.models[category] = agent.model
            logger.info(f"Saved {category} model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving {category} model: {str(e)}")
            raise
        
        # Create and save scaler
        logger.info("Creating and saving scaler")
        scaler = StandardScaler()
        states_list = []
        
        # Process states in batches for memory efficiency
        batch_size = 1000
        for i in tqdm(range(0, len(training_data) - window_size, batch_size), 
                     desc="Creating scaler"):
            batch_end = min(i + batch_size, len(training_data) - window_size)
            batch_states = []
            for j in range(i, batch_end):
                state, _ = getStateEnhanced(training_data, j, window_size, 0.5)
                batch_states.append(state)
            states_list.extend(batch_states)
        
        if states_list:
            try:
                scaler.fit(np.array(states_list))
                scaler_path = os.path.join(self.scalers_dir, f'{category}_scaler.pkl')
                joblib.dump(scaler, scaler_path)
                self.scalers[category] = scaler
                logger.info(f"Saved {category} scaler to {scaler_path}")
            except Exception as e:
                logger.error(f"Error saving {category} scaler: {str(e)}")
                raise
        
        logger.info(f"Completed training for {category} model")
    
    def get_model(self, symbol):
        """Get the appropriate model for a given stock symbol"""
        category = self.get_category(symbol)
        
        if category not in self.models:
            logger.warning(f"No model found for category {category}, using general model")
            category = 'general'
            
        if category not in self.models:
            raise ValueError(f"No model available for symbol {symbol} (category: {category})")
            
        return self.models[category]
    
    def get_scaler(self, symbol):
        """Get the appropriate scaler for a given stock symbol"""
        category = self.get_category(symbol)
        
        if category not in self.scalers:
            logger.warning(f"No scaler found for category {category}, using general scaler")
            category = 'general'
            
        if category not in self.scalers:
            raise ValueError(f"No scaler available for symbol {symbol} (category: {category})")
            
        return self.scalers[category]

# Singleton instance
model_manager = ModelManager()