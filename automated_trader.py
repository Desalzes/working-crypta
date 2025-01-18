import asyncio
import pandas as pd
from typing import Dict
import logging
import os
from pathlib import Path
import json
import aiohttp
from datetime import datetime, timedelta
from .ml_trader import MLTrader
from .market_analyzer import MarketAnalyzer
from .data_processor import DataProcessor
from .config import TRADING_PAIRS

logger = logging.getLogger(__name__)

class AutomatedTrader:
    def __init__(self):
        self.ml_trader = MLTrader()
        self.market_analyzer = MarketAnalyzer()
        self.data_processor = DataProcessor()
        self.ollama_url = "http://localhost:11434/api/generate"
        self.data_dir = os.path.join(Path(__file__).parent, 'data')
        self.position = 0
        self.balance = 10000
        self.trade_history = []
        self.trading_pairs = TRADING_PAIRS
        
    def _load_latest_data(self) -> Dict[str, pd.DataFrame]:
        """Load latest market data for all timeframes"""
        timeframes = {
            '1m': '_1m_data.csv',
            '5m': '_5m_data.csv',  
            '15m': '_15m_data.csv'
        }
        
        dataframes = {}
        for timeframe, suffix in timeframes.items():
            filepath = os.path.join(self.data_dir, 'historical', f'{self.current_pair}{suffix}')
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                dataframes[timeframe] = df
            
        return dataframes

    # Rest of the class implementation remains the same...