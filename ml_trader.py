import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import os
from .config import TRADING_PAIRS

logger = logging.getLogger(__name__)

class MLTrader:
    def __init__(self, sequence_length: int = 30, hidden_size: int = 128, num_layers: int = 2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.model = MultiTimeframeLSTM(
            input_size=7,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=3
        ).to(self.device)
        self.scalers = {}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.data_dir = os.path.join(Path(__file__).parent, 'data', 'historical')
        self.trading_pairs = TRADING_PAIRS
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        timeframes = {'1m': '_1m_data.csv', '5m': '_5m_data.csv', '15m': '_15m_data.csv'}
        
        dataframes = {}
        for timeframe, suffix in timeframes.items():
            for pair in self.trading_pairs:
                filepath = os.path.join(self.data_dir, f"{pair}{suffix}")
                try:
                    df = pd.read_csv(filepath)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    if pair not in dataframes:
                        dataframes[pair] = {}
                    dataframes[pair][timeframe] = df
                    logger.info(f"Loaded {filepath} with {len(df)} rows")
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")
                
        return dataframes

    # Rest of the class implementation remains the same...