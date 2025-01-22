import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models import MultiTimeframeTCN  # Import the model
import ta  # For technical indicators

logger = logging.getLogger(__name__)


class MLTrader:
    def __init__(self, sequence_length: int = 20, hidden_size: int = 64,
                 num_levels: int = 3, train_ratio: float = 0.8, val_ratio: float = 0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.scalers: Dict[str, StandardScaler] = {}

        self.n_features = 10  # OHLCV + RSI, MACD, MACD Signal, VWAP, ATR

        self.model = MultiTimeframeTCN(
            input_size=self.n_features,
            hidden_size=hidden_size,
            num_levels=num_levels,
            kernel_size=2,
            dropout=0.2,
            num_classes=3
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.data_dir = os.path.join(Path(__file__).parent.parent, 'data', 'historical')
        self.model_dir = os.path.join(Path(__file__).parent.parent, 'data', 'models')
        os.makedirs(self.model_dir, exist_ok=True)

    # Add other methods from your MLTrader class here
