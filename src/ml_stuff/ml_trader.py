# src/ml_stuff/ml_trader.py

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging
from .models import MultiTimeframeTCN
from .ml_model_trainer import ModelTrainer
from .ml_model_predictor import ModelPredictor
from .ml_data_processing_handler import DataHandler

logger = logging.getLogger(__name__)


class MLTrader:
    def __init__(self, sequence_length: int = 12, hidden_size: int = 64):
        self.model = MultiTimeframeTCN(
            input_size=21,
            hidden_size=hidden_size,
            num_levels=3,
            kernel_size=2,
            dropout=0.2,
            num_classes=3
        )

        self.trainer = ModelTrainer(self.model)
        self.predictor = ModelPredictor(self.model)
        self.data_handler = DataHandler()

        print(f"Using device: {self.trainer.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    async def train(self, pair: str, epochs: int = 30, batch_size: int = 64) -> Optional[Dict]:
        print(f"\nTraining model for {pair}...")
        # Train model using ModelTrainer
        result = await self.trainer.train(pair, epochs, batch_size)
        if result:
            print(f"Training completed for {pair}")
            return result
        else:
            print(f"No results for {pair}")
            return None

    def predict(self, timeframe_data: Dict[str, Dict]) -> Dict[str, float]:
        return self.predictor.predict(timeframe_data)

    def backtest(self, data: Dict[str, Dict], initial_balance: float = 10000,
                 position_size: float = 0.1) -> Dict:
        return self.predictor.backtest(data, initial_balance, position_size)

    def load_data(self, pair: str) -> Dict:
        return self.data_handler.load_data(pair)

    def load_model(self, model_path: str, scaler_path: str = None) -> bool:
        return self.trainer.load_model(model_path)