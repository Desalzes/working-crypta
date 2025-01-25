# src/ml_stuff/model_base.py

import torch
import torch.nn as nn
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class ModelBase:
    def __init__(self, model: nn.Module, sequence_length: int = 12, hidden_size: int = 64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.sequence_length = sequence_length
        self.n_features = 21
        self.data_dir = os.path.join(Path(__file__).parent.parent.parent, "data")
        self.model_dir = os.path.join(self.data_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def save_model(self, pair: str, suffix: str = ""):
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{pair}_{timestamp}{'_' + suffix if suffix else ''}.pt"
            model_path = os.path.join(self.model_dir, model_filename)
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Saved model for {pair} at {model_path}")
        except Exception as e:
            logger.error(f"Error saving model for {pair}: {e}", exc_info=True)

    def load_model(self, model_path: str) -> bool:
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file does not exist: {model_path}")
                return False
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return False