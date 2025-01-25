# src/ml_stuff/predictor.py

import torch
import logging
from typing import Dict, Optional
import pandas as pd
from .model_base import ModelBase
from .ml_data_processing_handler import DataHandler

logger = logging.getLogger(__name__)


class ModelPredictor(ModelBase):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.data_handler = DataHandler()

    def predict(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        try:
            self.model.eval()
            predictions = {}

            for timeframe, df in timeframe_data.items():
                if df.empty:
                    logger.warning(f"Empty dataframe for {timeframe}")
                    continue

                features = self.data_handler.prepare_sequences(df, self.sequence_length)
                if len(features[0]) == 0:
                    continue

                # Get last sequence
                last_sequence = torch.FloatTensor(features[0][-1:]).to(self.device)

                with torch.no_grad():
                    output = self.model(last_sequence)
                    probabilities = torch.softmax(output, dim=1)
                    predictions[timeframe] = probabilities[0].cpu().numpy().tolist()

            return predictions

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {}

    def backtest(self, data: Dict[str, pd.DataFrame], initial_balance: float = 10000,
                 position_size: float = 0.1) -> Dict:
        """Backtest the model on historical data"""
        try:
            balance = initial_balance
            position = 0
            trades = []

            # Use shortest timeframe data for backtesting
            df = data.get("1m")
            if df is None:
                logger.error("No 1m timeframe data available for backtesting")
                return {
                    "final_balance": balance,
                    "return": 0,
                    "n_trades": 0,
                    "trades": []
                }

            sequences, _ = self.data_handler.prepare_sequences(df, self.sequence_length)

            # Convert sequences to tensor once for better performance
            sequences_tensor = torch.FloatTensor(sequences).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                # Process in batches for better performance
                batch_size = 128
                for i in range(0, len(sequences), batch_size):
                    batch_end = min(i + batch_size, len(sequences))
                    batch = sequences_tensor[i:batch_end]
                    outputs = self.model(batch)
                    probabilities = torch.softmax(outputs, dim=1)

                    for j, probs in enumerate(probabilities):
                        current_idx = i + j
                        current_price = df['close'].iloc[current_idx + self.sequence_length]
                        
                        # Trading logic
                        if position == 0 and probs[1] > 0.7:  # Buy signal
                            position_size_usd = balance * position_size
                            position = position_size_usd / current_price
                            balance -= position_size_usd
                            trades.append({
                                'type': 'buy',
                                'price': current_price,
                                'size': position_size_usd,
                                'balance': balance
                            })

                        elif position > 0 and probs[0] > 0.7:  # Sell signal
                            sale_value = position * current_price
                            balance += sale_value
                            trades.append({
                                'type': 'sell',
                                'price': current_price,
                                'size': sale_value,
                                'balance': balance
                            })
                            position = 0

            # Close any remaining position
            if position > 0:
                final_price = df['close'].iloc[-1]
                balance += position * final_price

            return {
                "final_balance": balance,
                "return": ((balance - initial_balance) / initial_balance) * 100,
                "n_trades": len(trades),
                "trades": trades
            }

        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {
                "final_balance": initial_balance,
                "return": 0,
                "n_trades": 0,
                "trades": []
            }