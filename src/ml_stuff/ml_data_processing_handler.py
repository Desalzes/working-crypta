# src/ml_stuff/data_handler.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.indicators.indicators import Indicators
from src.indicators.ind_funcs import calculate_vwap

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            # Update path to correct location
            data_dir = os.path.join(Path(__file__).parent.parent, "data")
        self.data_dir = data_dir
        self.historical_dir = os.path.join(data_dir, "historical")
        self.indicators = Indicators()
        print(f"[DataHandler] Using data directory: {self.data_dir}")

    def get_data_files(self, pair: str) -> Dict[str, str]:
        # Extract base symbol (e.g., BTC from BTC/USD or BTCUSD)
        base_symbol = pair.upper().replace('USD', '').replace('/', '')
        print(f"[DataHandler] Looking for {base_symbol} data")
        
        pair_dir = os.path.join(self.historical_dir, base_symbol)
        if not os.path.exists(pair_dir):
            print(f"[DataHandler] Directory not found: {pair_dir}")
            return {}

        files = {}
        timeframes = ["1m", "5m", "15m", "1d"]
        
        try:
            file_list = os.listdir(pair_dir)
            for tf in timeframes:
                # Try compiled file formats
                for format in [f"{base_symbol}_{tf}.csv", f"{base_symbol}-{tf}.csv"]:
                    if format in file_list:
                        files[tf] = os.path.join(pair_dir, format)
                        print(f"[DataHandler] Found compiled file for {tf}: {files[tf]}")
                        break

                # If no compiled file, try latest monthly file
                if tf not in files:
                    monthly_files = sorted([
                        f for f in file_list 
                        if f.startswith(f"{base_symbol}-{tf}-")
                    ])
                    if monthly_files:
                        latest_file = monthly_files[-1]
                        files[tf] = os.path.join(pair_dir, latest_file)
                        print(f"[DataHandler] Using latest monthly file for {tf}: {files[tf]}")

        except Exception as e:
            logger.error(f"[DataHandler] Error reading directory {pair_dir}: {e}")

        if not files:
            print(f"[DataHandler] No valid data files found for {base_symbol}")
        else:
            print(f"[DataHandler] Found {len(files)} timeframe files for {base_symbol}")

        return files

    def load_data(self, pair: str) -> Dict:
        files = self.get_data_files(pair)
        dataframes = {}

        if not files:
            logger.warning(f"No data files found for {pair}")
            return {}

        for tf, filepath in files.items():
            try:
                # Read CSV in chunks if file is large
                chunks = []
                for chunk in pd.read_csv(filepath, chunksize=100000):
                    chunks.append(chunk)
                df = pd.concat(chunks)
                
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                
                # Basic columns check
                required_cols = ["open", "high", "low", "close", "volume"]
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Missing required columns in {filepath}")
                    continue
                
                # Add technical indicators
                df = self.add_technical_features(df)
                dataframes[tf] = df
                
                print(f"[DataHandler] Loaded {tf} data with {len(df)} rows")
                print(f"[DataHandler] Columns: {list(df.columns)}")
                
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                continue

        return dataframes

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        try:
            df = df.copy()
            
            # Calculate all indicators
            all_indicators = self.indicators.calculate_all(df)
            
            # Add VWAP
            df['vwap'] = calculate_vwap(df)
            
            # Extract numeric values from indicator results
            for indicator_name, indicator_data in all_indicators.items():
                if isinstance(indicator_data, dict):
                    for key, value in indicator_data.items():
                        if isinstance(value, (int, float, np.number)):
                            col_name = f"{indicator_name}_{key}"
                            df[col_name] = value
                elif isinstance(indicator_data, (int, float, np.number)):
                    df[indicator_name] = indicator_data
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical features: {e}")
            return df

    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume', 'vwap',
            'rsi_value', 'macd_macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_width', 'stoch_k', 'stoch_d',
            'atr_value', 'trend_ema20', 'trend_ema50', 'trend_sma200',
            'volume_current_ratio', 'abs_strength_value'
        ]
        
        # Ensure all required columns exist
        available_columns = [col for col in feature_columns if col in df.columns]
        if len(available_columns) < 5:  # Need at least OHLCV
            logger.error(f"Not enough features available. Found: {available_columns}")
            return np.array([]), np.array([])

        features = df[available_columns].values
        sequences = []
        labels = []

        for i in range(len(df) - sequence_length):
            sequence = features[i:(i + sequence_length)]
            # Generate labels based on future returns
            future_return = (df['close'].iloc[i + sequence_length] / df['close'].iloc[i + sequence_length - 1]) - 1
            # Use three classes: 0 (down), 1 (neutral), 2 (up)
            if future_return > 0.001:  # Up
                label = 2
            elif future_return < -0.001:  # Down
                label = 0
            else:  # Neutral
                label = 1
            
            sequences.append(sequence)
            labels.append(label)

        return np.array(sequences), np.array(labels)

    def split_data(self, sequences: np.ndarray, labels: np.ndarray,
                   train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple:
        n = len(sequences)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_seq = sequences[:train_end]
        val_seq = sequences[train_end:val_end]
        test_seq = sequences[val_end:]

        train_labels = labels[:train_end]
        val_labels = labels[train_end:val_end]
        test_labels = labels[val_end:]

        return (train_seq, train_labels), (val_seq, val_labels), (test_seq, test_labels)