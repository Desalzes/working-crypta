import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import os


class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_market_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load market data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return self._validate_and_clean_data(df)
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            return None

    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean market data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Forward fill missing values
        df = df.ffill()

        return df

    def remove_duplicates_and_save(self, data_dir: str) -> Dict[str, int]:
        """
        Remove duplicates from all CSV files in the directory and save cleaned versions.
        Returns a dictionary with the number of duplicates removed from each file.
        """
        results = {}
        try:
            for filename in os.listdir(data_dir):
                if not filename.endswith('.csv'):
                    continue

                filepath = os.path.join(data_dir, filename)
                try:
                    # Load the data
                    df = pd.read_csv(filepath)
                    initial_rows = len(df)

                    # Remove duplicates and clean
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = self._validate_and_clean_data(df)
                    final_rows = len(df)

                    # Save cleaned data back to file
                    df.to_csv(filepath, index=False)

                    # Store results
                    duplicates_removed = initial_rows - final_rows
                    if duplicates_removed > 0:
                        results[filename] = duplicates_removed
                        self.logger.info(f"Removed {duplicates_removed} duplicates from {filename}")

                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {e}")
                    continue

            return results

        except Exception as e:
            self.logger.error(f"Error accessing directory {data_dir}: {e}")
            return {}

    def calculate_returns(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate returns for different periods"""
        for period in periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
        return df

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features"""
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()

        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        return df

    def create_feature_groups(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create feature groups for analysis"""
        feature_groups = {
            'price': df[['timestamp', 'open', 'high', 'low', 'close']],
            'volume': df[['timestamp', 'volume', 'volume_ma', 'volume_ratio']],
            'technical': df[['timestamp', 'sma_20', 'sma_50', 'volatility']]
        }
        return feature_groups