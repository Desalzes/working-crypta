import json

import pandas as pd
import os
import requests
from datetime import datetime, timedelta
import zipfile
from io import BytesIO
from pathlib import Path

class BinanceDataDownloader:
    def __init__(self):
        self.base_url = "https://data.binance.vision/data/spot/monthly/klines"
        self.timeframes = {
            '1m': '1m', 
            '5m': '5m', 
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        self.data_dir = os.path.join(Path(__file__).parent.parent, 'data', 'historical')
        os.makedirs(self.data_dir, exist_ok=True)

    def compile_historical_data(self, months_back=6) -> dict:
        """Download historical market data"""
        compiled_data = {}
        total_pairs = len(self._get_pairs())
        
        for idx, pair in enumerate(self._get_pairs(), 1):
            print(f"\nDownloading {pair} ({idx}/{total_pairs}) - all timeframes (1m, 5m, 15m, 1h, 4h, 1d)")
            try:
                pair_data = self._download_pair_data(pair, months_back)
                if pair_data:
                    compiled_data[pair] = pair_data
            except Exception as e:
                print(f"Error downloading {pair}: {str(e)}")
                continue

        return compiled_data

    def save_data(self, data_dict) -> list:
        """Save downloaded data to CSV files"""
        filenames = []
        print("Data dictionary keys:", list(data_dict.keys()))

        for pair, timeframe_data in data_dict.items():
            pair_dir = os.path.join(self.data_dir, pair)
            os.makedirs(pair_dir, exist_ok=True)
            print(f"Creating directory: {pair_dir}")

            for timeframe, data in timeframe_data.items():
                filename = os.path.join(pair_dir, f"{pair}_{timeframe}_data.csv")
                print(f"Saving data to: {filename}")
                print(f"Data shape: {data.shape}")
                data.to_csv(filename, index=False)
                filenames.append(filename)

        return filenames

    def _get_pairs(self) -> list:
        """Get list of trading pairs and ensure proper format for Binance API"""
        config_path = os.path.join(Path(__file__).parent, 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                pairs = config.get('trading_pairs', [])

                # Filter out non-USDT pairs and ensure proper format
                valid_pairs = [pair for pair in pairs if 'USDT' in pair]
                if not valid_pairs:
                    # Fallback to default pairs if no valid pairs found
                    valid_pairs = [
                        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
                        'XRPUSDT', 'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT'
                    ]
                print(f"Downloading data for pairs: {valid_pairs}")
                return valid_pairs
        except Exception as e:
            print(f"Error loading config.json: {e}")
            # Return default pairs if config loading fails
            return [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
                'XRPUSDT', 'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT'
            ]

    def _download_pair_data(self, pair: str, months_back: int) -> dict:
        """Download data for a specific pair across all timeframes"""
        end_date = datetime.now().replace(day=1) - timedelta(days=1)  # End of last month
        start_date = end_date - timedelta(days=30 * months_back)
        pair_data = {}

        for timeframe in self.timeframes.values():
            print(f"Processing {timeframe} data for {pair}")
            all_data = []
            current_date = start_date

            while current_date <= end_date:
                if current_date.year >= end_date.year and current_date.month > end_date.month:
                    break

                monthly_data = self._download_monthly_data(
                    timeframe, pair,
                    current_date.year,
                    current_date.month
                )

                if monthly_data is not None and not monthly_data.empty:
                    all_data.append(monthly_data)

                current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)

            if all_data:
                pair_data[timeframe] = pd.concat(all_data, ignore_index=True)
                print(f"Total records for {pair} {timeframe}: {len(pair_data[timeframe])}")

        return pair_data

    def _download_monthly_data(self, timeframe: str, pair: str, year: int, month: int) -> pd.DataFrame:
        """Download data for a specific month"""
        month_str = str(month).zfill(2)
        filename = f"{pair}-{timeframe}-{year}-{month_str}"
        cache_path = os.path.join(self.data_dir, 'cache', pair, f"{filename}.csv")

        if os.path.exists(cache_path):
            return pd.read_csv(cache_path)

        url = f"{self.base_url}/{pair}/{timeframe}/{filename}.zip"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                z = zipfile.ZipFile(BytesIO(response.content))
                csv_name = z.namelist()[0]
                df = pd.read_csv(
                    z.open(csv_name),
                    names=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                           'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                           'taker_buy_quote', 'ignore']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_csv(cache_path, index=False)
                return df
            else:
                print(f"Failed to download {url}")
                return None
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return None