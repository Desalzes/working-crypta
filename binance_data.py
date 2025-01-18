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
        self.timeframes = {'1m': '1m', '5m': '5m', '15m': '15m'}
        self.data_dir = os.path.join(Path(__file__).parent, 'data', 'historical')
        os.makedirs(self.data_dir, exist_ok=True)
        self.trading_pairs = self._load_trading_pairs()

    def _load_trading_pairs(self):
        crypto_path = os.path.join(Path(__file__).parent, 'crypto.txt')
        try:
            with open(crypto_path, 'r') as f:
                pairs = f.read().strip().split(',')
                formatted_pairs = []
                for pair in pairs:
                    pair = pair.strip().upper()
                    if pair.startswith('XX'):
                        pair = pair[2:]
                    if not pair.endswith('USD') and not pair.endswith('USDT'):
                        pair = f"{pair}USDT"
                    if pair.endswith('USD'):
                        pair = f"{pair}T"
                    formatted_pairs.append(pair)
                print(f"Loaded {len(formatted_pairs)} trading pairs")
                return formatted_pairs
        except Exception as e:
            print(f"Error loading crypto.txt: {e}")
            return ['BTCUSDT']  # Fallback to BTC if file can't be loaded

    def compile_historical_data(self, months_back=6):
        compiled_data = {}
        total_pairs = len(self.trading_pairs)
        
        for idx, pair in enumerate(self.trading_pairs, 1):
            print(f"\nDownloading {pair} ({idx}/{total_pairs})")
            try:
                pair_data = self._download_pair_data(pair, months_back)
                if any(pair_data.values()):  # If any timeframe data was downloaded
                    compiled_data[pair] = pair_data
            except Exception as e:
                print(f"Error downloading {pair}: {str(e)}")
                continue
        return compiled_data

    def _download_pair_data(self, pair, months_back):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * months_back)
        pair_data = {}

        for timeframe in self.timeframes.values():
            print(f"Processing {timeframe} data for {pair}")
            all_data = []
            current_date = start_date

            while current_date <= end_date:
                monthly_data = self.download_monthly_data(
                    timeframe, 
                    pair,
                    current_date.year, 
                    current_date.month
                )
                
                if monthly_data is not None:
                    all_data.append(monthly_data)
                
                current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)

            if all_data:
                pair_data[timeframe] = pd.concat(all_data)
                print(f"Total records for {pair} {timeframe}: {len(pair_data[timeframe])}")

        return pair_data

    def download_monthly_data(self, timeframe, pair, year, month):
        month_str = str(month).zfill(2)
        filename = f"{pair}-{timeframe}-{year}-{month_str}"
        cache_path = os.path.join(self.data_dir, 'cache', pair, f"{filename}.csv")
        
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, index_col='timestamp', parse_dates=True)
            
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
                df.set_index('timestamp', inplace=True)
                
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_csv(cache_path)
                return df
            else:
                print(f"Failed to download {url}")
                return None
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return None

    def save_data(self, data_dict):
        filenames = []
        for pair, timeframe_data in data_dict.items():
            for timeframe, data in timeframe_data.items():
                filename = os.path.join(self.data_dir, f"{pair}_{timeframe}_data.csv")
                data.to_csv(filename)
                filenames.append(filename)
        return filenames