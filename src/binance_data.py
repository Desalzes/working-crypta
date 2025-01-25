import json
import pandas as pd
import os
import requests
from datetime import datetime, timedelta
import zipfile
from io import BytesIO
from pathlib import Path
import shutil

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
        # Main data directory for final compiled data
        self.data_dir = os.path.join(Path(__file__).parent.parent, 'src', 'data', 'historical')
        # Temporary cache directory for downloads
        self.cache_dir = os.path.join(Path(__file__).parent.parent, 'src', 'data', '.cache')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def compile_historical_data(self, months_back=6) -> dict:
        compiled_data = {}
        total_pairs = len(self._get_pairs())

        try:
            for idx, pair in enumerate(self._get_pairs(), 1):
                print(f"\nDownloading {pair} ({idx}/{total_pairs}) - all timeframes (1m, 5m, 15m, 1h, 4h, 1d)")
                try:
                    pair_data = self._download_pair_data(pair, months_back)
                    if pair_data:
                        compiled_data[pair] = pair_data
                except Exception as e:
                    print(f"Error downloading {pair}: {str(e)}")
                    continue

            self.save_data(compiled_data)
            return compiled_data
        finally:
            # Clean up cache directory after compilation
            self._cleanup_cache()

    def _cleanup_cache(self):
        """Remove all temporary files from cache directory"""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to clean cache directory: {e}")

    def save_data(self, data_dict) -> list:
        filenames = []

        for pair, timeframe_data in data_dict.items():
            base_name = pair.replace('USDT', '')
            pair_dir = os.path.join(self.data_dir, base_name)
            os.makedirs(pair_dir, exist_ok=True)

            for timeframe, data in timeframe_data.items():
                filename = os.path.join(pair_dir, f"{base_name}_{timeframe}.csv")
                if os.path.exists(filename):
                    existing_data = pd.read_csv(filename)
                    data = pd.concat([existing_data, data], ignore_index=True)
                print(f"Saving data to: {filename}")
                data.to_csv(filename, index=False)
                filenames.append(filename)

        return filenames

    def _get_pairs(self) -> list:
        """Get list of trading pairs from config and ensure proper format for Binance API"""
        config_path = os.path.join(Path(__file__).parent.parent, 'src/config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                pairs = config.get('trading_pairs', [])
                
            # Append USDT to each pair if not already present
            valid_pairs = [f"{pair}USDT" if not pair.endswith('USDT') else pair for pair in pairs]
            
            if not valid_pairs:
                raise ValueError("No trading pairs found in config.json")
            
            print(f"Downloading data for pairs: {valid_pairs}")
            return valid_pairs
            
        except Exception as e:
            print(f"Error loading config.json: {e}")
            raise  # Re-raise the exception instead of falling back to default pairs

    def _download_pair_data(self, pair: str, months_back: int) -> dict:
        end_date = datetime.now().replace(day=1) - timedelta(days=1)
        start_date = end_date - timedelta(days=30 * months_back)
        pair_data = {}
        base_name = pair.replace('USDT', '')

        for timeframe in self.timeframes.values():
            print(f"Processing {timeframe} data for {pair}")
            
            # Check if final file exists and has data for the date range
            final_file = os.path.join(self.data_dir, base_name, f"{base_name}_{timeframe}.csv")
            if os.path.exists(final_file):
                try:
                    existing_df = pd.read_csv(final_file)
                    if not existing_df.empty:
                        # Handle different timestamp formats
                        try:
                            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                        except:
                            try:
                                # Try parsing with different format if the first attempt fails
                                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], format='mixed')
                            except:
                                print(f"Error parsing timestamps in {final_file}, will redownload data")
                                existing_df = None
                        
                        if existing_df is not None and (existing_df['timestamp'].min() <= pd.Timestamp(start_date) and 
                            existing_df['timestamp'].max() >= pd.Timestamp(end_date)):
                            print(f"Data already exists for {pair} {timeframe}")
                            pair_data[timeframe] = existing_df
                            continue
                except Exception as e:
                    print(f"Error reading existing file {final_file}: {e}")
            
            all_data = []
            current_date = start_date

            while current_date <= end_date:
                if current_date.year >= end_date.year and current_date.month > end_date.month:
                    break

                monthly_data = self._download_monthly_data(timeframe, pair, current_date.year, current_date.month)
                if monthly_data is not None and not monthly_data.empty:
                    all_data.append(monthly_data)

                current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)

            if all_data:
                pair_data[timeframe] = pd.concat(all_data, ignore_index=True)
                # Ensure timestamp is in consistent ISO format
                pair_data[timeframe]['timestamp'] = pd.to_datetime(pair_data[timeframe]['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                print(f"Total records for {pair} {timeframe}: {len(pair_data[timeframe])}")

        return pair_data

    def _download_monthly_data(self, timeframe: str, pair: str, year: int, month: int) -> pd.DataFrame:
        base_name = pair.replace('USDT', '')
        month_str = str(month).zfill(2)
        filename = f"{base_name}-{timeframe}-{year}-{month_str}"
        
        # Use cache directory for temporary downloads
        cache_path = os.path.join(self.cache_dir, base_name, f"{filename}.csv")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.exists(cache_path):
            return pd.read_csv(cache_path)

        url = f"{self.base_url}/{pair}/{timeframe}/{pair}-{timeframe}-{year}-{month_str}.zip"

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
                
                # Calculate and add VWAP
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

                df.to_csv(cache_path, index=False)
                return df
            else:
                print(f"Failed to download {url}")
                # Try CryptoCompare as fallback
                return self._get_cryptocompare_data(timeframe, pair, year, month)
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            # Try CryptoCompare as fallback
            return self._get_cryptocompare_data(timeframe, pair, year, month)

    def _get_cryptocompare_data(self, timeframe: str, pair: str, year: int, month: int) -> pd.DataFrame:
        """Fallback to CryptoCompare when Binance data is not available"""
        print(f"Attempting to get data from CryptoCompare for {pair} {timeframe}")
        
        # Convert timeframe to CryptoCompare format
        timeframe_mapping = {
            '1h': {'endpoint': 'histohour', 'aggregate': 1},
            '4h': {'endpoint': 'histohour', 'aggregate': 4},
            '1d': {'endpoint': 'histoday', 'aggregate': 1}
        }
        
        if timeframe not in timeframe_mapping:
            print(f"Timeframe {timeframe} not supported by CryptoCompare fallback")
            return None
        
        params = timeframe_mapping[timeframe]
        
        # Calculate start and end timestamps for the month
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
        
        end_ts = int(end_date.timestamp())
        start_ts = int(start_date.timestamp())
        
        url = f"https://min-api.cryptocompare.com/data/v2/{params['endpoint']}"
        api_params = {
            'fsym': pair.replace('USDT', ''),
            'tsym': 'USDT',
            'limit': 2000,  # Max limit
            'toTs': end_ts,
            'aggregate': params['aggregate']
        }
        
        try:
            response = requests.get(url, params=api_params)
            if response.status_code == 200:
                data = response.json()
                if 'Data' in data and 'Data' in data['Data']:
                    df = pd.DataFrame(data['Data']['Data'])
                    
                    if not df.empty:
                        # Filter to only the requested month
                        df['datetime'] = pd.to_datetime(df['time'], unit='s')
                        df = df[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]
                        
                        if not df.empty:
                            # Format to match Binance structure with consistent timestamp format
                            df['timestamp'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                            df = df.rename(columns={
                                'volumefrom': 'volume',
                                'volumeto': 'quote_volume',
                                'time': 'close_time'
                            })
                            
                            # Add missing columns
                            df['trades'] = None
                            df['taker_buy_base'] = None
                            df['taker_buy_quote'] = None
                            
                            # Calculate VWAP
                            typical_price = (df['high'] + df['low'] + df['close']) / 3
                            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
                            
                            # Select and order columns
                            df = df[[
                                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                'taker_buy_quote', 'vwap'
                            ]]
                            
                            return df
        except Exception as e:
            print(f"Error getting CryptoCompare data: {str(e)}")
        
        return None
