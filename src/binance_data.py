import pandas as pd
import os
import requests
from datetime import datetime, timedelta
import zipfile
from io import BytesIO
from pathlib import Path

TRADING_PAIRS = [
    'BTCUSDT', 'XRPUSDT', 'ETHUSDT', 'SOLUSDT', 'USDCUSDT', 
    'DOGEUSDT', 'SUIUSDT', 'ADAUSDT', 'LTCUSDT', 'PEPEUSDT',
    'XLMUSDT', 'LINKUSDT', 'ALGOUSDT', 'SHIBUSDT', 'AVAXUSDT',
    'FTMUSDT', 'NEARUSDT'
]

class BinanceDataDownloader:
    def __init__(self):
        self.base_url = "https://data.binance.vision/data/spot/monthly/klines"
        self.timeframes = {'1m': '1m', '5m': '5m', '15m': '15m'}
        self.data_dir = os.path.join(Path(__file__).parent.parent, 'data', 'historical')
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Initialized downloader for {len(TRADING_PAIRS)} pairs")

    def get_data(self, months_back=6):
        compiled_data = {}
        
        for idx, pair in enumerate(TRADING_PAIRS, 1):
            print(f"\nDownloading {pair} ({idx}/{len(TRADING_PAIRS)})")
            try:
                pair_data = self._download_pair_data(pair, months_back)
                if pair_data:
                    compiled_data[pair] = pair_data
            except Exception as e:
                print(f"Error downloading {pair}: {str(e)}")
                continue
        return compiled_data
