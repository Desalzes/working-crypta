import os
import logging
from pathlib import Path
import pandas as pd
from binance_data import BinanceDataDownloader
from indicator_combinations import IndicatorCombinations  # Added import for missing class
from config import TRADING_PAIRS

logger = logging.getLogger(__name__)


class ResearchManager:
    def __init__(self):
        self.binance_downloader = BinanceDataDownloader()
        self.data_dir = os.path.join(Path(__file__).parent, 'data', 'historical')
        self.trading_pairs = TRADING_PAIRS
        self.indicator_combinations = IndicatorCombinations()  # Initialize the missing attribute

    async def run_indicator_analysis(self):
        try:
            results = {}
            for pair in self.trading_pairs:
                print(f"\nAnalyzing {pair}...")
                data = {}
                for timeframe in ['1m', '5m', '15m']:
                    file_path = os.path.join(self.data_dir, f'{pair}_{timeframe}_data.csv')
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                        data[timeframe] = df

                if data:
                    combo_results = await self.indicator_combinations.test_combinations(data['1m'])
                    results[pair] = combo_results
                    print(f"Found {len(combo_results)} qualifying combinations for {pair}")
                else:
                    print(f"No data found for {pair}")

            return results

        except Exception as e:
            logger.error(f"Error in indicator analysis: {e}")
            raise

    async def download_market_data(self):
        try:
            print("Downloading historical data from Binance...")
            data = self.binance_downloader.compile_historical_data()

            if data:
                filenames = self.binance_downloader.save_data(data)
                print("\nData saved to:")
                for filename in filenames:
                    print(filename)
            else:
                print("No new data to download")

            return data
        except Exception as e:
            logger.error(f"Error downloading market data: {e}")
            raise