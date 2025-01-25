import asyncio
import logging
import os
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta
import asyncio

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = logging.getLogger(__name__)

class MarketDataManager:
    def __init__(self, exchange, data_processor, data_dir):
        self.exchange = exchange
        self.data_processor = data_processor
        self.data_dir = data_dir

    async def get_market_data(self, pair: str, timeframe: str = '1m') -> Optional[pd.DataFrame]:
        try:
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv,
                pair,
                timeframe=timeframe,
                limit=1000
            )

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return self.data_processor.add_technical_features(df)

        except Exception as e:
            logger.error(f"Error fetching market data for {pair}: {e}")
            return None

    async def fetch_all_timeframes(self, pair: str) -> Dict[str, pd.DataFrame]:
        timeframes = ["1m", "5m", "15m"]
        data = {}

        tasks = [self.get_market_data(pair, tf) for tf in timeframes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tf, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.warning(f"Error fetching {tf} data for {pair}: {result}")
                continue
            if result is not None:
                data[tf] = result

        return data

    async def get_ticker(self, pair: str) -> Optional[Dict]:
        try:
            return await asyncio.to_thread(self.exchange.fetch_ticker, pair)
        except Exception as e:
            logger.error(f"Error fetching ticker for {pair}: {e}")
            return None

    async def get_account_balance(self) -> Optional[Dict]:
        try:
            balance = await asyncio.to_thread(self.exchange.fetch_balance)
            logger.info(f"Fetched account balance: {balance['total']}")
            return balance['total']
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return None

    def prepare_model_input(self, timeframe_data: Dict[str, pd.DataFrame], scaler) -> Optional[pd.DataFrame]:
        try:
            return self.data_processor.prepare_model_input(timeframe_data, scaler)
        except Exception as e:
            logger.error(f"Error preparing model input: {e}")
            return None