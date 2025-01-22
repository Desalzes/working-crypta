import asyncio
import logging
import json
import os
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import pandas as pd

from src.kraken_api import KrakenAPI
from src.portfolio_manager import PortfolioManager
from src.ml_trader import MLTrader
from src.market_analyzer import MarketAnalyzer
from src.indicators import Indicators


class LiveTradingEngine:
    def __init__(self, config_path=None):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='live_trading.log',
            filemode='a'
        )
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config_path is None:
            config_path = os.path.join(Path(__file__).parent, 'config.json')

        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise

        # Initialize components
        self.kraken = KrakenAPI(
            self.config['kraken_api_key'],
            self.config['kraken_secret_key']
        )
        self.portfolio = PortfolioManager(live_trading=True)
        self.ml_trader = MLTrader(load_live_model=True)
        self.market_analyzer = MarketAnalyzer()
        self.indicators = Indicators()

        # Trading parameters
        self.trading_pairs = self.config.get('trading_pairs', [])
        self.trade_interval = 60  # 1 minute default
        self.max_position_size = 0.1  # 10% of balance

        # Risk management
        self.stop_loss_percentage = 0.05  # 5% stop loss
        self.take_profit_percentage = 0.10  # 10% take profit

        # Filtering out known problematic pairs
        self.trading_pairs = [
            pair for pair in self.trading_pairs
            if pair not in ['DOGEUSDT', 'SUIUSDT']
        ]
        self.logger.info(f"Active trading pairs: {self.trading_pairs}")

    async def get_market_signals(self, pair: str) -> dict:
        """Generate trading signals for a specific pair"""
        try:
            # Fetch market data
            df = await self.kraken.get_market_data(pair)
            if df is None or len(df) < 30:
                self.logger.warning(f"Insufficient market data for {pair}")
                return {}

            # Calculate indicators
            try:
                indicators = self.indicators.calculate_all(df)
            except Exception as ind_error:
                self.logger.error(f"Indicator calculation error for {pair}: {ind_error}")
                indicators = {}

            # ML prediction
            try:
                ml_prediction = self.ml_trader.predict({
                    '1m': df,
                    '5m': df.resample('5min').agg({
                        'open': 'first', 'high': 'max',
                        'low': 'min', 'close': 'last',
                        'volume': 'sum'
                    })
                })
            except Exception as ml_error:
                self.logger.error(f"ML prediction error for {pair}: {ml_error}")
                ml_prediction = {}

            # Market regime analysis
            try:
                regime_data = self.market_analyzer.analyze_market_regime(df)
            except Exception as regime_error:
                self.logger.error(f"Market regime analysis error for {pair}: {regime_error}")
                regime_data = {}

            return {
                'indicators': indicators,
                'ml_prediction': ml_prediction,
                'regime': regime_data,
                'current_price': float(df['close'].iloc[-1])
            }

        except Exception as e:
            self.logger.error(f"Comprehensive error getting market signals for {pair}: {e}")
            return {}

    def determine_trade_action(self, signals: dict) -> str:
        """Determine trade action based on multiple signals"""
        try:
            ml_pred = signals.get('ml_prediction', {})
            indicators = signals.get('indicators', {})
            regime = signals.get('regime', {})

            # Complex decision logic
            buy_score = 0
            sell_score = 0

            # ML Prediction score
            if ml_pred.get('action') == 'BUY':
                buy_score += ml_pred.get('confidence', 0) * 2
            elif ml_pred.get('action') == 'SELL':
                sell_score += ml_pred.get('confidence', 0) * 2

            # Technical Indicator Scores
            rsi = indicators.get('rsi', 50)
            if rsi < 30:  # Oversold
                buy_score += 1
            elif rsi > 70:  # Overbought
                sell_score += 1

            # Market Regime Score
            if regime.get('trend') == 'bullish':
                buy_score += 1
            elif regime.get('trend') == 'bearish':
                sell_score += 1

            # Final Decision
            if buy_score > sell_score and buy_score > 1:
                return 'BUY'
            elif sell_score > buy_score and sell_score > 1:
                return 'SELL'

            return 'HOLD'

        except Exception as e:
            self.logger.error(f"Error in trade decision: {e}")
            return 'HOLD'

    async def execute_trade_decision(self, pair: str, signals: dict):
        """Make and execute trading decision based on signals"""
        try:
            # Validate signals
            if not signals:
                return

            current_price = signals.get('current_price')
            if not current_price:
                return

            # Determine trade action
            action = self.determine_trade_action(signals)

            # Get current position and balance
            current_position = self.portfolio.get_position(pair)
            available_balance = self.portfolio.balance

            # Trade execution logic
            if action == 'BUY' and available_balance > 0:
                # Calculate position size (10% of available balance)
                position_size = min(available_balance * 0.1, available_balance)

                # Execute buy
                trade_result = self.portfolio.execute_trade(
                    pair,
                    'BUY',
                    current_price,
                    position_size,
                    {
                        'confidence': signals.get('ml_prediction', {}).get('confidence', 0.5),
                        'stop_loss': current_price * (1 - self.stop_loss_percentage),
                        'take_profit': current_price * (1 + self.take_profit_percentage)
                    }
                )

                if trade_result:
                    self.logger.info(f"BUY {pair} @ {current_price}")

            elif action == 'SELL' and current_position['amount'] > 0:
                # Sell entire position
                trade_result = self.portfolio.execute_trade(
                    pair,
                    'SELL',
                    current_price,
                    current_position['amount'] * current_price,
                    {
                        'confidence': signals.get('ml_prediction', {}).get('confidence', 0.5)
                    }
                )

                if trade_result:
                    self.logger.info(f"SELL {pair} @ {current_price}")

        except Exception as e:
            self.logger.error(f"Trade execution error for {pair}: {e}")

    async def trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting Live Trading Engine")

        while True:
            try:
                # Process each trading pair
                tasks = [self.process_pair(pair) for pair in self.trading_pairs]
                await asyncio.gather(*tasks)

                # Wait for next interval
                await asyncio.sleep(self.trade_interval)

            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def process_pair(self, pair: str):
        """Process a single trading pair"""
        try:
            # Get market signals
            signals = await self.get_market_signals(pair)

            # Execute trade decision
            await self.execute_trade_decision(pair, signals)

        except Exception as e:
            self.logger.error(f"Error processing {pair}: {e}")

    async def start(self):
        """Start the live trading engine"""
        try:
            await self.trading_loop()
        except Exception as e:
            self.logger.critical(f"Trading engine crashed: {e}")


def main():
    trading_engine = LiveTradingEngine()
    asyncio.run(trading_engine.start())


if __name__ == "__main__":