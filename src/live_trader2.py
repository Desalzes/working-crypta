import os
import json
import logging
import asyncio
import ccxt
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import torch
import pickle
from datetime import datetime

from .portfolio_manager import PortfolioManager
from .market_analyzer import MarketAnalyzer
from .data_processor import DataProcessor
from .indicators import Indicators
from .models import MultiTimeframeTCN

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    filename='live_trader.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class LiveTrader:
    def __init__(self):
        config_path = os.path.join(Path(__file__).parent, 'config.json')
        self.max_allocation_per_pair = 0.2  # Max 20% of total portfolio in one pair
        self.min_balance_threshold = 20.0  # Minimum balance to start trading


        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}", exc_info=True)
            raise FileNotFoundError(f"Could not load configuration from {config_path}")

        base_trading_pairs = config.get('trading_pairs', [])
        if not base_trading_pairs:
            logger.error("No trading pairs found in the configuration.")
            raise ValueError("Trading pairs list is empty in the config file.")

        self.trading_pairs = [f"{symbol}/USD" for symbol in base_trading_pairs]
        logger.info(f"Trading pairs set to: {', '.join(self.trading_pairs)}")

        self.api_key = os.getenv('CRYPTO_API_KEY') or config.get('crypto_api_key')
        self.api_secret = os.getenv('CRYPTO_SECRET_KEY') or config.get('crypto_secret_key')

        if not self.api_key or not self.api_secret:
            logger.error("Crypto.com API credentials not found")
            raise ValueError("Missing Crypto.com API credentials.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.exchange = ccxt.cryptocom({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
        })
        self.exchange.set_sandbox_mode(False)
        logger.info("Initialized CCXT Crypto.com exchange")

        try:
            self.exchange.load_markets()
            logger.info("Loaded markets successfully")
        except Exception as e:
            logger.error(f"Error loading markets: {e}", exc_info=True)
            raise ConnectionError("Failed to load markets from Crypto.com")

        self.data_dir = os.path.join(Path(__file__).parent.parent, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.market_analyzer = MarketAnalyzer()
        self.data_processor = DataProcessor()
        self.indicators = Indicators()
        self.portfolio = PortfolioManager()

        # Enhanced position tracking
        self.crypto_holdings = {}
        self.trade_history = []

        self.models = {}
        self.scalers = {}
        self.load_pretrained_models()

        logger.info("LiveTrader initialized with Crypto.com trading pairs.")
        print(f"Initialized LiveTrader with pairs: {', '.join(self.trading_pairs)}")

        # Define a balance cap (example: $10,000)
        self.balance_cap = config.get('balance_cap', 10000.0)
        logger.info(f"Balance cap set to: ${self.balance_cap:.2f}")

    def load_pretrained_models(self):
        model_dir = os.path.join(self.data_dir, "models")
        if not os.path.exists(model_dir):
            logger.error(f"Models directory does not exist: {model_dir}")
            return

        for file in os.listdir(model_dir):
            if file.endswith('_best.pt'):
                pair = file.replace('_best.pt', '')
                model_path = os.path.join(model_dir, file)
                scaler_path = os.path.join(model_dir, f"{pair}_scalers.pkl")

                if not os.path.exists(scaler_path):
                    logger.error(f"Scaler file does not exist for {pair}: {scaler_path}")
                    continue

                try:
                    model = MultiTimeframeTCN(
                        input_size=10,
                        hidden_size=64,
                        num_levels=3,
                        kernel_size=2,
                        dropout=0.2,
                        num_classes=3
                    ).to(self.device)

                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.eval()
                    self.models[pair] = model

                    with open(scaler_path, "rb") as f:
                        self.scalers[pair] = pickle.load(f)

                    logger.info(f"Loaded model and scalers for {pair}")

                except Exception as e:
                    logger.error(f"Error loading model/scaler for {pair}: {e}", exc_info=True)

    async def get_market_data(self, pair: str, timeframe: str = '1m') -> Optional[pd.DataFrame]:
        try:
            since = self.exchange.milliseconds() - (3600 * 1000 * 72)

            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv,
                pair,
                timeframe=timeframe,
                since=since
            )

            if not ohlcv or len(ohlcv) < 2:
                logger.warning(f"Insufficient data for {pair}")
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            df = self.data_processor.add_technical_features(df)

            logger.info(f"Retrieved {len(df)} candles for {pair}")
            return df

        except Exception as e:
            logger.error(f"Error getting market data for {pair}: {e}")
            return None

    def fetch_all_timeframes(self, pair: str) -> Dict[str, pd.DataFrame]:
        timeframes = ["1m", "5m", "15m"]
        historical_hours = {"1m": 1, "5m": 12, "15m": 24}
        data = {}

        for tf in timeframes:
            since = self.exchange.milliseconds() - (historical_hours[tf] * 3600 * 1000)

            try:
                ohlcv = self.exchange.fetch_ohlcv(pair, tf, since=since)
                if not ohlcv:
                    logger.warning(f"No data returned for {pair} on {tf} timeframe.")
                    continue

                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                data[tf] = df

                logger.info(f"Fetched {len(df)} records for {pair} on {tf} timeframe.")
            except Exception as e:
                logger.error(f"Error fetching {tf} data for {pair}: {e}")

        return data

    def make_trading_decision(self, pair: str, indicators: Dict, ml_prediction: torch.Tensor) -> Dict:
        """
        Make a trading decision based on ML predictions and technical indicators.

        Args:
            pair (str): Trading pair.
            indicators (Dict): Technical indicators.
            ml_prediction (torch.Tensor): ML model prediction.

        Returns:
            Dict: Decision dictionary containing action, size, stop_loss, and take_profit.
        """
        # Convert ML prediction tensor to numpy
        prediction = ml_prediction.argmax(dim=1).item()  # Assuming classes: 0 - SELL, 1 - HOLD, 2 - BUY

        action = "HOLD"
        size = 0.0
        stop_loss = None
        take_profit = None

        # Define thresholds or conditions based on indicators
        rsi = indicators.get('RSI', 50)  # Example: RSI indicator
        macd = indicators.get('MACD', 0)  # Example: MACD indicator

        if prediction == 2 and rsi < 70 and macd > 0:
            action = "BUY"
            size = 0.1  # Example: 10% of available balance
            stop_loss = indicators.get('Support', None)
            take_profit = indicators.get('Resistance', None)
        elif prediction == 0 and rsi > 30 and macd < 0:
            action = "SELL"
            size = 1.0  # Example: Sell entire position
            stop_loss = None
            take_profit = None
        else:
            action = "HOLD"
            size = 0.0
            stop_loss = None
            take_profit = None

        decision = {
            "decision": {
                "action": action,
                "size": size,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            },
            "reasoning": {
                "technical_analysis": f"RSI: {rsi}, MACD: {macd}",
                "ml_prediction": prediction,
                "market_context": "Based on ML model and technical indicators."
            }
        }

        logger.info(f"Made decision for {pair}: {decision}")
        return decision

    def _default_decision(self):
        return {
            "decision": {
                "action": "HOLD",
                "size": 0,
                "stop_loss": None,
                "take_profit": None
            },
            "reasoning": {
                "technical_analysis": "Analysis error",
                "risk_assessment": "",
                "market_context": ""
            }
        }

    async def execute_trade(self, pair: str, decision: Dict):
        try:
            action = decision['decision']['action']
            if action not in ["BUY", "SELL"]:
                return

            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, pair)
            current_price = ticker['last']

            if action == "BUY":
                available_balance = float(self.portfolio.balance)
                position_size = available_balance * float(decision['decision'].get('size', 0.1))
                amount = position_size / current_price

                if self.portfolio.balance < position_size:
                    logger.warning(f"Insufficient balance to BUY {pair}. Required: ${position_size}, Available: ${self.portfolio.balance}")
                    return

                if pair not in self.crypto_holdings:
                    self.crypto_holdings[pair] = {
                        'amount': 0,
                        'avg_price': 0,
                        'total_cost': 0
                    }

                current = self.crypto_holdings[pair]
                total_amount = current['amount'] + amount
                total_cost = current['total_cost'] + position_size
                avg_price = total_cost / total_amount if total_amount > 0 else 0

                # Place order with stop-loss and take-profit
                order_params = {}
                if decision['decision']['stop_loss'] is not None:
                    order_params['stopLoss'] = {
                        'price': decision['decision']['stop_loss']
                    }
                if decision['decision']['take_profit'] is not None:
                    order_params['takeProfit'] = {
                        'price': decision['decision']['take_profit']
                    }

                order = await asyncio.to_thread(
                    self.exchange.create_order,
                    symbol=pair,
                    type='limit',
                    side='buy',
                    amount=amount,
                    price=current_price,
                    params=order_params
                )

                self.crypto_holdings[pair].update({
                    'amount': total_amount,
                    'avg_price': avg_price,
                    'total_cost': total_cost,
                    'stop_loss': float(decision['decision']['stop_loss']) if decision['decision']['stop_loss'] else None,
                    'take_profit': float(decision['decision']['take_profit']) if decision['decision']['take_profit'] else None,
                    'last_update': datetime.now().isoformat()
                })

                # Deduct from portfolio balance
                self.portfolio.balance -= position_size

                logger.info(f"Executed BUY for {pair}: {order}")

            elif action == "SELL":
                position = self.crypto_holdings.get(pair, {})
                if position.get('amount', 0) > 0:
                    order = await asyncio.to_thread(
                        self.exchange.create_order,
                        symbol=pair,
                        type='limit',
                        side='sell',
                        amount=position['amount'],
                        price=current_price
                    )

                    trade_value = position['amount'] * current_price
                    profit_loss = trade_value - position['total_cost']

                    self.crypto_holdings[pair] = {
                        'amount': 0,
                        'avg_price': 0,
                        'total_cost': 0,
                        'stop_loss': None,
                        'take_profit': None,
                        'last_update': datetime.now().isoformat()
                    }

                    # Add to portfolio balance
                    self.portfolio.balance += trade_value

                    self.record_trade(pair, action, current_price, trade_value, profit_loss, decision)
                    logger.info(f"Executed SELL for {pair}: {order}")

        except Exception as e:
            logger.error(f"Error executing trade for {pair}: {e}")

    def record_trade(self, pair: str, action: str, price: float, size: float, profit_loss: float, decision: Dict):
        trade = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'type': action,
            'price': price,
            'size': size,
            'profit_loss': profit_loss,
            'reasoning': decision['reasoning']
        }
        self.trade_history.append(trade)
        self.save_trade_history()
        logger.info(f"Recorded trade: {trade}")

    def save_trade_history(self):
        history_path = os.path.join(self.data_dir, 'trade_history.json')
        try:
            with open(history_path, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
            logger.info("Saved trade history successfully.")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}", exc_info=True)

    async def process_trading_pair(self, pair: str):
        try:
            print(f"\nProcessing {pair}...")
            df = await self.get_market_data(pair)
            if df is None:
                print(f"No data available for {pair}, skipping...")
                return

            current_price = float(df['close'].iloc[-1])
            print(f"Current {pair} price: ${current_price:.4f}")
            holdings = self.crypto_holdings.get(pair, {})

            # Check stop-loss/take-profit
            if holdings.get('amount', 0) > 0:
                stop_loss = holdings.get('stop_loss')
                take_profit = holdings.get('take_profit')
                print(f"Current position: {holdings['amount']:.8f} {pair}")
                print(f"Stop Loss: ${stop_loss}, Take Profit: ${take_profit}")

                if stop_loss and current_price <= float(stop_loss):
                    print(f"Stop Loss triggered for {pair} at ${current_price}")
                    decision = {
                        'decision': {
                            'action': 'SELL',
                            'size': 1.0,
                            'stop_loss': None,
                            'take_profit': None
                        },
                        'reasoning': {
                            'technical_analysis': "Stop loss triggered.",
                            'risk_assessment': "Price fell below stop loss level.",
                            'market_context': "Automated risk management."
                        }
                    }
                    await self.execute_trade(pair, decision)
                    return
                elif take_profit and current_price >= float(take_profit):
                    print(f"Take Profit triggered for {pair} at ${current_price}")
                    decision = {
                        'decision': {
                            'action': 'SELL',
                            'size': 1.0,
                            'stop_loss': None,
                            'take_profit': None
                        },
                        'reasoning': {
                            'technical_analysis': "Take profit triggered.",
                            'risk_assessment': "Price reached take profit level.",
                            'market_context': "Automated profit-taking."
                        }
                    }
                    await self.execute_trade(pair, decision)
                    return

            # Regular trading analysis
            print("Calculating indicators...")
            indicators = self.indicators.calculate_all(df)
            regime_data = self.market_analyzer.analyze_regime(df)

            # Get ML predictions
            print("Getting ML predictions...")
            timeframe_data = await asyncio.to_thread(self.fetch_all_timeframes, pair)
            ml_predictions = None
            if pair in self.models and pair in self.scalers:
                try:
                    model_input = self.data_processor.prepare_model_input(
                        timeframe_data,
                        self.scalers[pair]
                    )
                    with torch.no_grad():
                        ml_predictions = self.models[pair](model_input.to(self.device))
                    print(f"ML Predictions: {ml_predictions}")
                except Exception as e:
                    print(f"Error getting ML predictions: {e}")
                    logger.error(f"Error getting ML predictions for {pair}: {e}", exc_info=True)

            if ml_predictions is not None:
                decision = self.make_trading_decision(pair, indicators, ml_predictions)
                print(f"Decision: {decision}")
                await self.execute_trade(pair, decision)
            else:
                logger.warning(f"No ML predictions available for {pair}, skipping trade.")
                print(f"No ML predictions available for {pair}, skipping trade.")

        except Exception as e:
            print(f"Error processing {pair}: {e}")
            logger.error(f"Error processing pair {pair}: {e}")

    def run_trading_loop(self, interval_minutes: int = 1):
        async def _trading_loop():
            print("Starting trading loop")
            while True:
                try:
                    print("\nUpdating portfolio status...")
                    await self.update_portfolio_value()

                    print("\nFetching account balance...")
                    balance = await asyncio.to_thread(self.get_account_balance)
                    if balance:
                        print(f"Account Balance: {balance}")
                        # Check if balance cap is reached
                        total_balance = balance.get('USD', 0.0)
                        if total_balance >= self.balance_cap:
                            logger.info(f"Balance cap of ${self.balance_cap:.2f} reached. Skipping BUY actions.")
                            print(f"Balance cap of ${self.balance_cap:.2f} reached. Skipping BUY actions.")
                    else:
                        print("Unable to fetch account balance.")

                    print("\nProcessing trading pairs...")
                    tasks = [self.process_trading_pair(pair) for pair in self.trading_pairs]
                    await asyncio.gather(*tasks)

                except Exception as e:
                    print(f"Error in trading loop: {e}")
                    logger.error(f"Error in trading loop: {e}")

                print(f"\nSleeping for {interval_minutes} minutes...")
                await asyncio.sleep(interval_minutes * 60)

        return _trading_loop()

    def get_account_balance(self) -> Optional[Dict]:
        """Retrieve account balance from Crypto.com using CCXT."""
        try:
            balance = self.exchange.fetch_balance()
            logger.info(f"Fetched account balance: {balance['total']}")
            return balance['total']
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}", exc_info=True)
            return None

    async def update_portfolio_value(self):
        """Update and display total portfolio value"""
        try:
            balance = await asyncio.to_thread(self.get_account_balance)
            if balance is None:
                logger.warning("Unable to update portfolio value due to balance fetch failure.")
                return

            total_value = balance.get('USD', 0.0)

            for pair, holdings in self.crypto_holdings.items():
                if holdings['amount'] > 0:
                    df = await self.get_market_data(pair)
                    if df is not None:
                        current_price = float(df['close'].iloc[-1])
                        crypto_value = holdings['amount'] * current_price
                        total_value += crypto_value

            logger.info(f"Current Portfolio Value: ${total_value:.2f}")
            print(f"\nPortfolio Summary:")
            print(f"Total Value: ${total_value:.2f}")
            print(f"USD Balance: ${balance.get('USD', 0.0):.2f}")

            for pair, holdings in self.crypto_holdings.items():
                if holdings['amount'] > 0:
                    print(f"{pair}: {holdings['amount']:.8f} @ ${holdings['avg_price']:.4f}")
                    if holdings.get('stop_loss') is not None:
                        print(f"  Stop Loss: ${holdings['stop_loss']}")
                    if holdings.get('take_profit') is not None:
                        print(f"  Take Profit: ${holdings['take_profit']}")

        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")

if __name__ == "__main__":
    trader = LiveTrader()
    asyncio.run(trader.run_trading_loop())
