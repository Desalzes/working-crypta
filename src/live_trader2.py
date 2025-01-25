import os
import json
import logging
import asyncio
import random

import ccxt
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import torch
import pickle
from datetime import datetime

from src.portfolio_manager import PortfolioManager
from src.market_analyzer import MarketAnalyzer
from src.data_processor import DataProcessor
from src.indicators.indicators import Indicators
from src.ml_stuff.models import MultiTimeframeTCN

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    filename='live_trader.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class LiveTrader2:
    def __init__(self):
        config_path = os.path.join(Path(__file__).parent, 'config.json')
        self.max_allocation_per_pair = 0.2
        self.min_balance_threshold = 90.0

        # --------------------------------------------------------------------
        # Load config
        # --------------------------------------------------------------------
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
        self.api_key = os.getenv('CRYPTO_API_KEY') or config.get('crypto_api_key')
        self.api_secret = os.getenv('CRYPTO_SECRET_KEY') or config.get('crypto_secret_key')

        if not self.api_key or not self.api_secret:
            logger.error("Crypto.com API credentials not found")
            raise ValueError("Missing Crypto.com API credentials.")

        # --------------------------------------------------------------------
        # Set up device & exchange
        # --------------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.exchange = ccxt.cryptocom({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
        })
        self.exchange.set_sandbox_mode(False)

        try:
            self.exchange.load_markets()
            logger.info("Loaded markets successfully")
        except Exception as e:
            logger.error(f"Error loading markets: {e}", exc_info=True)
            raise ConnectionError("Failed to load markets from Crypto.com")

        # --------------------------------------------------------------------
        # Initialize data directories & submodules
        # --------------------------------------------------------------------
        self.data_dir = os.path.join(Path(__file__).parent.parent, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.market_analyzer = MarketAnalyzer()
        self.data_processor = DataProcessor()
        self.indicators = Indicators()
        self.portfolio = PortfolioManager()

        self.crypto_holdings = {}   # { "BTC/USD": {"amount": ..., "stop_loss": ..., "take_profit": ...}, ... }
        self.trade_history = []
        self.models = {}
        self.scalers = {}
        self.load_pretrained_models()

    # ------------------------------------------------------------------------
    # Load TCN-based ML models
    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # Market Data Fetching
    # ------------------------------------------------------------------------
    async def get_market_data(self, pair: str, timeframe: str = '1m') -> Optional[pd.DataFrame]:
        """
        Fetches OHLCV data for the given pair and timeframe, then adds technical features.
        """
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
            return df

        except Exception as e:
            logger.error(f"Error getting market data for {pair}: {e}")
            return None

    def fetch_all_timeframes(self, pair: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch multi-timeframe data for a pair, to feed into ML model if needed.
        """
        timeframes = ["1m", "5m", "15m"]
        historical_hours = {"1m": 1, "5m": 12, "15m": 24}
        data = {}

        for tf in timeframes:
            since = self.exchange.milliseconds() - (historical_hours[tf] * 3600 * 1000)
            try:
                ohlcv = self.exchange.fetch_ohlcv(pair, tf, since=since)
                if not ohlcv:
                    continue

                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                data[tf] = df

            except Exception as e:
                logger.error(f"Error fetching {tf} data for {pair}: {e}")

        return data

    # ------------------------------------------------------------------------
    # ML Prediction Helper
    # ------------------------------------------------------------------------
    async def _get_ml_predictions(self, pair: str, timeframe_data) -> Optional[torch.Tensor]:
        """
        Generates ML predictions for the given trading pair, if both a model and scaler exist.
        Returns a predictions tensor if successful, otherwise None.
        """
        if pair in self.models and pair in self.scalers:
            try:
                model_input = self.data_processor.prepare_model_input(
                    timeframe_data,
                    self.scalers[pair]
                )
                with torch.no_grad():
                    predictions = self.models[pair](model_input.to(self.device))
                return predictions
            except Exception as e:
                logger.error(f"Error getting ML predictions for {pair}: {e}", exc_info=True)
        return None

    # ------------------------------------------------------------------------
    # Example: Using Indicators + ML to Make a Decision
    # ------------------------------------------------------------------------
    def generate_decision(self, df: pd.DataFrame, indicators: Dict, ml_predictions: Optional[torch.Tensor]) -> Dict:
        """
        Here you combine indicator logic and ML predictions to produce a final trading decision.
        For now, it's just a simple placeholder logic that always HOLDs, or
        tries to interpret an example ML output shape of (1,3).
        Adjust as needed for your real trading rules.
        """
        if df is None:
            return self._default_decision()

        # Basic threshold or classification approach:
        if ml_predictions is not None:
            # Example: interpret model as [sell_prob, hold_prob, buy_prob]
            probs = torch.softmax(ml_predictions.squeeze(), dim=0).tolist()
            sell_p, hold_p, buy_p = probs

            # Simple logic
            if buy_p > 0.5:
                return {
                    "decision": {
                        "action": "BUY",
                        "size": 0.1,     # 10% of available
                        "stop_loss": None,
                        "take_profit": None
                    },
                    "reasoning": {
                        "technical_analysis": "ML suggests BUY",
                        "risk_assessment": f"buy_prob={buy_p:.2f}",
                        "market_context": "Auto ML-based signal"
                    }
                }
            elif sell_p > 0.5:
                return {
                    "decision": {
                        "action": "SELL",
                        "size": 1.0,     # Sell entire position
                        "stop_loss": None,
                        "take_profit": None
                    },
                    "reasoning": {
                        "technical_analysis": "ML suggests SELL",
                        "risk_assessment": f"sell_prob={sell_p:.2f}",
                        "market_context": "Auto ML-based signal"
                    }
                }
            else:
                return {
                    "decision": {
                        "action": "HOLD",
                        "size": 0.0,
                        "stop_loss": None,
                        "take_profit": None
                    },
                    "reasoning": {
                        "technical_analysis": "ML suggests HOLD",
                        "risk_assessment": f"hold_prob={hold_p:.2f}",
                        "market_context": "Auto ML-based signal"
                    }
                }
        else:
            # If no ML predictions are available, you can rely on indicators or just hold
            return self._default_decision()

    # ------------------------------------------------------------------------
    # Executes a trade based on the final decision
    # ------------------------------------------------------------------------
    async def execute_trade(self, pair: str, trading_decision: Dict):
        try:
            portfolio_state = self.portfolio.get_portfolio_summary()
            if portfolio_state['usd_balance'] < self.min_balance_threshold:
                logger.warning(f"Insufficient balance (${portfolio_state['usd_balance']:.2f}) to trade.")
                return

            action = trading_decision['decision']['action']
            if action not in ["BUY", "SELL"]:
                return

            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, pair)
            current_price = ticker['last']

            if action == "BUY":
                position_size = portfolio_state['usd_balance'] * float(
                    trading_decision['decision'].get('size', 0.1)
                )
                amount = position_size / current_price
                success = self.portfolio.execute_trade(
                    pair=pair,
                    action='BUY',
                    price=current_price,
                    size=position_size,
                    decision_data=trading_decision
                )
                if success:
                    logger.info(f"Executed BUY for {pair}: {amount} @ ${current_price}")

            elif action == "SELL":
                position = self.portfolio.get_position(pair)
                if position['amount'] > 0:
                    success = self.portfolio.execute_trade(
                        pair=pair,
                        action='SELL',
                        price=current_price,
                        size=position['amount'] * current_price,
                        decision_data=trading_decision
                    )
                    if success:
                        logger.info(f"Executed SELL for {pair}: {position['amount']} @ ${current_price}")

        except Exception as e:
            logger.error(f"Error executing trade for {pair}: {e}")

    # ------------------------------------------------------------------------
    # Default fallback decision
    # ------------------------------------------------------------------------
    def _default_decision(self):
        return {
            "decision": {
                "action": "HOLD",
                "size": 0.0,
                "stop_loss": None,
                "take_profit": None
            },
            "reasoning": {
                "technical_analysis": "No ML or no clear signal",
                "risk_assessment": "N/A",
                "market_context": "Neutral fallback"
            }
        }

    # ------------------------------------------------------------------------
    # Process each pair (Indicator + ML logic, then trade)
    # ------------------------------------------------------------------------
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

            # Check any existing stop-loss/take-profit
            if holdings.get('amount', 0) > 0:
                print(f"Current position: {holdings['amount']:.8f} {pair}")
                stop_loss = holdings.get('stop_loss')
                take_profit = holdings.get('take_profit')
                print(f"Stop Loss: ${stop_loss}, Take Profit: ${take_profit}")

                if await self._check_position_limits(pair, current_price, holdings):
                    return

            # Calculate technical indicators
            print("Calculating indicators...")
            indicators = self.indicators.calculate_all(df)

            # Get multi-timeframe data and run ML predictions
            print("Getting ML predictions...")
            timeframe_data = await asyncio.to_thread(self.fetch_all_timeframes, pair)
            ml_predictions = await self._get_ml_predictions(pair, timeframe_data)
            print(f"ML Predictions: {ml_predictions}")

            # Combine indicator logic & ML into a final decision
            final_decision = self.generate_decision(df, indicators, ml_predictions)
            print(f"Decision: {final_decision['decision']['action']} {pair}, "
                  f"Size: {final_decision['decision'].get('size', 0)}, "
                  f"Reasoning: {final_decision['reasoning'].get('technical_analysis', 'N/A')}")

            # Execute trade
            await self.execute_trade(pair, final_decision)

        except Exception as e:
            print(f"Error processing {pair}: {e}")
            logger.error(f"Error processing pair {pair}: {e}")

    async def _check_position_limits(self, pair: str, current_price: float, holdings: dict) -> bool:
        """
        Check if current_price triggers an existing stop_loss or take_profit.
        If triggered, executes a SELL for the entire position.
        """
        stop_loss = holdings.get('stop_loss')
        take_profit = holdings.get('take_profit')

        if stop_loss and current_price <= float(stop_loss):
            await self.execute_trade(pair, {
                'decision': {'action': 'SELL', 'size': 1.0}
            })
            return True

        if take_profit and current_price >= float(take_profit):
            await self.execute_trade(pair, {
                'decision': {'action': 'SELL', 'size': 1.0}
            })
            return True

        return False

    # ------------------------------------------------------------------------
    # Portfolio & Diversity
    # ------------------------------------------------------------------------
    def analyze_portfolio_diversity(self):
        try:
            total_portfolio_value = self.portfolio.balance
            pair_values = {}

            for pair, holdings in self.crypto_holdings.items():
                if holdings['amount'] > 0:
                    ticker = self.exchange.fetch_ticker(pair)
                    current_price = ticker['last']
                    pair_value = holdings['amount'] * current_price
                    total_portfolio_value += pair_value
                    pair_values[pair] = pair_value

            allocation_percentages = {
                pair: value / total_portfolio_value * 100
                for pair, value in pair_values.items()
            }

            recommendations = []
            for pair, percentage in allocation_percentages.items():
                if percentage > self.max_allocation_per_pair * 100:
                    recommendations.append({
                        'pair': pair,
                        'current_allocation': percentage,
                        'action': 'REDUCE',
                        'reason': 'Overexposure'
                    })

            underrepresented = [
                pair for pair in self.trading_pairs
                if pair not in allocation_percentages
            ]

            if underrepresented:
                recommendations.append({
                    'pairs': underrepresented,
                    'action': 'EXPLORE',
                    'reason': 'No current position'
                })

            return {
                'total_portfolio_value': total_portfolio_value,
                'current_allocations': allocation_percentages,
                'recommendations': recommendations
            }

        except Exception as e:
            logger.error(f"Error analyzing portfolio diversity: {e}")
            return {}

    def suggest_trades_for_diversity(self, diversity_analysis):
        try:
            if not diversity_analysis or not diversity_analysis.get('recommendations'):
                return []

            trade_suggestions = []
            for recommendation in diversity_analysis['recommendations']:
                if recommendation['action'] == 'REDUCE':
                    trade_suggestions.append({
                        'pair': recommendation['pair'],
                        'action': 'SELL',
                        'reason': 'Portfolio rebalancing'
                    })
                if recommendation['action'] == 'EXPLORE':
                    for p in recommendation['pairs']:
                        trade_suggestions.append({
                            'pair': p,
                            'action': 'EXPLORE_BUY',
                            'reason': 'Increase portfolio diversity'
                        })
            return trade_suggestions
        except Exception as e:
            logger.error(f"Error generating diversity trade suggestions: {e}")
            return []

    # ------------------------------------------------------------------------
    # Trade History
    # ------------------------------------------------------------------------
    def record_trade(self, pair: str, action: str, price: float, size: float, profit_loss: float, decision_data: Dict):
        trade = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'type': action,
            'price': price,
            'size': size,
            'profit_loss': profit_loss,
            'reasoning': decision_data.get('reasoning', {})
        }
        self.trade_history.append(trade)
        self.save_trade_history()

    def save_trade_history(self):
        history_path = os.path.join(self.data_dir, 'trade_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.trade_history, f, indent=2)

    # ------------------------------------------------------------------------
    # Portfolio & Stats
    # ------------------------------------------------------------------------
    def get_account_balance(self) -> Optional[Dict]:
        try:
            balance = self.exchange.fetch_balance()
            logger.info(f"Fetched account balance: {balance['total']}")
            return balance['total']
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}", exc_info=True)
            return None

    async def update_portfolio_value(self):
        try:
            portfolio_state = self.portfolio.get_portfolio_summary()
            print(f"\nPortfolio Summary:")
            print(f"Total Value: ${portfolio_state['total_value']:.2f}")
            print(f"USD Balance: ${portfolio_state['usd_balance']:.2f}")

            for pair, position in portfolio_state['positions'].items():
                if position['amount'] > 0:
                    print(f"{pair}: {position['amount']:.8f} @ ${position['avg_price']:.4f}")
                    if position.get('stop_loss'):
                        print(f"  Stop Loss: ${position['stop_loss']}")
                    if position.get('take_profit'):
                        print(f"  Take Profit: ${position['take_profit']}")
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")

    async def save_portfolio_stats(self):
        """
        Optional: Save or log portfolio stats periodically.
        """
        # Implement your method of archiving or logging portfolio snapshots
        pass

    # ------------------------------------------------------------------------
    # Main Trading Loop
    # ------------------------------------------------------------------------
    def run_trading_loop(self, interval_minutes: int = 1):
        async def _trading_loop():
            print("Starting trading loop")
            last_stats_update = datetime.now()

            while True:
                try:
                    # Update portfolio status
                    print("\nUpdating portfolio status...")
                    await self.update_portfolio_value()

                    # Occasionally do portfolio diversity analysis
                    if random.random() < 0.1:
                        diversity_analysis = self.analyze_portfolio_diversity()
                        diversity_suggestions = self.suggest_trades_for_diversity(diversity_analysis)
                        if diversity_suggestions:
                            print("\nDiversity Suggestions:")
                            for suggestion in diversity_suggestions:
                                print(f"- {suggestion['pair']}: {suggestion['action']} - {suggestion['reason']}")

                    # Save portfolio stats every 4 hours (optional)
                    current_time = datetime.now()
                    if (current_time - last_stats_update).total_seconds() >= 4 * 3600:
                        await self.save_portfolio_stats()
                        last_stats_update = current_time

                    # Fetch balances
                    print("\nFetching account balance...")
                    balance = await asyncio.to_thread(self.get_account_balance)
                    print(f"Account Balance: {balance}")

                    # Process each trading pair
                    print("\nProcessing trading pairs...")
                    tasks = [self.process_trading_pair(pair) for pair in self.trading_pairs]
                    await asyncio.gather(*tasks)

                except Exception as e:
                    print(f"Error in trading loop: {e}")
                    logger.error(f"Error in trading loop: {e}")

                print(f"\nSleeping for {interval_minutes} minutes...")
                await asyncio.sleep(interval_minutes * 60)

        return _trading_loop()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    trader = LiveTrader()
    asyncio.run(trader.run_trading_loop())
