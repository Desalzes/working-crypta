import os
import json
import logging
from typing import Dict, Optional, List
import pandas as pd
import asyncio
import aiohttp
from datetime import datetime
import hmac
import hashlib
import base64
import time
import urllib.parse
from pathlib import Path
from src.ml_stuff.ml_trader import MLTrader
from .market_analyzer import MarketAnalyzer
from .data_processor import DataProcessor
from .indicators.indicators import Indicators

logger = logging.getLogger(__name__)


class PaperTrader:
    def __init__(self):
        # Setup directories early so other methods can use them
        self.data_dir = os.path.join(Path(__file__).parent.parent, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize trading pairs and API credentials.
        self.trading_pairs = []
        self.api_key = None
        self.api_secret = None


        # Load config
        config_path = os.path.join(Path(__file__).parent, 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.api_key = config.get('kraken_api_key')
                self.api_secret = config.get('kraken_secret_key')
                self.trading_pairs = config.get('trading_pairs', [])
                logger.info(f"Loaded {len(self.trading_pairs)} trading pairs from config")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            print(f"Error loading config.json: {e}")

        self.api_url = "https://api.kraken.com"
        self.ollama_url = "http://localhost:11434/api/generate"

        # Initialize state
        self.balance = 10000  # Starting paper balance
        self.positions: Dict[str, float] = {}  # Current positions
        self.trade_history = []
        self.crypto_holdings = {}

        # Now that self.data_dir is set, load the ML model
        self.ml_trader = MLTrader()
        self._load_ml_model()

        self.market_analyzer = MarketAnalyzer()
        self.data_processor = DataProcessor()
        self.indicators = Indicators()

        logger.info("PaperTrader initialized successfully")
        print(f"Loaded {len(self.trading_pairs)} trading pairs: {', '.join(self.trading_pairs)}")

        # Load balance state
        self._load_balance()

    def _load_ml_model(self) -> None:
        """Load pre-trained model and scalers"""
        model_dir = os.path.join(self.data_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)

        # Find model files (both regular and _best suffix)
        model_files = []
        for f in os.listdir(model_dir):
            if f.endswith('.pt'):  # Model file
                scaler_name = f.replace('.pt', '_scalers.pkl')
                if os.path.exists(os.path.join(model_dir, scaler_name)):
                    model_files.append((f, scaler_name))

        if not model_files:
            print("No trained models found. Please train models first.")
            return

        # Prefer _best models if available
        best_models = [pair for pair in model_files if '_best' in pair[0]]
        if best_models:
            model_name, scaler_name = max(best_models, key=lambda x: os.path.getmtime(os.path.join(model_dir, x[0])))
        else:
            model_name, scaler_name = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x[0])))

        model_path = os.path.join(model_dir, model_name)
        scaler_path = os.path.join(model_dir, scaler_name)

        try:
            print(f"\nLoading model: {model_name}")
            self.ml_trader.load_model(model_path, scaler_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train models before running paper trading.")

    def _load_balance(self) -> None:
        """Load balance from balance.json"""
        balance_path = os.path.join(self.data_dir, 'balance.json')
        if os.path.exists(balance_path):
            try:
                with open(balance_path, 'r') as f:
                    balance_data = json.load(f)
                    self.balance = balance_data.get('USD', 10000.00)
                    self.crypto_holdings = balance_data.get('crypto_holdings', {})
                    self.trade_history = balance_data.get('trade_history', [])
            except Exception as e:
                logger.error(f"Error loading balance: {e}")
                self.balance = 10000.00
                self.crypto_holdings = {}
                self.trade_history = []
        else:
            # Create initial balance file
            self._save_balance()

    def _save_balance(self) -> None:
        """Save current balance to balance.json"""
        # Calculate net worth
        net_worth = sum(
            holdings.get('amount', 0) * holdings.get('avg_price', 0)
            for pair, holdings in self.crypto_holdings.items()
        )

        # Create simplified balance data
        balance_data = {
            "net_worth": round(net_worth, 2),
            "usd_balance": round(self.balance, 2),
            "xrp": round(self.crypto_holdings.get('XRPUSDT', {}).get('total_cost', 0), 2),
            "eth": round(self.crypto_holdings.get('ETHUSDT', {}).get('total_cost', 0), 2)
        }

        balance_path = os.path.join(self.data_dir, 'balance.json')
        with open(balance_path, 'w') as f:
            json.dump(balance_data, f, indent=2)
    def _get_kraken_signature(self, urlpath: str, data: Dict) -> str:
        """Generate Kraken API signature"""
        post_data = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + post_data).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()

        signature = hmac.new(base64.b64decode(self.api_secret),
                             message,
                             hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()

    async def _kraken_request(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make authenticated request to Kraken API"""
        if data is None:
            data = {}

        data['nonce'] = str(int(time.time() * 1000))
        headers = {
            'API-Key': self.api_key,
            'API-Sign': self._get_kraken_signature(endpoint, data)
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.api_url}{endpoint}",
                                    headers=headers,
                                    data=data) as response:
                return await response.json()

    async def get_market_data(self, pair: str, timeframe: str = '1m', historical_hours: int = 72) -> Optional[
        pd.DataFrame]:
        """Get market data from Kraken with longer historical window"""
        try:
            def convert_to_kraken_pair(pair):
                pair_mapping = {
                    'BTCUSDT': 'XXBTZUSD',
                    'XRPUSDT': 'XXRPZUSD',
                    'ETHUSDT': 'XETHZUSD',
                    'SOLUSDT': 'SOLUSD',
                    'DOGEUSDT': 'DOGEUSD',
                    'ADAUSDT': 'ADAUSD',
                    'LTCUSDT': 'XLTCZUSD',
                    'XLMUSDT': 'XXLMZUSD',
                    'LINKUSDT': 'LINKUSD',
                    'ALGOUSDT': 'ALGOUSD',
                    'SHIBUSDT': 'SHIBUSD',
                    'AVAXUSDT': 'AVAXUSD',
                    'FTMUSDT': 'FTMUSD',
                    'NEARUSDT': 'NEARUSD',
                    'PEPEUSDT': 'PEPEUSD',
                    'USDCUSDT': 'USDCUSD'
                }
                return pair_mapping.get(pair, pair.replace('USDT', 'USD'))

            kraken_pair = convert_to_kraken_pair(pair)

            # Map timeframes to Kraken intervals
            interval_map = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '1h': 60,
                '4h': 240,
                '1d': 1440
            }
            interval = interval_map.get(timeframe, 1)

            # Get more historical data
            since_time = int(time.time() - historical_hours * 3600)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"{self.api_url}/0/public/OHLC",
                        params={
                            "pair": kraken_pair,
                            "interval": interval,
                            "since": since_time
                        }
                ) as response:
                    data = await response.json()

                    if data.get('error'):
                        logger.error(f"Kraken API error for {pair}: {data['error']}")
                        return None

                    if "result" in data and kraken_pair in data["result"]:
                        ohlc = data["result"][kraken_pair]
                        df = pd.DataFrame(ohlc, columns=[
                            'timestamp', 'open', 'high', 'low', 'close',
                            'vwap', 'volume', 'count'
                        ])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        df = df.set_index('timestamp')

                        # Convert string values to float
                        for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        # Add technical indicators required by ML model
                        df = self.ml_trader.add_technical_features(df)

                        return df

            logger.error(f"No data received for {pair}")
            return None
        except Exception as e:
            logger.error(f"Error getting market data for {pair}: {e}")
            return None

    async def get_llm_analysis(self, df: pd.DataFrame, ml_signals: Dict,
                               regime_data: Dict, pair: str) -> Dict:
        """Get trading decision from Mistral"""
        current_price = float(df['close'].iloc[-1])
        position = self.crypto_holdings.get(pair, {}).get('amount', 0)

        # Calculate indicators
        indicators = self.indicators.calculate_all(df)

        # Calculate recent performance
        daily_return = float(df['close'].pct_change(1440).iloc[-1] * 100)
        hourly_return = float(df['close'].pct_change(60).iloc[-1] * 100)

        prompt = f"""You are an experienced cryptocurrency trading strategist. Make a trading decision based on the following data:

    Market Data:
    - Pair: {pair}
    - Current Price: ${current_price:.4f}
    - 24h Return: {daily_return:.2f}%
    - 1h Return: {hourly_return:.2f}%
    - 24h Volume: {float(df['volume'].iloc[-1440:].sum()):.2f}

    Technical Indicators:
    {json.dumps(indicators, indent=2)}

    ML Model Predictions:
    {json.dumps(ml_signals, indent=2)}

    Market Regime Analysis:
    {json.dumps(regime_data, indent=2)}

    Portfolio Status:
    - Current Position: {position}
    - Available Balance: ${self.balance:.2f}

    Make a decisive trading decision. Consider:
    1. Technical indicator signals and confirmations
    2. Market regime and volatility conditions
    3. ML model predictions
    4. Risk management and position sizing
    5. Current portfolio exposure

    Provide your decision in JSON format with this exact structure:
    {{
        "decision": {{
            "action": "BUY/SELL/HOLD",
            "size": "percentage of balance to use (0.0-1.0)",
            "stop_loss": "price level for stop loss",
            "take_profit": "price level for take profit"
        }},
        "reasoning": {{
            "technical_analysis": "your analysis of indicators",
            "ml_confirmation": "how ML signals influenced decision",
            "risk_assessment": "risk evaluation and management plan",
            "market_context": "current market conditions analysis"
        }}
    }}"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.ollama_url,
                        json={
                            "model": "mistral",
                            "prompt": prompt,
                            "temperature": 0.2,
                            "stream": False
                        }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        try:
                            return json.loads(result.get('response', '{}'))
                        except:
                            return {
                                "decision": {
                                    "action": "HOLD",
                                    "size": 0,
                                    "stop_loss": None,
                                    "take_profit": None
                                },
                                "reasoning": {
                                    "technical_analysis": "Error parsing response",
                                    "ml_confirmation": "",
                                    "risk_assessment": "",
                                    "market_context": ""
                                }
                            }
                    else:
                        logger.error(f"LLM request failed with status {response.status}")
                        return {
                            "decision": {
                                "action": "HOLD",
                                "size": 0,
                                "stop_loss": None,
                                "take_profit": None
                            },
                            "reasoning": {
                                "technical_analysis": "API error",
                                "ml_confirmation": "",
                                "risk_assessment": "",
                                "market_context": ""
                            }
                        }
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {e}")
            return {
                "decision": {
                    "action": "HOLD",
                    "size": 0,
                    "stop_loss": None,
                    "take_profit": None
                },
                "reasoning": {
                    "technical_analysis": str(e),
                    "ml_confirmation": "",
                    "risk_assessment": "",
                    "market_context": ""
                }
            }

    async def execute_trade(self, pair: str, action: str, price: float, size: float, llm_decision: Dict) -> None:
        """Execute paper trade"""
        timestamp = datetime.now()
        trade_type = None
        trade_profit_loss = None
        position_size = None

        if action == "BUY" and self.balance >= size:
            position_size = size / price
            self.balance -= size

            if pair not in self.crypto_holdings:
                self.crypto_holdings[pair] = {
                    'amount': 0,
                    'avg_price': 0,
                    'total_cost': 0
                }

            current = self.crypto_holdings[pair]
            total_amount = current['amount'] + position_size
            total_cost = current['total_cost'] + size
            avg_price = total_cost / total_amount if total_amount > 0 else 0

            self.crypto_holdings[pair] = {
                'amount': total_amount,
                'avg_price': avg_price,
                'total_cost': total_cost,
                'stop_loss': float(
                    str(llm_decision['decision'].get('stop_loss', 0)).replace('$', '').replace('$$', '')
                ),
                'take_profit': float(
                    str(llm_decision['decision'].get('take_profit', 0)).replace('$', '').replace('$$', '')
                ),
                'last_update': timestamp.isoformat()
            }

            trade_type = "BUY"

        elif action == "SELL" and pair in self.crypto_holdings:
            current = self.crypto_holdings[pair]
            if current['amount'] > 0:
                sell_amount = current['amount']
                trade_value = sell_amount * price
                self.balance += trade_value
                trade_profit_loss = trade_value - current['total_cost']

                self.crypto_holdings[pair] = {
                    'amount': 0,
                    'avg_price': 0,
                    'total_cost': 0,
                    'stop_loss': None,
                    'take_profit': None,
                    'last_update': timestamp.isoformat()
                }

                trade_type = "SELL"
                size = trade_value
            else:
                logger.warning(f"Invalid trade: SELL {pair} - no holdings")
                return
        else:
            logger.warning(f"Invalid trade: {action} {pair} @ {price}")
            return

        if trade_type:  # Only proceed if a valid trade was executed
            trade = {
                'timestamp': timestamp.isoformat(),
                'pair': pair,
                'type': trade_type,
                'price': price,
                'size': size,
                'usd_balance': self.balance,
                'crypto_amount': self.crypto_holdings[pair]['amount'],
                'avg_price': self.crypto_holdings[pair]['avg_price'],
                'stop_loss': llm_decision['decision'].get('stop_loss'),
                'take_profit': llm_decision['decision'].get('take_profit'),
                'reasoning': llm_decision['reasoning']
            }

            if trade_profit_loss is not None:
                trade['profit_loss'] = trade_profit_loss

            self.trade_history.append(trade)
            self._save_balance()

            print(f"\nExecuted {trade_type} {pair}:")
            print(f"Price: ${price:.4f}")
            print(f"Size: {size:.2f} USD")
            if trade_type == "BUY":
                print(f"Crypto Amount: {position_size:.8f}")
                print(f"Stop Loss: ${llm_decision['decision'].get('stop_loss', 'Not set')}")
                print(f"Take Profit: ${llm_decision['decision'].get('take_profit', 'Not set')}")
            else:
                print(f"Profit/Loss: ${trade_profit_loss:.2f}")
            print(f"New USD Balance: ${self.balance:.2f}")

            print("\nTrading Analysis:")
            print(f"Technical Analysis: {llm_decision['reasoning']['technical_analysis']}")
            print(f"ML Confirmation: {llm_decision['reasoning']['ml_confirmation']}")
            print(f"Risk Assessment: {llm_decision['reasoning']['risk_assessment']}")
            print(f"Market Context: {llm_decision['reasoning']['market_context']}")

    async def update_portfolio_value(self) -> None:
        """Update approximate total portfolio value"""
        total_value = self.balance

        for pair, holdings in self.crypto_holdings.items():
            if holdings['amount'] > 0:
                try:
                    df = await self.get_market_data(pair)
                    if df is not None:
                        current_price = float(df['close'].iloc[-1])
                        crypto_value = holdings['amount'] * current_price
                        total_value += crypto_value
                except Exception as e:
                    logger.error(f"Error updating value for {pair}: {e}")

        print(f"\nCurrent Portfolio Value: ${total_value:.2f}")
        print("Holdings:")
        print(f"USD: ${self.balance:.2f}")
        for pair, holdings in self.crypto_holdings.items():
            if holdings['amount'] > 0:
                print(f"{pair}: {holdings['amount']:.8f} @ ${holdings['avg_price']:.4f}")
                if holdings.get('stop_loss'):
                    print(f"  Stop Loss: ${holdings['stop_loss']}")
                if holdings.get('take_profit'):
                    print(f"  Take Profit: ${holdings['take_profit']}")

    async def run_trading_loop(self, timeframes: List[str] = ['1m', '5m', '15m'], interval_minutes: int = 1) -> None:
        """
        Run paper trading loop with specified timeframes
        Args:
            timeframes: List of timeframes to analyze ['1m', '5m', '15m']
            interval_minutes: Minutes between trading iterations
        """
        print(f"\nStarting paper trading with {interval_minutes}min interval")
        print(f"Analyzing timeframes: {', '.join(timeframes)}")
        print(f"Trading pairs: {', '.join(self.trading_pairs)}")
        print(f"Initial balance: ${self.balance:.2f}")

        if not self.api_key or not self.api_secret:
            raise ValueError("Kraken API credentials not found in config.json")

        while True:
            try:
                await self.update_portfolio_value()

                for pair in self.trading_pairs:
                    print(f"\nAnalyzing {pair}...")

                    # Get market data for all timeframes
                    market_data = {}
                    for tf in timeframes:
                        df = await self.get_market_data(pair, tf)
                        if df is None:
                            logger.warning(f"No data available for {pair} {tf}, skipping...")
                            continue
                        market_data[tf] = df

                    if not market_data:
                        continue

                    # Get ML predictions using all timeframes
                    ml_signals = self.ml_trader.predict(market_data)

                    # Market regime analysis using lowest timeframe
                    lowest_tf = min(timeframes, key=lambda x: int(''.join(filter(str.isdigit, x))))
                    regime_data = self.market_analyzer.analyze_market_regime(market_data[lowest_tf])

                    # Get LLM trading decision
                    llm_decision = await self.get_llm_analysis(market_data[lowest_tf], ml_signals, regime_data, pair)

                    current_price = float(market_data[lowest_tf]['close'].iloc[-1])
                    holdings = self.crypto_holdings.get(pair, {})

                    # Check stop loss/take profit if position exists
                    if holdings.get('amount', 0) > 0:
                        stop_loss = holdings.get('stop_loss')
                        take_profit = holdings.get('take_profit')

                        if stop_loss and current_price <= float(stop_loss):
                            print(f"\nStop Loss triggered for {pair} at ${current_price}")
                            llm_decision['decision'] = {
                                'action': 'SELL',
                                'size': 1.0,  # Sell entire position
                                'stop_loss': None,
                                'take_profit': None
                            }
                        elif take_profit and current_price >= float(take_profit):
                            print(f"\nTake Profit triggered for {pair} at ${current_price}")
                            llm_decision['decision'] = {
                                'action': 'SELL',
                                'size': 1.0,  # Sell entire position
                                'stop_loss': None,
                                'take_profit': None
                            }

                    if llm_decision['decision']['action'] in ['BUY', 'SELL']:
                        position_size = self.balance * float(llm_decision['decision']['size'])
                        await self.execute_trade(
                            pair,
                            llm_decision['decision']['action'],
                            current_price,
                            position_size,
                            llm_decision
                        )
                    else:
                        print(f"No trade executed for {pair}")
                        print("\nMistral's Analysis:")
                        print(f"Action: {llm_decision['decision']['action']}")
                        print(f"Technical Analysis: {llm_decision['reasoning']['technical_analysis']}")
                        print(f"ML Confirmation: {llm_decision['reasoning']['ml_confirmation']}")
                        print(f"Risk Assessment: {llm_decision['reasoning']['risk_assessment']}")
                        print(f"Market Context: {llm_decision['reasoning']['market_context']}")

                print(f"\nWaiting {interval_minutes} minutes until next analysis...")
                await asyncio.sleep(interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                print(f"\nError occurred: {e}")
                print("Retrying in 60 seconds...")
                await asyncio.sleep(60)