import asyncio
import pandas as pd
from typing import Dict
import logging
import os
from pathlib import Path
import json
import aiohttp
from datetime import datetime, timedelta
from .ml_trader import MLTrader
from .market_analyzer import MarketAnalyzer
from .data_processor import DataProcessor

logger = logging.getLogger(__name__)

class AutomatedTrader:
    def __init__(self):
        self.ml_trader = MLTrader()
        self.market_analyzer = MarketAnalyzer()
        self.data_processor = DataProcessor()
        self.ollama_url = "http://localhost:11434/api/generate"
        self.data_dir = os.path.join(Path(__file__).parent.parent, 'data')
        self.position = 0
        self.balance = 10000
        self.trade_history = []
        
    async def run_trading_loop(self, interval_minutes: int = 1):
        print(f"\nStarting automated trading with {interval_minutes}min interval")
        
        while True:
            try:
                # Get latest market data
                dataframes = self._load_latest_data()
                
                # Get ML predictions
                ml_signals = self.ml_trader.predict(dataframes)
                
                # Get market regime info
                regime_data = self.market_analyzer.analyze_market_regime(dataframes['1m'])
                
                # Get LLM analysis
                llm_decision = await self._get_llm_analysis(dataframes['1m'], ml_signals, regime_data)
                
                # Execute trades based on combined signals
                await self._execute_trades(llm_decision, ml_signals, dataframes['1m'])
                
                # Save state
                self._save_state()
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
                
    async def _get_llm_analysis(self, df: pd.DataFrame, ml_signals: Dict, regime_data: Dict) -> Dict:
        """Get LLM analysis of current market conditions"""
        current_price = float(df['close'].iloc[-1])
        
        prompt = f"""Analyze current market conditions and recommend trading action.

Market Data:
- Current Price: ${current_price:.4f}
- 24h Change: {float(df['close'].pct_change(1440).iloc[-1] * 100):.2f}%
- Volume: {float(df['volume'].iloc[-1440:].sum()):.2f}

ML Model Signals:
{json.dumps(ml_signals, indent=2)}

Market Regime:
{json.dumps(regime_data, indent=2)}

Current Position: {self.position}
Current Balance: ${self.balance:.2f}

Recommend 'BUY', 'SELL', or 'HOLD' based on:
1. Price action and volume
2. ML signals
3. Market regime
4. Risk management

Format response as JSON with reasoning and confidence level."""

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.ollama_url,
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "temperature": 0.2
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return json.loads(result['response'])
                return {"action": "HOLD", "confidence": 0, "reasoning": "API error"}
                
    async def _execute_trades(self, llm_decision: Dict, ml_signals: Dict, df: pd.DataFrame):
        """Execute trades based on combined signals"""
        current_price = float(df['close'].iloc[-1])
        
        # Combine ML and LLM signals
        ml_confidence = float(ml_signals['confidence'])
        llm_confidence = float(llm_decision['confidence'])
        
        # Only trade if both signals agree with high confidence
        if llm_decision['action'] == "BUY" and ml_confidence > 0.7 and llm_confidence > 0.7:
            if self.position <= 0:
                trade_size = self.balance * 0.1  # 10% position size
                self.position = trade_size / current_price
                self.balance -= trade_size
                self._log_trade("BUY", current_price, trade_size)
                
        elif llm_decision['action'] == "SELL" and ml_confidence > 0.7 and llm_confidence > 0.7:
            if self.position >= 0:
                trade_value = self.position * current_price
                self.balance += trade_value
                self.position = 0
                self._log_trade("SELL", current_price, trade_value)
                
    def _log_trade(self, action: str, price: float, value: float):
        """Log trade details"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'price': price,
            'value': value,
            'balance': self.balance,
            'position': self.position
        }
        self.trade_history.append(trade)
        print(f"\nExecuted {action} at ${price:.4f} for ${value:.2f}")
        
    def _save_state(self):
        """Save current trading state"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'balance': self.balance,
            'position': self.position,
            'trade_history': self.trade_history
        }
        
        state_path = os.path.join(self.data_dir, 'trading_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
            
    def _load_latest_data(self) -> Dict[str, pd.DataFrame]:
        """Load latest market data for all timeframes"""
        timeframes = {
            '1m': '1INCHUSD_1m_historical.csv',
            '5m': '1INCHUSD_5m_historical.csv',  
            '15m': '1INCHUSD_15m_historical.csv'
        }
        
        dataframes = {}
        for timeframe, filename in timeframes.items():
            filepath = os.path.join(self.data_dir, 'historical', filename)
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            dataframes[timeframe] = df
            
        return dataframes