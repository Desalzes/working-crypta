import pandas as pd
import logging
from typing import Dict
import os
import json
from pathlib import Path
import torch
import logging
# Local imports
from .data_processor import DataProcessor
from .market_analyzer import MarketAnalyzer
from .llm_analyzer import LLMAnalyzer, logger
from src.indicators.indicator_combinations import IndicatorCombinations
from . import IndicatorVisualizer
from . import MLTrader

class ResearchManager:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.market_analyzer = MarketAnalyzer()
        self.llm_analyzer = LLMAnalyzer()
        self.indicator_combinations = IndicatorCombinations()
        self.visualizer = IndicatorVisualizer()
        self.ml_trader = MLTrader()
        self.data_dir = os.path.join(Path(__file__).parent.parent, 'data')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    async def run_indicator_analysis(self):
        try:
            df = self.data_processor.load_market_data(os.path.join(self.data_dir, 'historical', 'ADAUSD_1m_historical.csv'))
            if df is None: return
            df = self.data_processor.add_technical_features(df)
            combo_results = await self.indicator_combinations.test_combinations(df)
            regime_data = self.market_analyzer.analyze_market_regime(df)
            key_levels = self.market_analyzer.find_key_levels(df)
            market_data = {'symbol': 'ADA/USD', 'price': df['close'].iloc[-1], 'change24h': df['close'].pct_change(1440).iloc[-1] * 100}
            llm_analysis = await self.llm_analyzer.analyze_indicators(market_data, regime_data)
            metrics = {'success_rate': max(c['success_rate'] for c in combo_results.values()), 'risk_level': llm_analysis['recommendation']['position']}
            self._save_results(combo_results, regime_data, llm_analysis)
        except Exception as e:
            logger.error(f"Error: {e}")

    async def run_ml_analysis(self):
        try:
            print("\n=== ML Analysis with TCN Model ===")
            pair = "ADAUSD"
            print(f"Training TCN model for {pair}...")
            results = await self.ml_trader.train(pair, 30, 64)
            if not results: print("No results.")
            else:
                dataframes = self.ml_trader.load_data(pair)
                backtest = self.ml_trader.backtest(dataframes, 10000, 0.1)
                print(f"Final Balance: ${backtest['final_balance']:.2f}, Return: {backtest['return']:.2f}%, Trades: {backtest['n_trades']}")
                return {'training': results, 'backtest': backtest}
        except Exception as e:
            logger.error(f"Error: {e}")

    async def run_ml_analysis(self):
        try:
            print("\n=== ML Analysis with TCN Model ===")
            # Read trading pairs from config
            with open(os.path.join(Path(__file__).parent, 'config.json'), 'r') as f:
                config = json.load(f)

            results = {}
            for pair in config['trading_pairs']:
                pair_symbol = f"{pair}USD"
                print(f"Training TCN model for {pair_symbol}...")
                pair_results = await self.ml_trader.train(pair_symbol, 30, 64)

                if pair_results:
                    dataframes = self.ml_trader.load_data(pair_symbol)
                    backtest = self.ml_trader.backtest(dataframes, 10000, 0.1)
                    results[pair_symbol] = {
                        'training': pair_results,
                        'backtest': backtest
                    }
                    print(
                        f"{pair_symbol} - Final Balance: ${backtest['final_balance']:.2f}, Return: {backtest['return']:.2f}%, Trades: {backtest['n_trades']}")
                else:
                    print(f"No results for {pair_symbol}")

            return results
        except Exception as e:
            logger.error(f"Error: {e}")

    def _save_results(self, combo_results, regime_data, llm_analysis):
        output_dir = os.path.join(self.data_dir, 'analysis_results')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(output_dir, f"analysis_{timestamp}.json")
        json.dump({'timestamp': timestamp, 'combo': combo_results, 'regime': regime_data, 'llm': llm_analysis}, open(path, 'w'), indent=2)
        print(f"Saved to {path}")

    def _save_ml_results(self, results, combined):
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.data_dir, 'analysis_results', f'ml_results_{timestamp}.json')
        json.dump({'timestamp': timestamp, 'individual': results, 'combined': combined}, open(path, 'w'), indent=2)
        print(f"Saved ML results to {path}")