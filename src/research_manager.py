import pandas as pd
import logging
import asyncio
from typing import Dict, List
from datetime import datetime
import os
import json
from pathlib import Path
import torch

# Local imports (update to match your actual modules)
from .data_processor import DataProcessor
from .market_analyzer import MarketAnalyzer
from .llm_analyzer import LLMAnalyzer
from .indicator_combinations import IndicatorCombinations
from . import IndicatorVisualizer
from . import MLTrader  # <--- Updated import to your TCN-based MLTrader

logger = logging.getLogger(__name__)

class ResearchManager:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.market_analyzer = MarketAnalyzer()
        self.llm_analyzer = LLMAnalyzer()
        self.indicator_combinations = IndicatorCombinations()
        self.visualizer = IndicatorVisualizer()
        self.ml_trader = MLTrader()  # <-- TCN-based model
        self.data_dir = os.path.join(Path(__file__).parent.parent, 'data')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    async def run_indicator_analysis(self):
        """Example workflow for indicator testing + LLM analysis."""
        try:
            data_path = os.path.join(self.data_dir, 'historical', 'ADAUSD_1m_historical.csv')
            df = self.data_processor.load_market_data(data_path)
            if df is None:
                logger.error("Failed to load market data for indicator analysis.")
                return

            # Add any technical indicators
            df = self.data_processor.add_technical_features(df)

            # Test various indicator combos
            combination_results = await self.indicator_combinations.test_combinations(df)

            # Run Market Regime detection
            regime_data = self.market_analyzer.analyze_market_regime(df)

            # Identify key levels (support/resistance, pivot points, etc.)
            key_levels = self.market_analyzer.find_key_levels(df)

            market_data = {
                'symbol': 'ADA/USD',
                'price': float(df['close'].iloc[-1]),
                'change24h': float(df['close'].pct_change(1440).iloc[-1] * 100)
            }

            # Ask LLM for a reading on these indicators
            llm_analysis = await self.llm_analyzer.analyze_indicators(market_data, regime_data)

            # Extract or calculate some performance metrics
            metrics = {
                'success_rate': max(combo['success_rate'] for combo in combination_results.values()),
                'risk_level': llm_analysis['recommendation']['position']
            }

            # Generate some example plots
            perf_plot = self.visualizer.plot_combination_performance(combination_results)
            metrics_plot = self.visualizer.plot_performance_metrics(metrics)

            # Save results to JSON or DB
            self._save_results(combination_results, regime_data, llm_analysis)

        except Exception as e:
            logger.error(f"Error in indicator analysis: {e}")
            raise

    async def run_ml_analysis(self):
        """
        Train and backtest the TCN-based ML model on ADAUSD data.
        - Adjust pair name, epochs, batch_size, etc. to suit your needs.
        - Ensure your CSVs match the naming/paths that MLTrader.load_data expects.
        """
        try:
            print("\n=== ML Analysis with TCN Model ===")

            # We train the TCN using your "ADAUSD" data
            pair_name = "ADAUSD"

            print(f"\nTraining TCN-based ML model for {pair_name}...")
            # The train() method is async, so we must await it
            # It will look for data in /data/historical/ADAUSD/{ADAUSD_1m_data.csv, ...}
            training_results = await self.ml_trader.train(
                pair=pair_name,
                epochs=30,
                batch_size=64
            )
            if not training_results:
                print("No training results were returned.")
                return

            print("\nTraining logs for each epoch are in training_results. Now running a backtest...")

            # After training, we can load the same data from disk and run a backtest
            dataframes = self.ml_trader.load_data(pair_name)
            if not dataframes:
                logger.error(f"No data found for pair {pair_name}. Check your CSV paths.")
                return

            backtest_results = self.ml_trader.backtest(dataframes, initial_balance=10000.0, position_size=0.1)

            final_balance = backtest_results['final_balance']
            final_return = backtest_results['return']
            n_trades = backtest_results['n_trades']

            print(f"\n=== Backtest Completed for {pair_name} ===")
            print(f"Final Balance: ${final_balance:.2f}")
            print(f"Return: {final_return:.2f}%")
            print(f"Total Trades: {n_trades}")

            return {
                "training_results": training_results,
                "backtest_results": backtest_results
            }

        except Exception as e:
            logger.error(f"Error in ML analysis: {e}")
            raise

    async def run_llm_analysis(self):
        """Run LLM-only market analysis on ADAUSD 1m data."""
        try:
            data_path = os.path.join(self.data_dir, 'historical', 'ADAUSD_1m_historical.csv')
            df = self.data_processor.load_market_data(data_path)
            if df is None:
                logger.error("Failed to load market data for LLM analysis.")
                return

            market_data = {
                'symbol': 'ADA/USD',
                'price': float(df['close'].iloc[-1]),
                'change24h': float(df['close'].pct_change(1440).iloc[-1] * 100),
                'volume24h': float(df['volume'].iloc[-1440:].sum())
            }

            regime_data = self.market_analyzer.analyze_market_regime(df)
            llm_analysis = await self.llm_analyzer.analyze_indicators(market_data, regime_data)

            # Save or print the LLM results
            self._save_results({}, regime_data, llm_analysis)

            logger.info("LLM analysis completed successfully")
            return llm_analysis

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            raise

    def _save_results(self, combination_results, regime_data, llm_analysis):
        """
        Example method showing how to save or log your results to JSON.
        You can adapt or remove as needed.
        """
        output_dir = os.path.join(self.data_dir, 'analysis_results')
        os.makedirs(output_dir, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        result_path = os.path.join(output_dir, f"analysis_{timestamp}.json")

        payload = {
            'timestamp': timestamp,
            'combination_results': combination_results,
            'regime_data': regime_data,
            'llm_analysis': llm_analysis
        }

        with open(result_path, 'w') as f:
            json.dump(payload, f, indent=2)

        print(f"\nIndicator/LLM analysis results saved to: {result_path}")

    def _save_ml_results(self, results: Dict, combined_results: Dict):
        """If you have other combined ML results to save, adapt this method accordingly."""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(self.data_dir, 'analysis_results')
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'ml_results_{timestamp}.json')
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'individual_results': results,
                'combined_results': combined_results
            }, f, indent=2)

        print(f"\nML analysis results saved to: {output_path}")
