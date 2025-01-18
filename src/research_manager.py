import pandas as pd
import logging
import asyncio
from typing import Dict, List
from datetime import datetime
import os
import json
from . import DataProcessor
from . import MarketAnalyzer
from . import LLMAnalyzer
from . import IndicatorCombinations
from . import IndicatorVisualizer
from . import MLTrader
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

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
            data_path = os.path.join(self.data_dir, 'historical', 'ADAUSD_1m_historical.csv')
            df = self.data_processor.load_market_data(data_path)
            if df is None:
                logger.error("Failed to load market data")
                return

            df = self.data_processor.add_technical_features(df)
            
            # Run indicator combination testing (now async)
            combination_results = await self.indicator_combinations.test_combinations(df)
            regime_data = self.market_analyzer.analyze_market_regime(df)
            key_levels = self.market_analyzer.find_key_levels(df)

            market_data = {
                'symbol': 'ADA/USD',
                'price': float(df['close'].iloc[-1]),
                'change24h': float(df['close'].pct_change(1440).iloc[-1] * 100)
            }

            llm_analysis = await self.llm_analyzer.analyze_indicators(market_data, regime_data)
            
            metrics = {
                'success_rate': max(combo['success_rate'] for combo in combination_results.values()),
                'risk_level': llm_analysis['recommendation']['position']
            }
            
            perf_plot = self.visualizer.plot_combination_performance(combination_results)
            metrics_plot = self.visualizer.plot_performance_metrics(metrics)

            self._save_results(combination_results, regime_data, llm_analysis)

        except Exception as e:
            logger.error(f"Error in indicator analysis: {e}")
            raise

    async def run_ml_analysis(self):
        """Train and evaluate ML models"""
        try:
            print("\nLoading data for ML training...")
            data_path = os.path.join(self.data_dir, 'historical', 'ADAUSD_1m_historical.csv')
            df = self.data_processor.load_market_data(data_path)
            if df is None:
                logger.error("Failed to load market data")
                return

            print("\nTraining ML model...")
            results = self.ml_trader.train(epochs=100, batch_size=32)
            
            if results:
                print(f"\nTraining completed:")
                print(f"Final Balance: ${results['final_balance']:.2f}")
                print(f"Return: {results['return']:.2f}%")
                print(f"Total Trades: {len(results['trades'])}")

            return results

        except Exception as e:
            logger.error(f"Error in ML analysis: {e}")
            raise

    async def run_llm_analysis(self):
        """Run LLM-only market analysis"""
        try:
            data_path = os.path.join(self.data_dir, 'historical', 'ADAUSD_1m_historical.csv')
            df = self.data_processor.load_market_data(data_path)
            if df is None:
                logger.error("Failed to load market data")
                return

            market_data = {
                'symbol': 'ADA/USD',
                'price': float(df['close'].iloc[-1]),
                'change24h': float(df['close'].pct_change(1440).iloc[-1] * 100),
                'volume24h': float(df['volume'].iloc[-1440:].sum())
            }

            regime_data = self.market_analyzer.analyze_market_regime(df)
            llm_analysis = await self.llm_analyzer.analyze_indicators(market_data, regime_data)
            
            self._save_results({}, regime_data, llm_analysis)
            
            logger.info("LLM analysis completed successfully")
            return llm_analysis

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            raise

    def _save_results(self, combinations: Dict, regime_data: Dict, llm_analysis: Dict):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {
            'timestamp': timestamp,
            'combinations': combinations,
            'regime_data': regime_data,
            'llm_analysis': llm_analysis
        }
        
        output_path = os.path.join(self.data_dir, f'analysis_results_{timestamp}.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)