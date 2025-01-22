from .binance_data import BinanceDataDownloader

import pandas as pd
import os
from pathlib import Path
import logging
import json
from .data_processor import DataProcessor
from .market_analyzer import MarketAnalyzer
from .llm_analyzer import LLMAnalyzer
from .indicator_combinations import IndicatorCombinations
from .visualization import IndicatorVisualizer
from .ml_trader import MLTrader
import torch

logger = logging.getLogger(__name__)

class ResearchManager:
    def __init__(self):
        self.binance_downloader = BinanceDataDownloader()
        self.data_processor = DataProcessor()
        self.market_analyzer = MarketAnalyzer()
        self.llm_analyzer = LLMAnalyzer()
        self.indicator_combinations = IndicatorCombinations()
        self.visualizer = IndicatorVisualizer()
        self.ml_trader = MLTrader()
        self.data_dir = os.path.join(Path(__file__).parent.parent, 'data', 'historical')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    async def download_market_data(self):
        """Download historical market data"""
        try:
            print("\nDownloading historical data from Binance...")
            data = self.binance_downloader.compile_historical_data(months_back=6)

            if data:
                filenames = self.binance_downloader.save_data(data)
                print("\nData saved to:")
                for filename in filenames:
                    print(filename)

            return data
        except Exception as e:
            logger.error(f"Error downloading market data: {e}")
            raise

    def get_available_pairs(self) -> list[str]:
        pairs = []
        for item in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, item)):
                pairs.append(item)
        return pairs

    async def run_indicator_analysis(self):
        try:
            pairs = self.get_available_pairs()
            if not pairs:
                logger.error("No trading pair data found")
                return

            print(f"\nAnalyzing {len(pairs)} trading pairs...")
            all_results = {}

            for pair in pairs:
                print(f"\nAnalyzing {pair}...")
                data_path = os.path.join(self.data_dir, pair, f"{pair}_1m_data.csv")
                
                if not os.path.exists(data_path):
                    logger.warning(f"No 1m data found for {pair}, skipping...")
                    continue

                df = self.data_processor.load_market_data(data_path)
                if df is None:
                    logger.error(f"Failed to load market data for {pair}")
                    continue

                df = self.data_processor.add_technical_features(df)
                
                combination_results = await self.indicator_combinations.test_combinations(df)
                regime_data = self.market_analyzer.analyze_market_regime(df)
                key_levels = self.market_analyzer.find_key_levels(df)

                market_data = {
                    'symbol': pair,
                    'price': float(df['close'].iloc[-1]),
                    'change24h': float(df['close'].pct_change(1440).iloc[-1] * 100)
                }

                llm_analysis = await self.llm_analyzer.analyze_indicators(market_data, regime_data)
                
                all_results[pair] = {
                    'combinations': combination_results,
                    'regime_data': regime_data,
                    'key_levels': key_levels,
                    'llm_analysis': llm_analysis
                }

                print(f"\nFound {len(combination_results)} qualifying combinations for {pair}")
                print("\nTop performing combinations and their parameters:")
                for combo_name, data in sorted(combination_results.items(), 
                                           key=lambda x: x[1]['success_rate'], 
                                           reverse=True)[:10]:
                    print(f"\n{combo_name}:")
                    print(f"  Success Rate: {data['success_rate']:.2f}")
                    print("  Parameters:")
                    for indicator, params in data['parameters'].items():
                        print(f"    {indicator}: {params}")

            self._save_results(all_results)

        except Exception as e:
            logger.error(f"Error in indicator analysis: {e}")
            raise

    def _save_results(self, results):
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(self.data_dir, 'analysis_results')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'analysis_results_{timestamp}.json')
        summary_path = os.path.join(output_dir, f'summary_{timestamp}.txt')
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(summary_path, 'w') as f:
            for pair, data in results.items():
                f.write(f"\n{'='*50}\n")
                f.write(f"Results for {pair}\n")
                f.write(f"{'='*50}\n\n")
                
                f.write("Top Performing Combinations:\n")
                for combo_name, combo_data in sorted(data['combinations'].items(), 
                                                   key=lambda x: x[1]['success_rate'], 
                                                   reverse=True)[:10]:
                    f.write(f"\n{combo_name}:\n")
                    f.write(f"  Success Rate: {combo_data['success_rate']:.2f}\n")
                    f.write("  Parameters:\n")
                    for indicator, params in combo_data['parameters'].items():
                        f.write(f"    {indicator}: {params}\n")
                
                f.write("\nMarket Regime:\n")
                for key, value in data['regime_data'].items():
                    f.write(f"  {key}: {value}\n")
                
                f.write("\nKey Levels:\n")
                for key, value in data['key_levels'].items():
                    f.write(f"  {key}: {value}\n")
        
        print(f"\nFull results saved to: {output_path}")
        print(f"Summary saved to: {summary_path}")

    async def run_ml_analysis(self):
        """Train and evaluate ML models"""
        try:
            pairs = self.get_available_pairs()
            if not pairs:
                logger.error("No trading pair data found")
                return {}

            print(f"\nTraining on {len(pairs)} trading pairs...")
            all_results = {}

            # Train on all pairs together
            print("\nTraining combined model...")
            combined_results = await self.ml_trader.train(epochs=30, batch_size=64)
            if combined_results and isinstance(combined_results, dict):
                print(f"\nCombined Model Results:")
                final_balance = combined_results.get('final_balance', 10000.0)
                returns = combined_results.get('return', 0.0)
                trades = combined_results.get('trades', [])
                print(f"Final Balance: ${final_balance:.2f}")
                print(f"Return: {returns:.2f}%")
                print(f"Total Trades: {len(trades)}")

            # Train individual models for each pair
            for pair in pairs:
                print(f"\nTraining model for {pair}...")
                results = await self.ml_trader.train(epochs=30, batch_size=64, pair=pair)
                
                if results:
                    all_results[pair] = results
                    print(f"\nResults for {pair}:")
                    if isinstance(results, dict):
                        final_balance = results.get('final_balance', 0)
                        returns = results.get('return', 0)
                        trades = results.get('trades', [])
                        print(f"Final Balance: ${final_balance:.2f}")
                        print(f"Return: {returns:.2f}%")
                        print(f"Total Trades: {len(trades)}")
                    else:
                        print("Training completed successfully")

            return all_results

        except Exception as e:
            logger.error(f"Error in ML analysis: {e}")
            return {}