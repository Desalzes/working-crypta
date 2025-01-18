import argparse
from typing import Optional
import logging
from .research_manager import ResearchManager
from .data_processor import DataProcessor
from .llm_analyzer import LLMAnalyzer
from .automated_trader import AutomatedTrader

class CLI:
    def __init__(self):
        self.research_manager = ResearchManager()
        self.data_processor = DataProcessor()
        self.llm_analyzer = LLMAnalyzer()
        self.automated_trader = AutomatedTrader()

    async def clean_historical_data(self):
        print("\nCleaning historical data...")
        data_dir = 'C:/Users/desal/anthropicFun/research/data/historical'
        results = self.data_processor.remove_duplicates_and_save(data_dir)
        for filename, duplicates_removed in results.items():
            print(f"Removed {duplicates_removed} duplicates from {filename}")

    async def run(self, option: Optional[int] = None):
        if option is None:
            option = self._get_user_choice()

        try:
            if option == 1:
                await self.research_manager.run_indicator_analysis()
            elif option == 2:
                print("\nStarting ML Analysis...")
                await self.research_manager.run_ml_analysis()
            elif option == 3:
                print("\nStarting Data Download...")
                await self.research_manager.download_market_data()
            elif option == 4:
                print("\nStarting LLM Market Review...")
                await self.research_manager.llm_market_review()
            elif option == 5:
                print("\nStarting Automated Trading...")
                await self.automated_trader.run_trading_loop()
            elif option == 6:
                print("Exiting...")
                return
            else:
                print("Invalid option selected")
        except Exception as e:
            logging.error(f"Error running selected option: {e}")

    def _get_user_choice(self) -> int:
        print("\nCrypto Analysis Tool")
        print("1. Test Indicators")
        print("2. Train ML Models")
        print("3. Download Market Data")
        print("4. LLM Market Review")
        print("5. Start Automated Trading")
        print("6. Exit")
        
        while True:
            try:
                choice = int(input("\nSelect an option (1-6): "))
                if 1 <= choice <= 6:
                    return choice
                print("Please enter a number between 1 and 6")
            except ValueError:
                print("Please enter a valid number")