import argparse
from typing import Optional
import logging
from . import resrch_manage
from .data_processor import DataProcessor
from .llm_analyzer import LLMAnalyzer
from .live_trader import LiveTrader
from .paper_trader import PaperTrader

class CLI:
    def __init__(self):
        logging.info("Initializing CLI...")
        self.research_manager = resrch_manage.ResearchManager()
        logging.info("ResearchManager initialized.")
        self.data_processor = DataProcessor()
        self.llm_analyzer = LLMAnalyzer()
        self.automated_trader = LiveTrader()
        self.paper_trader = PaperTrader()

    async def run(self, option: Optional[int] = None):
        if option is None:
            option = self._get_user_choice()

        try:
            if option == 1:
                print("\nStarting Automated Trading...")
                await self.automated_trader.run_trading_loop()
            elif option == 2:
                await self.research_manager.run_indicator_analysis()
            elif option == 3:
                print("\nStarting ML Analysis...")
                await self.research_manager.run_ml_analysis()
            elif option == 4:
                logging.info("Starting Data Download...")
                await self.research_manager.download_market_data()
                logging.info("Data Download completed.")
            elif option == 5:
                print("\nStarting LLM Market Review...")
                await self.research_manager.llm_market_review()
            elif option == 6:
                print("\nInitializing Paper Trading...")
                try:
                    await self.paper_trader.run_trading_loop()
                except ValueError as e:
                    print(f"\nError: {e}")
                    print("Please check your config.json file contains valid Kraken API credentials.")
                except Exception as e:
                    print(f"\nUnexpected error in paper trading: {e}")
                    print("Check logs for more details.")
            elif option == 7:
                print("Exiting...")
                return
            else:
                print("Invalid option selected")
        except Exception as e:
            logging.error(f"Error running selected option: {e}")
            print(f"\nError: {e}")

    def _get_user_choice(self) -> int:
        print("\nCrypto Analysis Tool")
        print("1. Start Automated Trading")
        print("2. Test Indicators")
        print("3. Train ML Models")
        print("4. Download Market Data")
        print("5. LLM Market Review")
        print("6. Start Paper Trading")
        print("7. Exit")

        while True:
            try:
                choice = int(input("\nSelect an option (1-7): "))
                if 1 <= choice <= 7:
                    return choice
                print("Please enter a number between 1 and 7")
            except ValueError:
                print("Please enter a valid number")