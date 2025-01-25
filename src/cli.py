from typing import Optional
import logging
import torch

from src.live_trader import LiveTrader
from src.live_trading.config_manager import ConfigManager
from src.live_trading.market_data_manager import MarketDataManager
from src.live_trading.analysis_engine import AnalysisEngine
from src.live_trading.execution_engine import ExecutionEngine
from src.live_trading.risk_manager import RiskManager
from src.live_trading.model_manager import ModelManager
from src.live_trading.trading_engine import TradingEngine
from src.binance_data import BinanceDataDownloader
from .data_processor import DataProcessor
from .market_analyzer import MarketAnalyzer
from .indicators.indicators import Indicators
from .portfolio_manager import PortfolioManager
from src.research_manager import ResearchManager
import ccxt

class CLI:
    def __init__(self, config_path: Optional[str] = None):
        logging.info("Initializing CLI...")

        # Initialize base components
        self.config_manager = ConfigManager()
        if config_path:
            self.config_manager.load_config(config_path)

        # Initialize exchange
        credentials = self.config_manager.get_api_credentials()
        self.exchange = ccxt.cryptocom({
            'apiKey': credentials['api_key'],
            'secret': credentials['api_secret'],
            'enableRateLimit': True,
        })
        self.exchange.set_sandbox_mode(False)

        # Initialize supporting components
        self.research_manager = ResearchManager()
        self.data_processor = DataProcessor()
        self.market_analyzer = MarketAnalyzer()
        self.indicators = Indicators()
        self.portfolio = PortfolioManager()
        self.binance_data_downloader = BinanceDataDownloader()

        # Initialize main components
        self.market_data = MarketDataManager(
            exchange=self.exchange,
            data_processor=self.data_processor,
            data_dir=self.config_manager.get_data_dir()
        )
        self.model_manager = ModelManager(self.config_manager.get_data_dir())

        self.analysis = AnalysisEngine(
            indicators=self.indicators,
            market_analyzer=self.market_analyzer,
            ollama_url=self.config_manager.get_ollama_url(),
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.execution = ExecutionEngine(
            exchange=self.exchange,
            portfolio_manager=self.portfolio,
            data_dir=self.config_manager.get_data_dir(),
            min_balance_threshold=self.config_manager.get_thresholds()['min_balance_threshold']
        )

        self.risk = RiskManager(
            max_allocation_per_pair=self.config_manager.get_thresholds()['max_allocation_per_pair']
        )

        # Initialize trading engine
        self.trading_engine = TradingEngine(
            config_manager=self.config_manager,
            market_data_manager=self.market_data,
            analysis_engine=self.analysis,
            execution_engine=self.execution,
            risk_manager=self.risk,
            model_manager=self.model_manager
        )

    async def run(self, option: Optional[int] = None):
        if option is None:
            option = self._get_user_choice()

        try:
            if option == 1:
                print("Starting Automated Trading...")
                await self.trading_engine.start()
            elif option == 2:
                await self.research_manager.run_indicator_analysis()
            elif option == 3:
                print("Starting ML Analysis...")
                await self.research_manager.run_ml_analysis()
            elif option == 4:
                print("Starting Binance Data Download...")
                binance_data = self.binance_data_downloader.compile_historical_data()
                if binance_data:
                    self.binance_data_downloader.save_data(binance_data)
            elif option == 5:
                print("Starting LLM Market Review...")
                await self.research_manager.llm_market_review()
            elif option == 6:
                print("Starting Paper Trading...")
                await self.trading_engine.start()
            elif option == 7:
                print("Exiting...")
                return
            else:
                print("Invalid option selected")
        except Exception as e:
            logging.error(f"Error running selected option: {e}")
            print(f"Error: {e}")

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