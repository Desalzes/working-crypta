import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self):
        """Initialize ConfigManager with default values"""
        self.trading_pairs: List[str] = []
        self.api_key: Optional[str] = None
        self.api_secret: Optional[str] = None
        self.max_allocation_per_pair: float = 0.2
        self.min_balance_threshold: float = 500.0
        self.data_dir: str = os.path.join(Path(__file__).parent.parent, "../data")
        self.ollama_url: str = "http://localhost:11434/api/generate"

        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)

    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from file"""
        if config_path is None:
            config_path = os.path.join(Path(__file__).parent, '../config.json')

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")

            # Extract trading pairs
            base_trading_pairs = config.get('trading_pairs', [])
            if not base_trading_pairs:
                logger.error("No trading pairs found in the configuration.")
                raise ValueError("Trading pairs list is empty in the config file.")

            # Convert to SYMBOL/USD format
            self.trading_pairs = [f"{symbol}/USD" for symbol in base_trading_pairs]

            # Load API credentials
            self.api_key = os.getenv('CRYPTO_API_KEY') or config.get('crypto_api_key')
            self.api_secret = os.getenv('CRYPTO_SECRET_KEY') or config.get('crypto_secret_key')

            if not self.api_key or not self.api_secret:
                logger.error("Crypto.com API credentials not found")
                raise ValueError("Missing Crypto.com API credentials.")

        except Exception as e:
            logger.error(f"Error loading config: {e}", exc_info=True)
            raise FileNotFoundError(f"Could not load configuration from {config_path}")

    def get_trading_pairs(self) -> List[str]:
        """Get list of configured trading pairs"""
        return self.trading_pairs

    def get_api_credentials(self) -> Dict[str, str]:
        """Get API credentials"""
        return {
            'api_key': self.api_key,
            'api_secret': self.api_secret
        }

    def get_thresholds(self) -> Dict[str, float]:
        """Get configured thresholds"""
        return {
            'max_allocation_per_pair': self.max_allocation_per_pair,
            'min_balance_threshold': self.min_balance_threshold
        }

    def get_data_dir(self) -> str:
        """Get data directory path"""
        return self.data_dir

    def get_ollama_url(self) -> str:
        """Get Ollama API URL"""
        return self.ollama_url

    def validate_config(self) -> bool:
        """Validate configuration"""
        if not self.trading_pairs:
            logger.error("No trading pairs configured")
            return False

        if not self.api_key or not self.api_secret:
            logger.error("Missing API credentials")
            return False

        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return False

        return True