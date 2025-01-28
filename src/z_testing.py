import json
import logging
from pathlib import Path

from web3 import Web3
from web3.exceptions import BadFunctionCallOutput

# --------------------------- CONFIGURATION ---------------------------

CONFIG_PATH = Path(__file__).parent / 'config.json'

# USDC Contract Address on Ethereum Mainnet
USDC_CONTRACT_ADDRESS = '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606EB48'

# ERC-20 Token ABI (Only the `balanceOf` function is needed)
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    }
]

# --------------------------- LOGGING ---------------------------

def setup_logging():
    """Configure logging with a simple format."""
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more detailed logs
        format='%(asctime)s %(levelname)s:%(message)s'
    )

# --------------------------- FUNCTIONS ---------------------------

def load_config(config_path: Path) -> dict:
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}.")
        return config
    except FileNotFoundError:
        logging.critical(f"Configuration file not found at {config_path}.")
        raise
    except json.JSONDecodeError as e:
        logging.critical(f"Error parsing JSON configuration: {e}")
        raise
    except Exception as e:
        logging.critical(f"Unexpected error loading configuration: {e}")
        raise

def get_web3_connection(api_url: str) -> Web3:
    """Establish a Web3 connection using the provided API URL."""
    try:
        web3 = Web3(Web3.HTTPProvider(api_url))
        if not web3.isConnected():
            logging.critical(f"Failed to connect to Ethereum node at {api_url}.")
            raise ConnectionError(f"Cannot connect to Ethereum node at {api_url}.")
        logging.info(f"Connected to Ethereum node at {api_url}.")
        return web3
    except Exception as e:
        logging.critical(f"Error establishing Web3 connection: {e}")
        raise

def get_usdc_balance(web3: Web3, wallet_address: str) -> float:
    """Retrieve the USDC balance for the specified wallet address."""
    try:
        usdc_contract = web3.eth.contract(address=Web3.toChecksumAddress(USDC_CONTRACT_ADDRESS), abi=ERC20_ABI)
        balance = usdc_contract.functions.balanceOf(Web3.toChecksumAddress(wallet_address)).call()
        # USDC has 6 decimals
        usdc_balance = balance / (10 ** 6)
        return usdc_balance
    except BadFunctionCallOutput as e:
        logging.error(f"Bad function call output: {e}. Check if the contract address is correct.")
        raise
    except Exception as e:
        logging.error(f"Error fetching USDC balance: {e}")
        raise

def main():
    """Main function to execute the USDC balance retrieval."""
    setup_logging()
    logging.info("Starting USDC balance retrieval script.")

    # Load configuration
    try:
        config = load_config(CONFIG_PATH)
    except Exception as e:
        logging.critical(f"Configuration loading failed: {e}")
        return

    # Extract necessary configuration
    api_url = config.get('ethereum_api_url')  # You need to add this key in config.json
    wallet_address = config.get('hyperliquid_api_wallet_address')

    if not api_url or not wallet_address:
        logging.critical("Missing 'ethereum_api_url' or 'hyperliquid_api_wallet_address' in config.")
        return

    # Establish Web3 connection
    try:
        web3 = get_web3_connection(api_url)
    except Exception as e:
        logging.critical(f"Web3 connection failed: {e}")
        return

    # Fetch USDC balance
    try:
        usdc_balance = get_usdc_balance(web3, wallet_address)
        print(f"Your USDC balance: {usdc_balance} USDC")
        logging.info(f"USDC balance: {usdc_balance} USDC")
    except Exception as e:
        logging.error(f"An error occurred while fetching USDC balance: {e}")

# --------------------------- ENTRY POINT ---------------------------

if __name__ == "__main__":
    main()
