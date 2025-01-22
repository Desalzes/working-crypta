import json
import os
import logging
from typing import Dict, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class PortfolioManager:
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.positions: Dict[str, float] = {}
        self.trade_history: List[Dict] = []
        self.crypto_holdings: Dict[str, Dict] = {}

        # Setup directories
        self.data_dir = os.path.join(Path(__file__).parent.parent, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        # Load initial state
        self._load_balance()

    def _load_balance(self) -> None:
        """Load balance from balance.json"""
        balance_path = os.path.join(self.data_dir, 'balance.json')
        if os.path.exists(balance_path):
            try:
                with open(balance_path, 'r') as f:
                    balance_data = json.load(f)
                    self.balance = balance_data.get('USD', 10000.00)
                    self.crypto_holdings = balance_data.get('crypto_holdings', {})
                    self.trade_history = balance_data.get('trade_history', [])
            except Exception as e:
                logger.error(f"Error loading balance: {e}")
                self.balance = 10000.00
                self.crypto_holdings = {}
                self.trade_history = []
        else:
            # Create initial balance file
            self._save_balance()

    def _save_balance(self) -> None:
        """Save current balance to balance.json"""
        balance_data = {
            'USD': self.balance,
            'crypto_holdings': self.crypto_holdings,
            'trade_history': self.trade_history,
            'last_updated': datetime.now().isoformat()
        }

        balance_path = os.path.join(self.data_dir, 'balance.json')
        with open(balance_path, 'w') as f:
            json.dump(balance_data, f, indent=2)

    def execute_trade(self, pair: str, action: str, price: float, size: float,
                      decision_data: Dict) -> bool:
        """Execute a paper trade and update portfolio state"""
        timestamp = datetime.now()

        try:
            if action == "BUY" and self.balance >= size:
                position_size = size / price
                self.balance -= size

                # Update crypto holdings
                if pair not in self.crypto_holdings:
                    self.crypto_holdings[pair] = {
                        'amount': 0,
                        'avg_price': 0,
                        'total_cost': 0
                    }

                # Calculate new average price
                current = self.crypto_holdings[pair]
                total_amount = current['amount'] + position_size
                total_cost = current['total_cost'] + size
                avg_price = total_cost / total_amount if total_amount > 0 else 0

                self.crypto_holdings[pair] = {
                    'amount': total_amount,
                    'avg_price': avg_price,
                    'total_cost': total_cost,
                    'last_update': timestamp.isoformat()
                }

                trade_type = "BUY"
                trade_profit_loss = None

            elif action == "SELL" and pair in self.crypto_holdings:
                current = self.crypto_holdings[pair]
                if current['amount'] > 0:
                    sell_amount = current['amount']  # Sell all holdings
                    trade_value = sell_amount * price
                    self.balance += trade_value

                    # Calculate profit/loss
                    profit_loss = trade_value - current['total_cost']

                    # Clear holdings for this pair
                    self.crypto_holdings[pair] = {
                        'amount': 0,
                        'avg_price': 0,
                        'total_cost': 0,
                        'last_update': timestamp.isoformat()
                    }

                    trade_type = "SELL"
                    size = trade_value
                    trade_profit_loss = profit_loss
                else:
                    return False
            else:
                return False

            # Record the trade
            trade = {
                'timestamp': timestamp.isoformat(),
                'pair': pair,
                'type': trade_type,
                'price': price,
                'size': size,
                'usd_balance': self.balance,
                'crypto_amount': self.crypto_holdings[pair]['amount'],
                'avg_price': self.crypto_holdings[pair]['avg_price'],
                'confidence': decision_data.get('confidence', 0),
                'reasoning': decision_data.get('reasoning', ''),
                'stop_loss': decision_data.get('stop_loss', None),
                'take_profit': decision_data.get('take_profit', None)
            }

            if trade_profit_loss is not None:
                trade['profit_loss'] = trade_profit_loss

            self.trade_history.append(trade)
            self._save_balance()

            return True

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        return {
            'usd_balance': self.balance,
            'crypto_holdings': self.crypto_holdings,
            'total_trades': len(self.trade_history),
            'last_trade': self.trade_history[-1] if self.trade_history else None
        }

    def get_position(self, pair: str) -> Dict:
        """Get current position for a trading pair"""
        if pair in self.crypto_holdings:
            return {
                'amount': self.crypto_holdings[pair]['amount'],
                'avg_price': self.crypto_holdings[pair]['avg_price'],
                'total_cost': self.crypto_holdings[pair]['total_cost']
            }
        return {'amount': 0, 'avg_price': 0, 'total_cost': 0}


