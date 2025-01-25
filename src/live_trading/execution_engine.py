import os
import json
import logging
from typing import Dict, Optional
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class ExecutionEngine:
    def __init__(self, exchange, portfolio_manager, data_dir: str, min_balance_threshold: float = 500.0):
        self.exchange = exchange
        self.portfolio = portfolio_manager
        self.data_dir = data_dir
        self.min_balance_threshold = min_balance_threshold
        self.trade_history = []

    async def execute_trade(self, pair: str, trading_decision: Dict) -> bool:
        try:
            portfolio_state = self.portfolio.get_portfolio_summary()
            if portfolio_state['usd_balance'] < self.min_balance_threshold:
                logger.warning(f"Insufficient balance (${portfolio_state['usd_balance']:.2f}) to trade.")
                return False

            action = trading_decision['decision']['action']
            if action not in ["BUY", "SELL"]:
                return False

            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, pair)
            current_price = ticker['last']

            if action == "BUY":
                position_size = portfolio_state['usd_balance'] * float(
                    trading_decision['decision'].get('size', 0.1)
                )
                amount = position_size / current_price
                success = self.portfolio.execute_trade(
                    pair=pair,
                    action='BUY',
                    price=current_price,
                    size=position_size,
                    decision_data=trading_decision
                )
                if success:
                    logger.info(f"Executed BUY for {pair}: {amount} @ ${current_price}")
                    self.record_trade(pair, 'BUY', current_price, amount, 0, trading_decision)
                    return True

            elif action == "SELL":
                position = self.portfolio.get_position(pair)
                if position['amount'] > 0:
                    success = self.portfolio.execute_trade(
                        pair=pair,
                        action='SELL',
                        price=current_price,
                        size=position['amount'] * current_price,
                        decision_data=trading_decision
                    )
                    if success:
                        profit_loss = (current_price - position['avg_price']) * position['amount']
                        logger.info(f"Executed SELL for {pair}: {position['amount']} @ ${current_price}")
                        self.record_trade(pair, 'SELL', current_price, position['amount'], profit_loss, trading_decision)
                        return True

            return False

        except Exception as e:
            logger.error(f"Error executing trade for {pair}: {e}", exc_info=True)
            return False

    def record_trade(self, pair: str, action: str, price: float, size: float, profit_loss: float, decision_data: Dict):
        """Record trade details"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'type': action,
            'price': price,
            'size': size,
            'profit_loss': profit_loss,
            'reasoning': decision_data.get('reasoning', {})
        }
        self.trade_history.append(trade)
        self.save_trade_history()

    def save_trade_history(self):
        """Save trade history to file"""
        history_path = os.path.join(self.data_dir, 'trade_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.trade_history, f, indent=2)

    async def check_position_limits(self, pair: str, current_price: float, holdings: Dict) -> bool:
        """Check and handle stop-loss/take-profit limits"""
        stop_loss = holdings.get('stop_loss')
        take_profit = holdings.get('take_profit')

        if stop_loss and current_price <= float(stop_loss):
            await self.execute_trade(pair, {
                'decision': {'action': 'SELL', 'size': 1.0},
                'reasoning': {'technical_analysis': 'Stop loss triggered'}
            })
            return True

        if take_profit and current_price >= float(take_profit):
            await self.execute_trade(pair, {
                'decision': {'action': 'SELL', 'size': 1.0},
                'reasoning': {'technical_analysis': 'Take profit triggered'}
            })
            return True

        return False