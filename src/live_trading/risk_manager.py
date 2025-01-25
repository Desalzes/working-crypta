import logging
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, max_allocation_per_pair: float = 0.2):
        self.max_allocation_per_pair = max_allocation_per_pair
        self.position_limits = {}

    def analyze_portfolio_diversity(self, portfolio_state: Dict, trading_pairs: List[str]) -> Dict:
        try:
            total_portfolio_value = portfolio_state['usd_balance']
            pair_values = {}

            for pair, position in portfolio_state['positions'].items():
                if position['amount'] > 0:
                    pair_value = position['amount'] * position['avg_price']
                    total_portfolio_value += pair_value
                    pair_values[pair] = pair_value

            allocation_percentages = {
                pair: value / total_portfolio_value * 100
                for pair, value in pair_values.items()
            }

            recommendations = []
            for pair, percentage in allocation_percentages.items():
                if percentage > self.max_allocation_per_pair * 100:
                    recommendations.append({
                        'pair': pair,
                        'current_allocation': percentage,
                        'action': 'REDUCE',
                        'reason': 'Overexposure'
                    })

            underrepresented = [
                pair for pair in trading_pairs
                if pair not in allocation_percentages
            ]

            if underrepresented:
                recommendations.append({
                    'pairs': underrepresented,
                    'action': 'EXPLORE',
                    'reason': 'No current position'
                })

            return {
                'total_portfolio_value': total_portfolio_value,
                'current_allocations': allocation_percentages,
                'recommendations': recommendations
            }

        except Exception as e:
            logger.error(f"Error analyzing portfolio diversity: {e}")
            return {}

    def suggest_trades_for_diversity(self, diversity_analysis: Dict) -> List[Dict]:
        try:
            if not diversity_analysis or not diversity_analysis.get('recommendations'):
                return []

            trade_suggestions = []
            for recommendation in diversity_analysis['recommendations']:
                if recommendation['action'] == 'REDUCE':
                    trade_suggestions.append({
                        'pair': recommendation['pair'],
                        'action': 'SELL',
                        'reason': 'Portfolio rebalancing'
                    })
                if recommendation['action'] == 'EXPLORE':
                    for pair in recommendation.get('pairs', []):
                        trade_suggestions.append({
                            'pair': pair,
                            'action': 'EXPLORE_BUY',
                            'reason': 'Increase portfolio diversity'
                        })
            return trade_suggestions
        except Exception as e:
            logger.error(f"Error generating diversity trade suggestions: {e}")
            return []

    def validate_trade(self, pair: str, action: str, size: float, portfolio_state: Dict) -> bool:
        try:
            # Check basic trade validity
            if action not in ['BUY', 'SELL']:
                return False

            if size <= 0:
                return False

            position = portfolio_state['positions'].get(pair, {})
            total_value = portfolio_state['total_value']

            # For buys, check allocation limits
            if action == 'BUY':
                proposed_value = size
                current_allocation = position.get('amount', 0) * position.get('avg_price', 0)
                new_allocation = (current_allocation + proposed_value) / total_value

                if new_allocation > self.max_allocation_per_pair:
                    logger.warning(f"Trade would exceed max allocation for {pair}")
                    return False

            # For sells, verify position exists
            elif action == 'SELL':
                if not position or position.get('amount', 0) < size:
                    logger.warning(f"Insufficient position for {pair} sell")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False

    def should_close_position(self, pair: str, current_price: float, position: Dict) -> bool:
        try:
            if not position or not position.get('amount', 0):
                return False

            stop_loss = position.get('stop_loss')
            take_profit = position.get('take_profit')

            if stop_loss and current_price <= float(stop_loss):
                return True

            if take_profit and current_price >= float(take_profit):
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return False