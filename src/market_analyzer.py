import pandas as pd
import numpy as np
from typing import Dict

from src.indicators.indicator_combinations import logger


class MarketAnalyzer:
    def analyze_regime(self, all_timeframes_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze market regime using volatility and trend indicators across multiple timeframes.

        Args:
            all_timeframes_data (Dict[str, pd.DataFrame]): OHLCV data per timeframe.

        Returns:
            Dict: A dictionary containing the market regime information.
        """
        try:
            # Select the primary timeframe for regime analysis
            primary_tf = "15m"  # You can choose the most relevant timeframe
            df = all_timeframes_data.get(primary_tf)

            if df is None or df.empty:
                logger.warning(f"No data available for primary timeframe: {primary_tf}")
                return {
                    'volatility': None,
                    'trend_strength': None,
                    'momentum': None,
                    'regime': 'unknown',
                    'sma50_200_cross': None
                }

            # Calculate volatility
            returns = df['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)

            # Calculate trend strength
            sma_50 = df['close'].rolling(window=50).mean()
            sma_200 = df['close'].rolling(window=200).mean()
            trend_strength = (df['close'] - sma_200) / sma_200

            # Calculate momentum
            momentum = df['close'].pct_change(20)

            # Determine market regime
            is_high_vol = volatility > 0.5
            is_trending = abs(trend_strength.iloc[-1]) > 0.1
            is_momentum = abs(momentum.iloc[-1]) > 0.1

            regime = 'neutral'
            if is_high_vol:
                regime = 'high_volatility'
                if is_trending:
                    regime = 'trending_volatile'
            elif is_trending:
                regime = 'trending'
            elif is_momentum:
                regime = 'momentum'

            sma50_200_cross = float(sma_50.iloc[-1] - sma_200.iloc[-1]) if not sma_50.empty and not sma_200.empty else None

            return {
                'volatility': float(volatility),
                'trend_strength': float(trend_strength.iloc[-1]),
                'momentum': float(momentum.iloc[-1]),
                'regime': regime,
                'sma50_200_cross': sma50_200_cross
            }

        except Exception as e:
            logger.error(f"Error in analyze_regime: {e}", exc_info=True)
            return {
                'volatility': None,
                'trend_strength': None,
                'momentum': None,
                'regime': 'unknown',
                'sma50_200_cross': None
            }

    def get_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Calculate support and resistance levels.

        Args:
            df (pd.DataFrame): OHLCV data.
            window (int, optional): Rolling window size. Defaults to 20.

        Returns:
            Dict: Support and resistance levels along with price relations.
        """
        try:
            highs = df['high'].rolling(window=window).max()
            lows = df['low'].rolling(window=window).min()

            current_price = df['close'].iloc[-1]
            support = lows.iloc[-1]
            resistance = highs.iloc[-1]

            return {
                'support': float(support),
                'resistance': float(resistance),
                'price_to_support': float((current_price - support) / support) if support != 0 else None,
                'price_to_resistance': float((resistance - current_price) / current_price) if current_price != 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_support_resistance: {e}", exc_info=True)
            return {
                'support': None,
                'resistance': None,
                'price_to_support': None,
                'price_to_resistance': None
            }