import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class Indicators:
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> Dict:
        """Calculate all available indicators"""
        try:
            return {
                'rsi': Indicators._calc_rsi(df),
                'macd': Indicators._calc_macd(df),
                'bb': Indicators._calc_bb(df),
                'stoch': Indicators._calc_stoch(df),
                'atr': Indicators._calc_atr(df),
                'trend': Indicators._calc_trend(df),
                'volume': Indicators._calc_volume(df)
            }
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    @staticmethod
    def _calc_rsi(df: pd.DataFrame, period=14) -> Dict:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return {
            'value': float(rsi.iloc[-1]),
            'signal': 'oversold' if rsi.iloc[-1] < 30 else 'overbought' if rsi.iloc[-1] > 70 else 'neutral',
            'strength': abs(50 - rsi.iloc[-1]) / 50
        }

    @staticmethod
    def _calc_macd(df: pd.DataFrame) -> Dict:
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': float(macd.iloc[-1]),
            'signal': float(signal.iloc[-1]),
            'histogram': float(histogram.iloc[-1]),
            'trend': 'bullish' if macd.iloc[-1] > signal.iloc[-1] else 'bearish'
        }

    @staticmethod
    def _calc_bb(df: pd.DataFrame, period=20) -> Dict:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        current_price = df['close'].iloc[-1]
        
        return {
            'upper': float(upper.iloc[-1]),
            'middle': float(sma.iloc[-1]),
            'lower': float(lower.iloc[-1]),
            'width': float((upper.iloc[-1] - lower.iloc[-1]) / sma.iloc[-1]),
            'position': float((current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]))
        }

    @staticmethod
    def _calc_stoch(df: pd.DataFrame, period=14) -> Dict:
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=3).mean()
        
        return {
            'k': float(k.iloc[-1]),
            'd': float(d.iloc[-1]),
            'signal': 'oversold' if k.iloc[-1] < 20 else 'overbought' if k.iloc[-1] > 80 else 'neutral'
        }

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period=14) -> Dict:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return {
            'value': float(atr.iloc[-1]),
            'percent': float(atr.iloc[-1] / df['close'].iloc[-1] * 100)
        }

    @staticmethod
    def _calc_trend(df: pd.DataFrame) -> Dict:
        ema20 = df['close'].ewm(span=20, adjust=False).mean()
        ema50 = df['close'].ewm(span=50, adjust=False).mean()
        sma200 = df['close'].rolling(window=200).mean()
        
        price = df['close'].iloc[-1]
        trend_strength = abs(ema20.iloc[-1] - sma200.iloc[-1]) / sma200.iloc[-1]
        
        return {
            'ema20': float(ema20.iloc[-1]),
            'ema50': float(ema50.iloc[-1]),
            'sma200': float(sma200.iloc[-1]),
            'direction': 'up' if ema20.iloc[-1] > ema50.iloc[-1] else 'down',
            'strength': float(trend_strength)
        }

    @staticmethod
    def _calc_volume(df: pd.DataFrame) -> Dict:
        vol_sma = df['volume'].rolling(window=20).mean()
        vol_ratio = df['volume'] / vol_sma
        
        # On Balance Volume
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        
        return {
            'current_ratio': float(vol_ratio.iloc[-1]),
            'obv': float(obv.iloc[-1]),
            'obv_trend': 'up' if obv.iloc[-1] > obv.iloc[-5] else 'down',
            'volume_trend': 'high' if vol_ratio.iloc[-1] > 1.5 else 'low' if vol_ratio.iloc[-1] < 0.5 else 'normal'
        }