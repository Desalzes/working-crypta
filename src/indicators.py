import pandas as pd
import numpy as np
from typing import Dict
import logging
from . import ind_funcs

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
                'volume': Indicators._calc_volume(df),
                'abs_strength': Indicators._calc_abs_strength(df),
                'tonux_scalper': Indicators._calc_tonux_scalper(df),
                'coral_trend': Indicators._calc_coral_trend(df),
                'ichimoku': Indicators._calc_ichimoku(df),
                'squeeze': Indicators._calc_squeeze(df)
            }
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    @staticmethod
    def _calc_rsi(df: pd.DataFrame, period=14) -> Dict:
        """Calculate RSI indicator"""
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
        """Calculate MACD indicator"""
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
        """Calculate Bollinger Bands"""
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
        """Calculate Stochastic Oscillator"""
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
        """Calculate Average True Range"""
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
        """Calculate trend indicators"""
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
        """Calculate volume indicators"""
        vol_sma = df['volume'].rolling(window=20).mean()
        vol_ratio = df['volume'] / vol_sma

        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()

        return {
            'current_ratio': float(vol_ratio.iloc[-1]),
            'obv': float(obv.iloc[-1]),
            'obv_trend': 'up' if obv.iloc[-1] > obv.iloc[-5] else 'down',
            'volume_trend': 'high' if vol_ratio.iloc[-1] > 1.5 else 'low' if vol_ratio.iloc[-1] < 0.5 else 'normal'
        }

    @staticmethod
    def _calc_abs_strength(df: pd.DataFrame) -> Dict:
        """Calculate Absolute Strength Histogram"""
        histogram = ind_funcs.calculate_absolute_strength_histogram(df)
        return {
            'value': float(histogram.iloc[-1]),
            'trend': 'bullish' if histogram.iloc[-1] > 0 else 'bearish',
            'strength': abs(float(histogram.iloc[-1]))
        }

    @staticmethod
    def _calc_tonux_scalper(df: pd.DataFrame) -> Dict:
        """Calculate Tonux EMA Scalper"""
        result = ind_funcs.calculate_tonux_ema_scalper(df)
        return {
            'signal': 'buy' if result['signal'].iloc[-1] > 0 else 'sell' if result['signal'].iloc[
                                                                                -1] < 0 else 'neutral',
            'ema3': float(result['ema3'].iloc[-1]),
            'ema5': float(result['ema5'].iloc[-1]),
            'ema8': float(result['ema8'].iloc[-1]),
            'ema13': float(result['ema13'].iloc[-1])
        }

    @staticmethod
    def _calc_coral_trend(df: pd.DataFrame) -> Dict:
        """Calculate Coral Trend Indicator"""
        result = ind_funcs.calculate_coral_trend(df)
        return {
            'trend': 'bullish' if result['trend'].iloc[-1] > 0 else 'bearish' if result['trend'].iloc[
                                                                                     -1] < 0 else 'neutral',
            'ma': float(result['coral_ma'].iloc[-1]),
            'upper': float(result['upper_band'].iloc[-1]),
            'lower': float(result['lower_band'].iloc[-1]),
            'volatility': float(result['atr'].iloc[-1])
        }

    @staticmethod
    def _calc_ichimoku(df: pd.DataFrame) -> Dict:
        """Calculate Ichimoku Cloud"""
        result = ind_funcs.calculate_ichimoku_cloud(df)
        cloud_status = 'neutral'
        if result['cloud_color'].iloc[-1] > 0:
            cloud_status = 'bullish'
        elif result['cloud_color'].iloc[-1] < 0:
            cloud_status = 'bearish'

        return {
            'cloud_status': cloud_status,
            'tenkan': float(result['tenkan_sen'].iloc[-1]),
            'kijun': float(result['kijun_sen'].iloc[-1]),
            'span_a': float(result['senkou_span_a'].iloc[-1]),
            'span_b': float(result['senkou_span_b'].iloc[-1]),
            'chikou': float(result['chikou_span'].iloc[-26]) if len(df) > 26 else None
        }

    @staticmethod
    def _calc_squeeze(df: pd.DataFrame) -> Dict:
        """Calculate TTM Squeeze Index"""
        result = ind_funcs.calculate_squeeze_index(df)
        return {
            'is_squeezed': bool(result['squeeze_on'].iloc[-1]),
            'momentum': float(result['momentum'].iloc[-1]),
            'momentum_direction': 'up' if result['momentum'].iloc[-1] > result['momentum'].iloc[-2] else 'down',
            'bb_width': float(result['bb_upper'].iloc[-1] - result['bb_lower'].iloc[-1]),
            'kc_width': float(result['kc_upper'].iloc[-1] - result['kc_lower'].iloc[-1])
        }

    def calculate_all_timeframes(self, all_timeframes_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Calculate indicators for each timeframe.

        Args:
            all_timeframes_data (Dict[str, pd.DataFrame]): OHLCV data per timeframe.

        Returns:
            Dict[str, Dict]: Indicators per timeframe.
        """
        indicators_per_timeframe = {}
        for tf, df in all_timeframes_data.items():
            if df.empty:
                logger.warning(f"No data for timeframe {tf}")
                continue
            indicators_per_timeframe[tf] = self.calculate_all(df)
        return indicators_per_timeframe