from typing import Dict

import pandas as pd


def calculate_absolute_strength_histogram(data: pd.DataFrame, period: int = 14) -> pd.Series:
    momentum = data['close'].diff(period)
    sma_momentum = momentum.rolling(window=period).mean()
    return momentum - sma_momentum


def calculate_tonux_ema_scalper(data: pd.DataFrame) -> Dict[str, pd.Series]:
    ema3 = data['close'].ewm(span=3, adjust=False).mean()
    ema5 = data['close'].ewm(span=5, adjust=False).mean()
    ema8 = data['close'].ewm(span=8, adjust=False).mean()
    ema13 = data['close'].ewm(span=13, adjust=False).mean()

    signal = pd.Series(0, index=data.index)
    signal[(ema3 > ema5) & (ema5 > ema8) & (ema8 > ema13)] = 1  # Bullish
    signal[(ema3 < ema5) & (ema5 < ema8) & (ema8 < ema13)] = -1  # Bearish

    return {
        'ema3': ema3,
        'ema5': ema5,
        'ema8': ema8,
        'ema13': ema13,
        'signal': signal
    }


def calculate_coral_trend(data: pd.DataFrame, atr_period: int = 14, coral_period: int = 21) -> Dict[str, pd.Series]:
    # Calculate ATR
    tr = pd.concat([data['high'] - data['low'],
                    abs(data['high'] - data['close'].shift()),
                    abs(data['low'] - data['close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()

    coral_ma = data['close'].ewm(span=coral_period, adjust=False).mean()
    coral_std = data['close'].rolling(window=coral_period).std()

    upper_band = coral_ma + (coral_std * 2)
    lower_band = coral_ma - (coral_std * 2)

    trend = pd.Series(0, index=data.index)
    trend[data['close'] > upper_band] = 1
    trend[data['close'] < lower_band] = -1

    return {
        'coral_ma': coral_ma,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'trend': trend,
        'atr': atr
    }


def calculate_ichimoku_cloud(data: pd.DataFrame) -> Dict[str, pd.Series]:
    tenkan_sen = (data['high'].rolling(window=9).max() + data['low'].rolling(window=9).min()) / 2
    kijun_sen = (data['high'].rolling(window=26).max() + data['low'].rolling(window=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)
    chikou_span = data['close'].shift(-26)

    cloud_color = pd.Series(0, index=data.index)
    cloud_color[senkou_span_a > senkou_span_b] = 1
    cloud_color[senkou_span_a < senkou_span_b] = -1

    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span,
        'cloud_color': cloud_color
    }


def calculate_squeeze_index(data: pd.DataFrame, bb_period: int = 20, kc_period: int = 20,
                            bb_std: float = 2.0, kc_multiplier: float = 1.5) -> Dict[str, pd.Series]:
    bb_ma = data['close'].rolling(window=bb_period).mean()
    bb_std = data['close'].rolling(window=bb_period).std()
    bb_upper = bb_ma + (bb_std * bb_std)
    bb_lower = bb_ma - (bb_std * bb_std)

    kc_ma = data['close'].rolling(window=kc_period).mean()
    tr = pd.concat([data['high'] - data['low'],
                    abs(data['high'] - data['close'].shift()),
                    abs(data['low'] - data['close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=kc_period).mean()
    kc_upper = kc_ma + (atr * kc_multiplier)
    kc_lower = kc_ma - (atr * kc_multiplier)

    squeeze_on = pd.Series(0, index=data.index)
    squeeze_on[(bb_upper <= kc_upper) & (bb_lower >= kc_lower)] = 1
    mom = data['close'] - (
                (data['high'].rolling(window=bb_period).max() + data['low'].rolling(window=bb_period).min()) / 2)

    return {
        'squeeze_on': squeeze_on,
        'momentum': mom,
        'bb_ma': bb_ma,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'kc_upper': kc_upper,
        'kc_lower': kc_lower
    }


def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price (VWAP)"""
    # Calculate price * volume
    pv = (data['high'] + data['low'] + data['close']) / 3 * data['volume']
    # Calculate cumulative values
    cumulative_pv = pv.cumsum()
    cumulative_volume = data['volume'].cumsum()
    # Calculate VWAP
    vwap = cumulative_pv / cumulative_volume
    return vwap