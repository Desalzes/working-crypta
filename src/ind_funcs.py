import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler
import ta
from scipy import stats
import math


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[
    str, pd.Series]:
    """Calculate MACD indicator"""
    exp1 = data['close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return {
        'macd': macd,
        'signal': signal,
        'hist': hist
    }


def calculate_bollinger(data: pd.DataFrame, period: int = 20, std: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = data['close'].rolling(window=period).mean()
    rolling_std = data['close'].rolling(window=period).std()
    return {
        'upper': sma + (rolling_std * std),
        'middle': sma,
        'lower': sma - (rolling_std * std)
    }


def calculate_stochastic(data: pd.DataFrame, period: int = 14, k_period: int = 3, d_period: int = 3) -> Dict[
    str, pd.Series]:
    """Calculate Stochastic Oscillator"""
    low_min = data['low'].rolling(window=period).min()
    high_max = data['high'].rolling(window=period).max()
    k = 100 * ((data['close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return {
        'k': k,
        'd': d
    }


def calculate_adx(data: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
    """Calculate ADX indicator"""
    # Make sure all inputs are the same length
    df = data.copy()

    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate directional movement
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']

    # Calculate DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Convert to pandas series with same index as original data
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Smooth the TR and DM
    tr_smooth = tr.rolling(window=period).sum()
    plus_dm_smooth = plus_dm.rolling(window=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period).sum()

    # Calculate DI
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    # Fill NaN values with 0 to maintain length
    adx = adx.fillna(0)
    plus_di = plus_di.fillna(0)
    minus_di = minus_di.fillna(0)

    return {
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di
    }

def calculate_mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index"""
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()

    return 100 - (100 / (1 + positive_flow / negative_flow))


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_obv(data: pd.DataFrame, ema_period: int = 20) -> Dict[str, pd.Series]:
    """Calculate On Balance Volume"""
    obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
    obv_ema = obv.ewm(span=ema_period, adjust=False).mean()
    return {
        'obv': obv,
        'obv_ema': obv_ema
    }


def calculate_vwap(data: pd.DataFrame, period: int = None) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    v = data['volume']
    tp = (data['high'] + data['low'] + data['close']) / 3
    if period:
        return (tp * v).rolling(period).sum() / v.rolling(period).sum()
    return (tp * v).cumsum() / v.cumsum()


def calculate_trend_strength(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate trend strength using linear regression"""
    x = np.arange(len(data))
    slope, _, r_value, _, _ = stats.linregress(x, data['close'])
    return r_value ** 2


def calculate_volatility(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate price volatility"""
    log_returns = np.log(data['close'] / data['close'].shift(1))
    return log_returns.rolling(window=period).std() * np.sqrt(period)


def calculate_order_imbalance(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate order imbalance based on volume and price action"""
    buy_volume = data['volume'] * (data['close'] - data['low']) / (data['high'] - data['low'])
    sell_volume = data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'])
    return (buy_volume - sell_volume) / (buy_volume + sell_volume)


def calculate_support_resistance(data: pd.DataFrame, period: int = 20, threshold: float = 0.02) -> Dict[
    str, List[float]]:
    """Calculate support and resistance levels"""
    pivots = pd.concat([
        data['high'].rolling(window=period, center=True).max(),
        data['low'].rolling(window=period, center=True).min()
    ])
    levels = []

    for price in pivots:
        if not price:
            continue
        ranges = abs(pivots - price) < (price * threshold)
        if ranges.sum() >= 2:
            levels.append(price)

    levels = sorted(list(set([round(level, 4) for level in levels])))
    current_price = data['close'].iloc[-1]

    supports = [level for level in levels if level < current_price]
    resistances = [level for level in levels if level > current_price]

    return {
        'supports': supports[-3:] if supports else [],
        'resistances': resistances[:3] if resistances else []
    }


def calculate_fibonacci_levels(data: pd.DataFrame, period: int = 20) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels"""
    high = data['high'].rolling(window=period).max().iloc[-1]
    low = data['low'].rolling(window=period).min().iloc[-1]
    diff = high - low

    return {
        '0.0': low,
        '0.236': low + 0.236 * diff,
        '0.382': low + 0.382 * diff,
        '0.5': low + 0.5 * diff,
        '0.618': low + 0.618 * diff,
        '0.786': low + 0.786 * diff,
        '1.0': high
    }


def calculate_ichimoku(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculate Ichimoku Cloud indicators"""
    high_values = data['high']
    low_values = data['low']

    nine_period_high = high_values.rolling(window=9).max()
    nine_period_low = low_values.rolling(window=9).min()
    tenkan_sen = (nine_period_high + nine_period_low) / 2

    twenty_six_period_high = high_values.rolling(window=26).max()
    twenty_six_period_low = low_values.rolling(window=26).min()
    kijun_sen = (twenty_six_period_high + twenty_six_period_low) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    fifty_two_period_high = high_values.rolling(window=52).max()
    fifty_two_period_low = low_values.rolling(window=52).min()
    senkou_span_b = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)

    chikou_span = data['close'].shift(-26)

    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }


def normalize_indicator(indicator: pd.Series, method: str = 'zscore') -> pd.Series:
    """Normalize indicator values"""
    if method == 'zscore':
        return (indicator - indicator.mean()) / indicator.std()
    elif method == 'minmax':
        return (indicator - indicator.min()) / (indicator.max() - indicator.min())
    return indicator


def calculate_turning_points(data: pd.DataFrame, period: int = 20) -> Dict[str, List[int]]:
    """Identify potential turning points in the price"""
    highs = data['high'].rolling(window=period, center=True).max()
    lows = data['low'].rolling(window=period, center=True).min()

    pivot_highs = []
    pivot_lows = []

    for i in range(period, len(data) - period):
        if data['high'].iloc[i] == highs.iloc[i]:
            pivot_highs.append(i)
        if data['low'].iloc[i] == lows.iloc[i]:
            pivot_lows.append(i)

    return {
        'highs': pivot_highs,
        'lows': pivot_lows
    }