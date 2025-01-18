import optuna
import numpy as np
import pandas as pd
from typing import Dict
import logging
from concurrent.futures import ProcessPoolExecutor
import torch

logger = logging.getLogger(__name__)

class IndicatorOptimizer:
    def __init__(self, n_trials=100, n_jobs=-1):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def optimize(self, df: pd.DataFrame, indicator_name: str) -> Dict:
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            params = self._get_parameter_space(trial, indicator_name)
            signals = self._calculate_signals(df, indicator_name, params)
            return self._calculate_profit(df, signals)
            
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        return study.best_params
        
    def _get_parameter_space(self, trial: optuna.Trial, indicator: str) -> Dict:
        spaces = {
            'rsi': {
                'period': trial.suggest_int('period', 9, 24),
                'overbought': trial.suggest_int('overbought', 65, 85, 5),
                'oversold': trial.suggest_int('oversold', 15, 35, 5)
            },
            'macd': {
                'fast_period': trial.suggest_int('fast_period', 8, 20),
                'slow_period': trial.suggest_int('slow_period', 20, 34),
                'signal_period': trial.suggest_int('signal_period', 7, 14)
            },
            'bb': {
                'period': trial.suggest_int('period', 15, 30),
                'std_dev': trial.suggest_float('std_dev', 1.5, 3.0)
            },
            'stoch': {
                'period': trial.suggest_int('period', 9, 24),
                'd_period': trial.suggest_int('d_period', 3, 7)
            },
            'adx': {
                'period': trial.suggest_int('period', 10, 24),
                'threshold_high': trial.suggest_int('threshold_high', 22, 35),
                'threshold_low': trial.suggest_int('threshold_low', 15, 28)
            },
            'mfi': {
                'period': trial.suggest_int('period', 10, 24),
                'overbought': trial.suggest_int('overbought', 75, 90, 5),
                'oversold': trial.suggest_int('oversold', 10, 25, 5)
            }
        }
        return spaces[indicator]
        
    def _calculate_signals(self, df: pd.DataFrame, indicator: str, params: Dict) -> np.ndarray:
        indicator_funcs = {
            'rsi': self._rsi_signal,
            'macd': self._macd_signal,
            'bb': self._bollinger_signal,
            'stoch': self._stoch_signal,
            'adx': self._adx_signal,
            'mfi': self._mfi_signal
        }
        return indicator_funcs[indicator](df, **params)
        
    def _calculate_profit(self, df: pd.DataFrame, signals: np.ndarray) -> float:
        position = 0
        balance = 10000
        prices = df['close'].values
        
        for i in range(1, len(signals)):
            if signals[i] == 1 and position <= 0:
                position = balance / prices[i]
            elif signals[i] == -1 and position >= 0:
                balance = position * prices[i]
                position = 0
                
        if position > 0:
            balance = position * prices[-1]
            
        return (balance - 10000) / 10000 * 100
        
    # Indicator calculation methods
    def _rsi_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=params['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=params['period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = np.zeros(len(df))
        signals[rsi > params['overbought']] = -1
        signals[rsi < params['oversold']] = 1
        return signals
        
    def _macd_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
        exp1 = df['close'].ewm(span=params['fast_period'], adjust=False).mean()
        exp2 = df['close'].ewm(span=params['slow_period'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=params['signal_period'], adjust=False).mean()
        
        signals = np.zeros(len(df))
        signals[macd > signal] = 1
        signals[macd < signal] = -1
        return signals
        
    def _bollinger_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
        sma = df['close'].rolling(window=params['period']).mean()
        std = df['close'].rolling(window=params['period']).std()
        upper = sma + (std * params['std_dev'])
        lower = sma - (std * params['std_dev'])
        
        signals = np.zeros(len(df))
        signals[df['close'] > upper] = -1
        signals[df['close'] < lower] = 1
        return signals
        
    def _stoch_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
        low_min = df['low'].rolling(window=params['period']).min()
        high_max = df['high'].rolling(window=params['period']).max()
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=params['d_period']).mean()
        
        signals = np.zeros(len(df))
        signals[k > d] = 1
        signals[k < d] = -1
        return signals
        
    def _adx_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        dx = 100 * tr.rolling(params['period']).mean()
        adx = dx.rolling(params['period']).mean()
        
        signals = np.zeros(len(df))
        signals[adx > params['threshold_high']] = 1
        signals[adx < params['threshold_low']] = -1
        return signals
        
    def _mfi_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(params['period']).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(params['period']).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        
        signals = np.zeros(len(df))
        signals[mfi > params['overbought']] = -1
        signals[mfi < params['oversold']] = 1
        return signals