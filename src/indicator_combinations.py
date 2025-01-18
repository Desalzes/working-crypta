"""
Indicator Combinations Class - for testing different technical indicator combinations
"""
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from itertools import combinations, product
import torch
from tqdm import tqdm
import aiohttp
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from . import ind_funcs

logger = logging.getLogger(__name__)

class IndicatorCombinations:
   def __init__(self):
       self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       self.ollama_url = "http://localhost:11434/api/generate"
       self.batch_size = 32
       self.num_workers = os.cpu_count()
       torch.set_num_threads(self.num_workers)
       torch.cuda.empty_cache()

       # Parameter ranges for optimization
       self.parameter_ranges = {
           'rsi': {'period': list(range(9, 25)), 'overbought': list(range(65, 86, 5)),
                   'oversold': list(range(15, 36, 5))},
           'macd': {'fast_period': list(range(8, 21)), 'slow_period': list(range(20, 35)),
                    'signal_period': list(range(7, 15))},
           'bb': {'period': list(range(15, 31)), 'std_dev': [1.5, 2.0, 2.5, 3.0]},
           'ema': {'fast_span': list(range(8, 21)), 'slow_span': list(range(20, 35))},
           'stoch': {'period': list(range(9, 25)), 'd_period': list(range(3, 8))},
           'adx': {'period': list(range(10, 25)), 'threshold_high': list(range(22, 36)),
                   'threshold_low': list(range(15, 29))},
           'mfi': {'period': list(range(10, 25)), 'overbought': list(range(75, 91, 5)),
                   'oversold': list(range(10, 26, 5))},
           'obv': {'ema_span': list(range(15, 31))},
           'atr': {'period': list(range(10, 31))},
           'trend': {'ema20_period': list(range(15, 25)), 'ema50_period': list(range(45, 55)),
                     'sma200_period': list(range(180, 220))},
           'vwap': {'period': list(range(10, 31))},
           'order_imbalance': {'depth_levels': [5, 10, 20], 'threshold': [1.5, 2.0, 2.5, 3.0]},
           'bid_ask_spread': {'ma_period': list(range(10, 31))},
           'market_depth': {'levels': [5, 10, 20], 'threshold': [1.5, 2.0, 2.5, 3.0]},
           'order_flow': {'window': list(range(5, 21))}
       }

       self.base_indicators = {
           'rsi': self._rsi_signal,
           'macd': self._macd_signal,
           'bb': self._bollinger_signal,
           'ema': self._ema_signal,
           'stoch': self._stoch_signal,
           'adx': self._adx_signal,
           'mfi': self._mfi_signal,
           'obv': self._obv_signal,
           'atr': self._atr_signal,
           'trend': self._trend_signal,
           'vwap': self._vwap_signal,
           'order_imbalance': self._order_imbalance_signal,
           'bid_ask_spread': self._bid_ask_spread_signal,
           'market_depth': self._market_depth_signal,
           'order_flow': self._order_flow_signal
       }

       self.threshold = 0.40  # 40% success rate for pairs
       self.triple_threshold = 0.45  # 45% success rate for triples

       if torch.cuda.is_available():
           logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

   def _get_default_params(self, indicator: str) -> Dict[str, Any]:
       defaults = {
           'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
           'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
           'bb': {'period': 20, 'std_dev': 2.0},
           'ema': {'fast_span': 12, 'slow_span': 26},
           'stoch': {'period': 14, 'd_period': 3},
           'adx': {'period': 14, 'threshold_high': 25, 'threshold_low': 20},
           'mfi': {'period': 14, 'overbought': 80, 'oversold': 20},
           'obv': {'ema_span': 20},
           'atr': {'period': 14},
           'trend': {'ema20_period': 20, 'ema50_period': 50, 'sma200_period': 200},
           'vwap': {'period': 20},
           'order_imbalance': {'depth_levels': 10, 'threshold': 2.0},
           'bid_ask_spread': {'ma_period': 20},
           'market_depth': {'levels': 10, 'threshold': 2.0},
           'order_flow': {'window': 10}
       }
       return defaults.get(indicator, {})

   async def _get_optimal_params(self, indicator: str, market_data: Dict) -> Dict[str, Any]:
       try:
           prompt = f"""Analyze market data and determine optimal parameters for the {indicator} indicator.
           Parameter ranges: {json.dumps(self.parameter_ranges[indicator])}
           Market data: {json.dumps(market_data)}
           Return optimal parameters in JSON format"""

           async with aiohttp.ClientSession() as session:
               async with session.post(self.ollama_url,
                                       json={"model": "mistral", "prompt": prompt, "temperature": 0.2}) as response:
                   if response.status == 200:
                       result = await response.json()
                       try:
                           params_start = result.get('response', '').find('{')
                           params_end = result.get('response', '').rfind('}') + 1
                           params = json.loads(result.get('response', '')[params_start:params_end])
                           return params.get('parameters', self._get_default_params(indicator))
                       except:
                           return self._get_default_params(indicator)
           return self._get_default_params(indicator)
       except:
           return self._get_default_params(indicator)

   def _calculate_success_rate(self, signals: torch.Tensor, returns: torch.Tensor) -> float:
       """Calculate success rate of signals vs actual returns"""
       correct_predictions = ((signals > 0) & (returns > 0)) | ((signals < 0) & (returns < 0))
       return float(correct_predictions.sum() / len(signals))

   async def test_combinations(self, df: pd.DataFrame, lookback_period: int = 100) -> Dict:
       """Test various indicator combinations and return results"""
       results = {}
       test_data = df[-lookback_period:].copy()
       price_tensor = torch.tensor(test_data['close'].values, device=self.device)
       returns = (price_tensor[1:] - price_tensor[:-1]) / price_tensor[:-1]

       # Create shared memory for test data
       manager = mp.Manager()
       shared_data = manager.dict({
           'test_data': test_data.to_dict(),
           'returns': returns.cpu().numpy()
       })

       market_data = {
           'volatility': float(returns.std()),
           'trend': 'up' if returns.mean() > 0 else 'down',
           'avg_volume': float(test_data['volume'].mean()),
           'price_range': float(test_data['high'].max() - test_data['low'].min())
       }

       # Test pairs
       all_pairs = list(combinations(self.base_indicators.keys(), 2))
       qualifying_pairs = []

       print("\nTesting indicator pairs...")

       # Process pairs in parallel
       with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
           futures = [executor.submit(self._process_pair, pair, shared_data) for pair in all_pairs]

           with tqdm(total=len(all_pairs)) as pbar:
               for future in as_completed(futures):
                   result = future.result()
                   if result:
                       ind1, ind2, success_rate, params1, params2 = result
                       combo_name = f"{ind1}_{ind2}"
                       if success_rate >= self.threshold:
                           qualifying_pairs.append((ind1, ind2))
                           results[combo_name] = {
                               'indicators': [ind1, ind2],
                               'success_rate': float(success_rate),
                               'parameters': {ind1: params1, ind2: params2}
                           }
                       pbar.update(1)
                       pbar.set_postfix({'success_rate': f"{success_rate:.2f}"})

       # Test triples
       if qualifying_pairs:
           print("\nTesting triple combinations...")
           used_indicators = set(sum(qualifying_pairs, ()))
           remaining_indicators = set(self.base_indicators.keys()) - used_indicators

           # Generate all possible triples
           triples = [(ind1, ind2, ind3)
                      for ind1, ind2 in qualifying_pairs
                      for ind3 in remaining_indicators]

           # Process triples in parallel
           with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
               futures = [executor.submit(self._process_triple, triple, shared_data, results) for triple in triples]

               with tqdm(total=len(triples)) as pbar:
                   for future in as_completed(futures):
                       result = future.result()
                       if result:
                           ind1, ind2, ind3, success_rate, params1, params2, params3 = result
                           combo_name = f"{ind1}_{ind2}_{ind3}"
                           if success_rate >= self.triple_threshold:
                               results[combo_name] = {
                                   'indicators': [ind1, ind2, ind3],
                                   'success_rate': float(success_rate),
                                   'parameters': {ind1: params1, ind2: params2, ind3: params3}
                               }
                           pbar.update(1)
                           pbar.set_postfix({'success_rate': f"{success_rate:.2f}"})

       print("\nQualifying combinations:")
       for combo_name, data in sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True):
           print(f"{combo_name}: {data['success_rate']:.2f}")

       # Cleanup
       torch.cuda.empty_cache()
       del shared_data

       return results

   def _process_pair(self, pair_data: Tuple, shared_data: Dict) -> Optional[Tuple]:
       """Process a single pair of indicators"""
       ind1, ind2 = pair_data
       try:
           params1 = self._get_default_params(ind1)
           params2 = self._get_default_params(ind2)

           # Reconstruct DataFrame from shared dictionary
           test_data = pd.DataFrame(shared_data['test_data'])
           returns = torch.tensor(shared_data['returns'])

           # Generate signals for entire dataset first
           full_signals1 = np.array(self.base_indicators[ind1](test_data, **params1))
           full_signals2 = np.array(self.base_indicators[ind2](test_data, **params2))

           # Ensure both signal arrays are of length 100 (equal to lookback_period)
           if len(full_signals1) > 100:
               full_signals1 = full_signals1[-100:]
           if len(full_signals2) > 100:
               full_signals2 = full_signals2[-100:]

           # Convert to tensor and ensure same device
           signals1 = torch.tensor(full_signals1, device=self.device)
           signals2 = torch.tensor(full_signals2, device=self.device)

           # Ensure returns match signal length
           returns = returns[:99]  # One less than signals due to returns calculation

           # Calculate combined signals
           combined_signals = (signals1[:-1] + signals2[:-1]) / 2
           success_rate = self._calculate_success_rate(combined_signals, returns)

           return (ind1, ind2, success_rate, params1, params2)
       except Exception as e:
           logger.error(f"Error processing pair {ind1}_{ind2}: {e}")
           return None
   def _process_triple(self, triple_data: Tuple, shared_data: Dict, pair_results: Dict) -> Optional[Tuple]:
       """Process a single triple of indicators"""
       ind1, ind2, ind3 = triple_data
       try:
           params1 = pair_results[f"{ind1}_{ind2}"]['parameters'][ind1]
           params2 = pair_results[f"{ind1}_{ind2}"]['parameters'][ind2]
           params3 = self._get_default_params(ind3)

           # Reconstruct DataFrame from shared dictionary
           test_data = pd.DataFrame(shared_data['test_data'])
           returns = torch.tensor(shared_data['returns'])

           # Process in batches
           signals = []
           for i in range(0, len(test_data), self.batch_size):
               batch = test_data.iloc[i:i + self.batch_size]
               s1 = self.base_indicators[ind1](batch, **params1)
               s2 = self.base_indicators[ind2](batch, **params2)
               s3 = self.base_indicators[ind3](batch, **params3)
               signals.append((torch.tensor(s1) + torch.tensor(s2) + torch.tensor(s3)) / 3)

           combined_signals = torch.cat(signals)[:-1]
           success_rate = self._calculate_success_rate(combined_signals, returns)

           return (ind1, ind2, ind3, success_rate, params1, params2, params3)
       except Exception as e:
           logger.error(f"Error processing triple {ind1}_{ind2}_{ind3}: {e}")
           return None

   def _rsi_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from RSI indicator"""
       rsi = ind_funcs.calculate_rsi(df, params.get('period', 14))
       signals = np.zeros(len(df))
       signals[rsi > params.get('overbought', 70)] = -1
       signals[rsi < params.get('oversold', 30)] = 1
       return signals

   def _macd_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from MACD indicator"""
       macd_data = ind_funcs.calculate_macd(df,
                                        fast_period=params.get('fast_period', 12),
                                        slow_period=params.get('slow_period', 26),
                                        signal_period=params.get('signal_period', 9))
       signals = np.zeros(len(df))
       signals[macd_data['macd'] > macd_data['signal']] = 1
       signals[macd_data['macd'] < macd_data['signal']] = -1
       return signals

   def _bollinger_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from Bollinger Bands indicator"""
       bb = ind_funcs.calculate_bollinger(df,
                                          period=params.get('period', 20),
                                          std=params.get('std_dev', 2.0))

       signals = np.zeros(len(df))
       signals[df['close'] > bb['upper']] = -1
       signals[df['close'] < bb['lower']] = 1
       return signals

   def _ema_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from EMA crossover"""
       fast_span = params.get('fast_span', 12)
       slow_span = params.get('slow_span', 26)

       ema_fast = df['close'].ewm(span=fast_span, adjust=False).mean()
       ema_slow = df['close'].ewm(span=slow_span, adjust=False).mean()

       signals = np.zeros(len(df))
       signals[ema_fast > ema_slow] = 1
       signals[ema_fast < ema_slow] = -1
       return signals

   def _stoch_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from Stochastic Oscillator"""
       stoch = ind_funcs.calculate_stochastic(df,
                                         period=params.get('period', 14),
                                         k_period=params.get('d_period', 3))

       signals = np.zeros(len(df))
       signals[stoch['k'] > stoch['d']] = 1
       signals[stoch['k'] < stoch['d']] = -1
       return signals

   def _adx_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from ADX indicator"""
       adx_data = ind_funcs.calculate_adx(df, period=params.get('period', 14))

       signals = np.zeros(len(df))

       # Convert ADX components to numpy arrays and handle NaN values
       adx = np.nan_to_num(adx_data['adx'].values)
       plus_di = np.nan_to_num(adx_data['plus_di'].values)
       minus_di = np.nan_to_num(adx_data['minus_di'].values)

       # Generate signals ensuring same length as input
       threshold = params.get('threshold_high', 25)
       signals[(adx > threshold) & (plus_di > minus_di)] = 1
       signals[(adx > threshold) & (plus_di < minus_di)] = -1

       return signals[:len(df)]  # Ensure output length matches input

   def _mfi_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from Money Flow Index"""
       mfi = ind_funcs.calculate_mfi(df, period=params.get('period', 14))

       signals = np.zeros(len(df))
       signals[mfi > params.get('overbought', 80)] = -1
       signals[mfi < params.get('oversold', 20)] = 1
       return signals

   def _obv_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from On-Balance Volume"""
       obv_data = ind_funcs.calculate_obv(df, ema_period=params.get('ema_span', 20))

       signals = np.zeros(len(df))
       signals[obv_data['obv'] > obv_data['obv_ema']] = 1
       signals[obv_data['obv'] < obv_data['obv_ema']] = -1
       return signals

   def _atr_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from Average True Range"""
       atr = ind_funcs.calculate_atr(df, period=params.get('period', 14))
       atr_pct = atr / df['close'] * 100

       signals = np.zeros(len(df))
       signals[atr_pct > atr_pct.mean() + atr_pct.std()] = 1  # High volatility expansion
       signals[atr_pct < atr_pct.mean() - atr_pct.std()] = -1  # Low volatility contraction
       return signals

   def _trend_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from trend analysis using multiple timeframes"""
       ema20 = df['close'].ewm(span=params.get('ema20_period', 20), adjust=False).mean()
       ema50 = df['close'].ewm(span=params.get('ema50_period', 50), adjust=False).mean()
       sma200 = df['close'].rolling(window=params.get('sma200_period', 200)).mean()

       signals = np.zeros(len(df))

       # Strong trend signals when all moving averages align
       signals[(ema20 > ema50) & (ema50 > sma200)] = 1
       signals[(ema20 < ema50) & (ema50 < sma200)] = -1

       # Golden/Death cross signals
       ema20_crosses_up = (ema20 > ema50) & (ema20.shift(1) <= ema50.shift(1))
       ema20_crosses_down = (ema20 < ema50) & (ema20.shift(1) >= ema50.shift(1))

       signals[ema20_crosses_up] = 1
       signals[ema20_crosses_down] = -1

       return signals

   def _vwap_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from Volume Weighted Average Price"""
       vwap = ind_funcs.calculate_vwap(df, period=params.get('period', 20))

       signals = np.zeros(len(df))
       vwap_std = vwap.rolling(window=20).std()
       upper_band = vwap + vwap_std
       lower_band = vwap - vwap_std

       signals[df['close'] > upper_band] = -1  # Overbought
       signals[df['close'] < lower_band] = 1   # Oversold

       return signals

   def _order_imbalance_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from order book imbalance analysis"""
       imbalance = ind_funcs.calculate_order_imbalance(df, period=params.get('depth_levels', 10))
       threshold = params.get('threshold', 2.0)

       signals = np.zeros(len(df))
       signals[imbalance > threshold] = 1       # Strong buying pressure
       signals[imbalance < -threshold] = -1     # Strong selling pressure

       return signals

   def _bid_ask_spread_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from bid-ask spread analysis"""
       ma_period = params.get('ma_period', 20)

       # Approximate spread using high-low range
       spread = (df['high'] - df['low']) / df['close'] * 100
       spread_ma = spread.rolling(window=ma_period).mean()
       spread_std = spread.rolling(window=ma_period).std()

       signals = np.zeros(len(df))
       signals[spread > spread_ma + spread_std] = -1  # Wide spread - potential volatility
       signals[spread < spread_ma - spread_std] = 1   # Narrow spread - potential accumulation

       return signals

   def _market_depth_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from market depth analysis"""
       levels = params.get('levels', 10)
       threshold = params.get('threshold', 2.0)

       # Approximate depth using volume and price ranges
       depth_buy = df['volume'] * (df['close'] - df['low'])
       depth_sell = df['volume'] * (df['high'] - df['close'])

       # Calculate rolling depth ratio
       depth_ratio = depth_buy.rolling(window=levels).sum() / depth_sell.rolling(window=levels).sum()

       signals = np.zeros(len(df))
       signals[depth_ratio > threshold] = 1       # Strong buying depth
       signals[depth_ratio < 1/threshold] = -1    # Strong selling depth

       return signals

   def _order_flow_signal(self, df: pd.DataFrame, **params) -> np.ndarray:
       """Generate signals from order flow analysis"""
       window = params.get('window', 10)

       # Calculate aggressive buying/selling pressure
       buying_pressure = ((df['close'] - df['low']) / (df['high'] - df['low'])) * df['volume']
       selling_pressure = ((df['high'] - df['close']) / (df['high'] - df['low'])) * df['volume']

       # Calculate cumulative delta
       delta = buying_pressure - selling_pressure
       delta_ma = delta.rolling(window=window).mean()
       delta_std = delta.rolling(window=window).std()

       signals = np.zeros(len(df))
       signals[delta > delta_ma + delta_std] = 1    # Strong buying flow
       signals[delta < delta_ma - delta_std] = -1   # Strong selling flow

       return signals