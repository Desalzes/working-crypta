import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from data_processor import DataProcessor

class MarketAnalyzer:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.logger = logging.getLogger(__name__)
        
    def analyze_market_regime(self, df: pd.DataFrame) -> Dict:
        try:
            regime_data = {}
            returns = df['close'].pct_change()
            regime_data['trend'] = {
                'direction': 'up' if returns.mean() > 0 else 'down',
                'strength': abs(returns.mean() / returns.std())
            }
            
            vol = returns.rolling(window=20).std()
            regime_data['volatility'] = {
                'current': float(vol.iloc[-1]),
                'percentile': float(vol.rank(pct=True).iloc[-1])
            }
            
            volume_ma = df['volume'].rolling(window=20).mean()
            regime_data['volume'] = {
                'current_vs_ma': float(df['volume'].iloc[-1] / volume_ma.iloc[-1]),
                'trend': 'increasing' if df['volume'].iloc[-20:].is_monotonic_increasing else 'decreasing'
            }
            
            return regime_data
            
        except Exception as e:
            self.logger.error(f"Error in market regime analysis: {e}")
            return {}
            
    def find_key_levels(self, df: pd.DataFrame) -> Dict:
        try:
            levels = {
                'support': [],
                'resistance': []
            }
            
            price_range = df['high'].max() - df['low'].min()
            min_distance = price_range * 0.02
            
            lows = df['low'].rolling(window=20, center=True).min()
            support_points = df[df['low'] == lows]
            
            highs = df['high'].rolling(window=20, center=True).max()
            resistance_points = df[df['high'] == highs]
            
            levels['support'] = self._cluster_price_levels(support_points['low'].values, min_distance)
            levels['resistance'] = self._cluster_price_levels(resistance_points['high'].values, min_distance)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error finding key levels: {e}")
            return {'support': [], 'resistance': []}
            
    def _cluster_price_levels(self, prices: np.array, min_distance: float) -> List[float]:
        if len(prices) == 0:
            return []
            
        clusters = []
        current_cluster = [prices[0]]
        
        for price in sorted(prices[1:]):
            if abs(price - np.mean(current_cluster)) < min_distance:
                current_cluster.append(price)
            else:
                clusters.append(float(np.mean(current_cluster)))
                current_cluster = [price]
                
        if current_cluster:
            clusters.append(float(np.mean(current_cluster)))
            
        return sorted(clusters)
        
    def analyze_volatility_regimes(self, df: pd.DataFrame) -> Dict:
        try:
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            
            low_vol = volatility.quantile(0.25)
            high_vol = volatility.quantile(0.75)
            
            regimes = {
                'low_volatility': {
                    'threshold': float(low_vol),
                    'frequency': float((volatility <= low_vol).mean()),
                    'avg_return': float(returns[volatility <= low_vol].mean())
                },
                'medium_volatility': {
                    'threshold': float(high_vol),
                    'frequency': float(((volatility > low_vol) & (volatility <= high_vol)).mean()),
                    'avg_return': float(returns[(volatility > low_vol) & (volatility <= high_vol)].mean())
                },
                'high_volatility': {
                    'threshold': float(high_vol),
                    'frequency': float((volatility > high_vol).mean()),
                    'avg_return': float(returns[volatility > high_vol].mean())
                }
            }
            
            return regimes
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility regimes: {e}")
            return {}