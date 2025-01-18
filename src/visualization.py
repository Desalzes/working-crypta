import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class IndicatorVisualizer:
    def __init__(self):
        pass
        
    def plot_combination_performance(self, combinations: Dict) -> go.Figure:
        sorted_combos = sorted(combinations.items(), 
                             key=lambda x: x[1]['success_rate'], 
                             reverse=True)
        
        names = [combo[0] for combo in sorted_combos]
        rates = [combo[1]['success_rate'] for combo in sorted_combos]
        num_indicators = [len(combo[1]['indicators']) for combo in sorted_combos]
        
        colors = ['rgb(33, 150, 243)' if n == 2 else 'rgb(76, 175, 80)' 
                 for n in num_indicators]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=names,
            y=rates,
            marker_color=colors,
            text=[f"{rate:.2%}" for rate in rates],
            textposition='auto',
        ))
        
        fig.update_layout(
            template='plotly_dark',
            title='Indicator Combination Performance',
            xaxis_title='Indicator Combinations',
            yaxis_title='Success Rate',
            yaxis_tickformat='%',
            showlegend=False,
            height=600
        )
        
        return fig
        
    def plot_signal_analysis(self, df: pd.DataFrame, signals: Dict, combo_name: str) -> go.Figure:
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           row_heights=[0.7, 0.3])
        
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        buy_signals = df[signals > 0]
        sell_signals = df[signals < 0]
        
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['close'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Buy Signal'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['close'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Sell Signal'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=signals,
                name='Signal Strength',
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template='plotly_dark',
            title=f'Signal Analysis for {combo_name}',
            yaxis_title='Price',
            yaxis2_title='Signal Strength',
            height=800
        )
        
        return fig
        
    def plot_performance_metrics(self, metrics: Dict) -> go.Figure:
        fig = make_subplots(rows=1, cols=2, 
                           specs=[[{"type": "domain"}, {"type": "domain"}]])
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics['success_rate'] * 100,
                title={'text': "Success Rate"},
                domain={'x': [0, 0.5], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 60], 'color': "gray"},
                        {'range': [60, 100], 'color': "darkblue"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 40
                    }
                }
            ),
            row=1, col=1
        )
        
        risk_value = {
            'LOW': 25,
            'MEDIUM': 50,
            'HIGH': 75,
            'EXTREME': 100
        }.get(metrics.get('risk_level', 'MEDIUM'), 50)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_value,
                title={'text': "Risk Level"},
                domain={'x': [0.5, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'steps': [
                        {'range': [0, 25], 'color': "green"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_value
                    }
                }
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            template='plotly_dark',
            title='Performance Metrics Dashboard',
            height=400
        )
        
        return fig