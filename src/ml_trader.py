import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class MLTrader:
    def __init__(self, sequence_length: int = 30, hidden_size: int = 128, num_layers: int = 2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.model = MultiTimeframeLSTM(
            input_size=7,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=3
        ).to(self.device)
        self.scalers = {}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.data_dir = os.path.join(Path(__file__).parent.parent, 'data', 'historical')
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        timeframes = {
            '1m': '1INCHUSD_1m_historical.csv',
            '5m': '1INCHUSD_5m_historical.csv',
            '15m': '1INCHUSD_15m_historical.csv'
        }
        
        dataframes = {}
        for timeframe, filename in timeframes.items():
            filepath = os.path.join(self.data_dir, filename)
            try:
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                dataframes[timeframe] = df
                logger.info(f"Loaded {filename} with {len(df)} rows")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                
        return dataframes
        
    def prepare_data(self, dataframes: Dict[str, pd.DataFrame]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_length = min(len(df) for df in dataframes.values()) - self.sequence_length
        X_timeframes = []
        
        for timeframe, df in dataframes.items():
            if timeframe not in self.scalers:
                self.scalers[timeframe] = StandardScaler()
            
            df = df.iloc[-sequence_length:]
            features = df[['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']].values
            scaled_data = self.scalers[timeframe].fit_transform(features)
            
            # Pad sequences to match sequence_length
            padded_sequences = []
            for i in range(len(scaled_data) - self.sequence_length):
                seq = scaled_data[i:i + self.sequence_length]
                padded_sequences.append(seq)
            
            sequences_tensor = torch.FloatTensor(np.array(padded_sequences))
            X_timeframes.append(sequences_tensor)
        
        min_len = min(x.size(0) for x in X_timeframes)
        X_timeframes = [x[-min_len:] for x in X_timeframes]
        
        df_1m = dataframes['1m'].iloc[-sequence_length:]
        future_returns = df_1m['close'].pct_change().shift(-1)
        threshold = future_returns.std() * 0.5
        
        y = torch.zeros(min_len, dtype=torch.long)
        future_returns = future_returns.iloc[self.sequence_length:].iloc[-min_len:]
        y[future_returns.values > threshold] = 1
        y[future_returns.values < -threshold] = 2
        
        return torch.stack(X_timeframes, dim=1), y
        
    def train(self, epochs: int = 100, batch_size: int = 32):
        print("\nLoading and preparing data...")
        dataframes = self.load_data()
        if not dataframes:
            logger.error("No data available for training")
            return None
            
        X, y = self.prepare_data(dataframes)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print("\nStarting training...")
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}, '
                      f'Accuracy: {100 * correct/total:.2f}%')
                
        results = self.backtest(dataframes)
        return results
                
    def predict(self, dataframes: Dict[str, pd.DataFrame]) -> torch.Tensor:
        with torch.no_grad():
            X, _ = self.prepare_data(dataframes)
            X = X.to(self.device)
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities
            
    def backtest(self, dataframes: Dict[str, pd.DataFrame], 
                initial_balance: float = 10000.0,
                position_size: float = 0.1) -> Dict:
        predictions = self.predict(dataframes)
        signals = torch.argmax(predictions, dim=1).cpu().numpy()
        
        df = dataframes['1m']
        prices = df['close'].values[-len(signals):]
        
        balance = initial_balance
        position = 0
        trades = []
        
        for i in range(len(signals)):
            if signals[i] == 1 and position <= 0:  # Buy signal
                position = (balance * position_size) / prices[i]
                trades.append({
                    'type': 'buy',
                    'price': prices[i],
                    'size': position,
                    'balance': balance
                })
            elif signals[i] == 2 and position >= 0:  # Sell signal
                if position > 0:
                    balance += position * prices[i]
                    position = -(balance * position_size) / prices[i]
                    trades.append({
                        'type': 'sell',
                        'price': prices[i],
                        'size': abs(position),
                        'balance': balance
                    })
        
        if position != 0:
            balance += position * prices[-1]
            trades.append({
                'type': 'close',
                'price': prices[-1],
                'size': abs(position),
                'balance': balance
            })
            
        return {
            'final_balance': balance,
            'return': (balance - initial_balance) / initial_balance * 100,
            'trades': trades
        }

class MultiTimeframeLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super(MultiTimeframeLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        n_timeframes = x.size(1)
        
        # Reshape for LSTM
        x = x.view(-1, x.size(2), x.size(3))
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        
        # Process each timeframe
        out = out.view(batch_size, n_timeframes, -1)
        out = torch.mean(out, dim=1)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out