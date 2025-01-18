import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import os
import pickle

logger = logging.getLogger(__name__)


class MLTrader:
    def __init__(self, sequence_length: int = 30, hidden_size: int = 128, num_layers: int = 2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.sequence_length = sequence_length
        self.model = MultiTimeframeLSTM(
            input_size=5,  # open, high, low, close, volume
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=3
        ).to(self.device)
        self.scalers = {}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.data_dir = os.path.join(Path(__file__).parent.parent, 'data', 'historical')

    def get_data_files(self, pair: str) -> Dict[str, str]:
        timeframes = {'1m': '_1m_data.csv', '5m': '_5m_data.csv', '15m': '_15m_data.csv'}
        files = {}

        pair_dir = os.path.join(self.data_dir, pair)
        if not os.path.exists(pair_dir):
            return files

        for timeframe, suffix in timeframes.items():
            filepath = os.path.join(pair_dir, f'{pair}{suffix}')
            if os.path.exists(filepath):
                files[timeframe] = filepath

        return files

    def load_data(self, pair: str) -> Dict[str, pd.DataFrame]:
        dataframes = {}
        files = self.get_data_files(pair)

        for timeframe, filepath in files.items():
            try:
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                dataframes[timeframe] = df
                logger.info(f'Loaded {filepath} with {len(df)} rows')
            except Exception as e:
                logger.error(f'Error loading {filepath}: {e}')

        return dataframes

    def prepare_data(self, dataframes: Dict[str, pd.DataFrame]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_length = min(len(df) for df in dataframes.values()) - self.sequence_length
        X_timeframes = []

        for timeframe, df in dataframes.items():
            if timeframe not in self.scalers:
                self.scalers[timeframe] = StandardScaler()

            df = df.iloc[-sequence_length:]
            features = df[['open', 'high', 'low', 'close', 'volume']].values
            scaled_data = self.scalers[timeframe].fit_transform(features)

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

    async def train(self, epochs: int = 100, batch_size: int = 32, pair: str = None):
        if pair:
            dataframes = self.load_data(pair)
            if not dataframes:
                logger.error(f'No data available for {pair}')
                return None
        else:
            all_dataframes = {}
            for pair_dir in os.listdir(self.data_dir):
                if os.path.isdir(os.path.join(self.data_dir, pair_dir)):
                    pair_data = self.load_data(pair_dir)
                    if pair_data:
                        all_dataframes[pair_dir] = pair_data

            if not all_dataframes:
                logger.error('No data available for training')
                return None

            dataframes = {}
            for timeframe in ['1m', '5m', '15m']:
                dfs = [data[timeframe] for data in all_dataframes.values() if timeframe in data]
                if dfs:
                    dataframes[timeframe] = pd.concat(dfs)

        X, y = self.prepare_data(dataframes)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print('\nStarting training...')
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
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(loader):.4f}, '
                      f'Accuracy: {100 * correct / total:.2f}%')

        results = self.backtest(dataframes)
        self.save_model(pair)
        return results

    def save_model(self, pair: str = None):
        """Save model state and scalers"""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        model_dir = os.path.join(Path(__file__).parent.parent, 'data', 'models')
        os.makedirs(model_dir, exist_ok=True)

        model_name = f"{pair}_model_{timestamp}" if pair else f"combined_model_{timestamp}"
        model_path = os.path.join(model_dir, f"{model_name}.pt")
        scalers_path = os.path.join(model_dir, f"{model_name}_scalers.pkl")

        # Save model state
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, model_path)

        # Save scalers
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)

        print(f"\nModel saved to: {model_path}")
        print(f"Scalers saved to: {scalers_path}")

    def load_model(self, model_path: str, scalers_path: str):
        """Load model state and scalers"""
        if os.path.exists(model_path) and os.path.exists(scalers_path):
            # Load model state
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            # Load scalers
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)

            print(f"\nModel loaded from: {model_path}")
            print(f"Scalers loaded from: {scalers_path}")
        else:
            raise FileNotFoundError("Model or scalers file not found")

    def predict(self, dataframes: Dict[str, pd.DataFrame]) -> Dict:
        with torch.no_grad():
            X, _ = self.prepare_data(dataframes)
            X = X.to(self.device)
            outputs = self.model(X)
            probabilities = torch.softmax(outputs[-1], dim=0)

            action = "HOLD"
            if torch.argmax(probabilities) == 1:
                action = "BUY"
            elif torch.argmax(probabilities) == 2:
                action = "SELL"

            return {
                "action": action,
                "confidence": float(torch.max(probabilities)),
                "probabilities": {
                    "hold": float(probabilities[0]),
                    "buy": float(probabilities[1]),
                    "sell": float(probabilities[2])
                }
            }

    def backtest(self, dataframes: Dict[str, pd.DataFrame],
                 initial_balance: float = 10000.0,
                 position_size: float = 0.1) -> Dict:
        with torch.no_grad():
            X, _ = self.prepare_data(dataframes)
            X = X.to(self.device)
            outputs = self.model(X)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        df = dataframes['1m']
        prices = df['close'].values[-len(predictions):]

        balance = initial_balance
        position = 0
        trades = []

        for i in range(len(predictions)):
            if predictions[i] == 1 and position <= 0:  # Buy signal
                position = (balance * position_size) / prices[i]
                balance -= position * prices[i]
                trades.append({
                    'type': 'buy',
                    'price': prices[i],
                    'position': position,
                    'balance': balance
                })

            elif predictions[i] == 2 and position >= 0:  # Sell signal
                if position > 0:
                    balance += position * prices[i]
                    position = 0
                    trades.append({
                        'type': 'sell',
                        'price': prices[i],
                        'position': position,
                        'balance': balance
                    })

        if position != 0:
            balance += position * prices[-1]
            trades.append({
                'type': 'close',
                'price': prices[-1],
                'position': 0,
                'balance': balance
            })

        return {
            'final_balance': balance,
            'return': (balance - initial_balance) / initial_balance * 100,
            'n_trades': len(trades),
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

        x = x.view(-1, x.size(2), x.size(3))
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        out = out.view(batch_size, n_timeframes, -1)
        out = torch.mean(out, dim=1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out