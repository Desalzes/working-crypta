import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ta

# Set thread numbers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logger = logging.getLogger(__name__)


class Chomp1d(nn.Module):
    """Remove extra padding from the right side for causal convolution."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN block with dilated causal convolution and residual connection."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super().__init__()

        # First dilated convolution
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated convolution
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection if needed
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Multi-scale TCN with skip connections."""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiTimeframeTCN(nn.Module):
    """Multi-timeframe TCN for market data processing."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_levels: int = 3,
        kernel_size: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()

        channel_list = [hidden_size // 2, hidden_size, hidden_size] if num_levels >= 2 else [hidden_size]

        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=channel_list,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, n_timeframes, seq_len, features]
        b, n_timeframes, seq_len, features = x.shape
        x = x.view(-1, seq_len, features).permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # Last timestep
        out = out.view(b, n_timeframes, -1)
        out = torch.mean(out, dim=1)  # Average across timeframes
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class MLTrader:
    def __init__(
        self,
        sequence_length: int = 14,
        hidden_size: int = 64,
        num_levels: int = 3,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        """Initialize MLTrader with TCN model for multi-timeframe trading."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.scalers: Dict[str, StandardScaler] = {}
        self.n_features = 10  # OHLCV + RSI, MACD, MACD Signal, VWAP, ATR

        self.model = MultiTimeframeTCN(
            input_size=self.n_features,
            hidden_size=hidden_size,
            num_levels=num_levels,
            kernel_size=2,
            dropout=0.2,
            num_classes=3,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            from torch.amp import GradScaler

            self.scaler = GradScaler("cuda")
        else:
            self.scaler = None

        self.data_dir = os.path.join(Path(__file__).parent.parent, "data", "historical")
        self.model_dir = os.path.join(Path(__file__).parent.parent, "data", "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and fill NaN values."""
        df = df.copy()
        for col in ["rsi", "macd", "macd_signal", "vwap", "atr"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        df["volume"] = df["volume"].replace(0, 1e-8)

        # Calculate indicators
        rsi = ta.momentum.RSIIndicator(df["close"], window=14)
        df["rsi"] = rsi.rsi()

        macd = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        vwap = ta.volume.VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14
        )
        df["vwap"] = vwap.volume_weighted_average_price()

        atr = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        )
        df["atr"] = atr.average_true_range()

        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    def get_data_files(self, pair: str) -> Dict[str, str]:
        """Get CSV file paths for all timeframes."""
        timeframes = {
            "1m": f"{pair}_1m_data.csv",
            "5m": f"{pair}_5m_data.csv",
            "15m": f"{pair}_15m_data.csv",
            "1h": f"{pair}_1h_data.csv",
            "4h": f"{pair}_4h_data.csv",
            "1d": f"{pair}_1d_data.csv",
        }
        pair_dir = os.path.join(self.data_dir, pair)
        files = {}

        for tf, fname in timeframes.items():
            fp = os.path.join(pair_dir, fname)
            if os.path.exists(fp):
                files[tf] = fp
        return files

    def load_data(self, pair: str) -> Dict[str, pd.DataFrame]:
        """Load and preprocess data for all timeframes."""
        dataframes = {}
        files = self.get_data_files(pair)

        for tf, fp in files.items():
            try:
                df = pd.read_csv(fp)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df = self.add_technical_features(df)
                dataframes[tf] = df
                logger.info(f"Loaded {fp} with {len(df)} rows after indicators")
            except Exception as e:
                logger.error(f"Error loading {fp}: {e}")

        return dataframes

    def split_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        n = len(df)
        if n < 2:
            return df, pd.DataFrame(), pd.DataFrame()

        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        return train_df, val_df, test_df

    def prepare_data(
        self, dataframes: Dict[str, pd.DataFrame], phase: str = "train", max_samples: int = 10000
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare multi-timeframe data for training/prediction."""
        if not dataframes or "1m" not in dataframes:
            logger.error("Missing dataframes or 1m data required")
            return None, None

        # Split data for each timeframe
        phase_dfs = {}
        for tf, df_full in dataframes.items():
            train_df, val_df, test_df = self.split_dataframe(df_full)
            if phase == "train":
                phase_dfs[tf] = train_df
            elif phase == "val":
                # Use more data for validation by including test data
                phase_dfs[tf] = pd.concat([val_df, test_df.iloc[: len(val_df)]])
            else:
                phase_dfs[tf] = test_df

        min_len_across = min(len(df) for df in phase_dfs.values())
        usable_length = min(min_len_across, max_samples)

        if usable_length < self.sequence_length:
            logger.error(f"[{phase}] Insufficient data: {usable_length} samples")
            return None, None

        X_timeframes = []
        df_1m = phase_dfs.get("1m", pd.DataFrame())
        if df_1m.empty:
            logger.error(f"[{phase}] 1m data missing")
            return None, None

        # Calculate labels from 1m data
        df_1m = df_1m.iloc[-usable_length:].copy()
        future_returns = df_1m["close"].pct_change(3).shift(-3)
        threshold = future_returns.std() * 0.3

        feature_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "vwap",
            "atr",
        ]

        for tf, df_slice in phase_dfs.items():
            if df_slice.empty:
                logger.error(f"[{phase}] {tf} timeframe empty")
                return None, None

            df_slice = df_slice.iloc[-usable_length:].copy()
            features = df_slice[feature_columns].values

            # Scale features
            if phase == "train":
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(features)
                self.scalers[tf] = scaler
            else:
                if tf not in self.scalers:
                    logger.error(f"No scaler found for {tf}")
                    return None, None
                scaled_data = self.scalers[tf].transform(features)

            # Create sequences
            sequences = []
            for i in range(len(scaled_data) - self.sequence_length + 1):
                seq = scaled_data[i : i + self.sequence_length]
                sequences.append(seq)

            if not sequences:
                logger.warning(f"[{phase}] No sequences for {tf}")
                return None, None

            seq_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
            X_timeframes.append(seq_tensor)

        # Align sequences across timeframes
        min_windows = min(x.size(0) for x in X_timeframes)
        X_timeframes = [x[-min_windows:] for x in X_timeframes]
        X = torch.stack(X_timeframes, dim=1)

        # Create labels
        future_returns = future_returns.iloc[self.sequence_length - 1 :][-min_windows:]
        y = torch.zeros(min_windows, dtype=torch.long)
        y[future_returns.values > threshold] = 1  # BUY
        y[future_returns.values < -threshold] = 2  # SELL

        return X, y

    async def train(self, pair: Optional[str] = None, epochs: int = 30, batch_size: int = 64):
        """Train the model on historical data."""
        try:
            if pair:
                all_pairs = [pair]
            else:
                all_pairs = [
                    d for d in os.listdir(self.data_dir)
                    if os.path.isdir(os.path.join(self.data_dir, d))
                ]
            all_pairs.sort()

            results = {}
            for current_pair in all_pairs:
                print(f"\nTraining on Pair: {current_pair}")
                print("Loading data...")
                pair_data = self.load_data(current_pair)
                if not pair_data:
                    print(f"No data for {current_pair}, skipping.")
                    continue

                print("Preparing training data...")
                X_train, y_train = self.prepare_data(pair_data, phase="train")
                if X_train is None or y_train is None:
                    print(f"Could not prepare training data for {current_pair}, skipping.")
                    continue

                X_val, y_val = self.prepare_data(pair_data, phase="val")
                has_validation = (X_val is not None and y_val is not None)

                # Create data loaders with 12 workers
                train_ds = torch.utils.data.TensorDataset(X_train, y_train)
                train_loader = torch.utils.data.DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10
                )

                if has_validation:
                    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
                    val_loader = torch.utils.data.DataLoader(
                        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10
                    )
                else:
                    val_loader = None

                # Calculate class weights
                unique_labels, counts = y_train.unique(return_counts=True)
                weights = counts.float()
                weights = weights.sum() / (weights * len(unique_labels))
                class_weights = weights.to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)

                print(
                    f"Training Data Shape: {X_train.shape}, "
                    f"Validation Data Shape: {X_val.shape if has_validation else 'N/A'}, "
                    f"Batch Size: {batch_size}, "
                    f"Class Weights: {class_weights.cpu().numpy()}, "
                    f"Label Distribution: {dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))}"
                )

                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"Total Model Parameters: {total_params:,}")

                # Training constants
                epoch_logs = []
                best_val_loss = float("inf")
                patience = 15
                patience_counter = 0
                improvement_threshold = 0.005
                minimum_epochs_before_saving = 5

                for ep in range(epochs):
                    self.model.train()
                    total_loss = 0.0
                    correct = 0
                    total = 0
                    batch_losses = []

                    for batch_idx, (batch_x, batch_y) in enumerate(train_loader, 1):
                        batch_x = batch_x.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)
                        self.optimizer.zero_grad()

                        try:
                            if self.scaler:
                                with torch.amp.autocast("cuda"):
                                    outputs = self.model(batch_x)
                                    loss = self.criterion(outputs, batch_y)
                                self.scaler.scale(loss).backward()
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                outputs = self.model(batch_x)
                                loss = self.criterion(outputs, batch_y)
                                loss.backward()
                                self.optimizer.step()

                            total_loss += loss.item()
                            batch_losses.append(loss.item())
                            _, predicted = outputs.max(1)
                            total += batch_y.size(0)
                            correct += predicted.eq(batch_y).sum().item()

                        except RuntimeError as e:
                            print(f"Training error on batch {batch_idx}: {e}")
                            torch.cuda.empty_cache()
                            continue

                    avg_train_loss = total_loss / len(train_loader)
                    train_acc = 100.0 * correct / total

                    # Validation phase
                    val_loss = None
                    val_acc = None
                    if val_loader:
                        self.model.eval()
                        val_total_loss = 0.0
                        val_correct = 0
                        val_total = 0

                        with torch.no_grad():
                            for batch_x, batch_y in val_loader:
                                batch_x = batch_x.to(self.device, non_blocking=True)
                                batch_y = batch_y.to(self.device, non_blocking=True)
                                outputs = self.model(batch_x)
                                loss = self.criterion(outputs, batch_y)
                                val_total_loss += loss.item()
                                _, predicted = outputs.max(1)
                                val_total += batch_y.size(0)
                                val_correct += predicted.eq(batch_y).sum().item()

                        val_loss = val_total_loss / len(val_loader)
                        val_acc = 100.0 * val_correct / val_total

                    val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                    val_acc_str = f"{val_acc:.2f}" if val_acc is not None else "N/A"

                    # Print progress
                    print(
                        f"Epoch {ep + 1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Train Acc: {train_acc:.2f}%, "
                        f"Val Loss: {val_loss_str}, "
                        f"Val Acc: {val_acc_str}% "
                        f"(Min/Max/Mean Batch Loss: {min(batch_losses):.4f}/{max(batch_losses):.4f}/{np.mean(batch_losses):.4f})"
                    )
                    # Model saving logic
                    if (ep + 1) >= minimum_epochs_before_saving:
                        if val_loss is not None and val_loss < (best_val_loss - improvement_threshold):
                            best_val_loss = val_loss
                            patience_counter = 0
                            self.save_model(pair=current_pair, suffix="best")
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"\nEarly stopping triggered after {ep + 1} epochs")
                                break

                    # Log epoch results
                    epoch_logs.append(
                        {
                            "epoch": ep + 1,
                            "train_loss": avg_train_loss,
                            "train_accuracy": train_acc,
                            "val_loss": val_loss,
                            "val_accuracy": val_acc,
                        }
                    )

                # Clean up and save final model
                torch.cuda.empty_cache()
                self.save_model(pair=current_pair)

                results[current_pair] = {
                    "training_history": epoch_logs,
                    "final_metrics": {
                        "train_loss": avg_train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    },
                }

            return results

        except Exception as e:
            print(f"Error in training: {e}")
            import traceback

            traceback.print_exc()
            return None

    def save_model(self, pair: Optional[str] = None, suffix: str = "") -> None:
        """Save model state and scalers."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        name_parts = [pair if pair else "combined_model", timestamp]
        if suffix:
            name_parts.append(suffix)
        model_name = "_".join(name_parts)

        model_path = os.path.join(self.model_dir, f"{model_name}.pt")
        scalers_path = os.path.join(self.model_dir, f"{model_name}_scalers.pkl")

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "sequence_length": self.sequence_length,
                "n_features": self.n_features,
            },
            model_path,
        )

        with open(scalers_path, "wb") as f:
            pickle.dump(self.scalers, f)

        print(f"\nModel saved to: {model_path}")
        print(f"Scalers saved to: {scalers_path}")

    def load_model(self, model_path: str, scalers_path: str) -> None:
        """Load model state and scalers."""
        if not (os.path.exists(model_path) and os.path.exists(scalers_path)):
            raise FileNotFoundError("Model or scalers file not found")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.sequence_length = checkpoint.get("sequence_length", self.sequence_length)
        self.n_features = checkpoint.get("n_features", self.n_features)

        with open(scalers_path, "rb") as f:
            self.scalers = pickle.load(f)

        print(f"\nModel loaded from: {model_path}")
        print(f"Scalers loaded from: {scalers_path}")

    def predict(self, dataframes: Dict[str, pd.DataFrame]) -> Dict:
        """Generate trading signals from recent data."""
        with torch.no_grad():
            processed_dfs = {}
            for tf, df in dataframes.items():
                processed_dfs[tf] = self.add_technical_features(df)

            X, _ = self.prepare_data(processed_dfs, phase="test")
            if X is None or len(X) == 0:
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "probabilities": {"hold": 1.0, "buy": 0.0, "sell": 0.0},
                }

            X = X.to(self.device)
            outputs = self.model(X)
            final_output = outputs[-1]
            probs = torch.softmax(final_output, dim=0)

            action_idx = torch.argmax(probs).item()
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}

            return {
                "action": action_map[action_idx],
                "confidence": float(torch.max(probs)),
                "probabilities": {
                    "hold": float(probs[0]),
                    "buy": float(probs[1]),
                    "sell": float(probs[2]),
                },
            }

    def backtest(
        self,
        dataframes: Dict[str, pd.DataFrame],
        initial_balance: float = 10000.0,
        position_size: float = 0.1,
    ) -> Dict:
        """
        Run backtest on test data with:
        - BUY: Invest position_size if no position
        - SELL: Liquidate position
        Returns balance, return, trades, and metrics.
        """
        with torch.no_grad():
            processed_dfs = {}
            for tf, df in dataframes.items():
                processed_dfs[tf] = self.add_technical_features(df)

            X, _ = self.prepare_data(processed_dfs, phase="test")
            if X is None or len(X) == 0:
                return {
                    "final_balance": initial_balance,
                    "return": 0.0,
                    "n_trades": 0,
                    "trades": [],
                }

            X = X.to(self.device)
            outputs = self.model(X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            df_1m = processed_dfs.get("1m", pd.DataFrame())
            if df_1m.empty:
                return {
                    "final_balance": initial_balance,
                    "return": 0.0,
                    "n_trades": 0,
                    "trades": [],
                }

            _, _, df_test = self.split_dataframe(df_1m)
            prices = df_test["close"].values[-len(preds):]

            balance = initial_balance
            position = 0.0
            trades = []

            for i, signal in enumerate(preds):
                current_price = prices[i]

                if signal == 1 and position <= 0:  # BUY
                    amount_to_buy = (balance * position_size) / current_price
                    cost = amount_to_buy * current_price
                    balance -= cost
                    position = amount_to_buy

                    trades.append(
                        {
                            "type": "buy",
                            "price": float(current_price),
                            "amount": float(amount_to_buy),
                            "cost": float(cost),
                            "balance": float(balance),
                            "timestamp": df_test.index[i].isoformat(),
                        }
                    )

                elif signal == 2 and position > 0:  # SELL
                    revenue = position * current_price
                    profit_loss = revenue - trades[-1]["cost"]
                    balance += revenue

                    trades.append(
                        {
                            "type": "sell",
                            "price": float(current_price),
                            "amount": float(position),
                            "revenue": float(revenue),
                            "profit_loss": float(profit_loss),
                            "balance": float(balance),
                            "timestamp": df_test.index[i].isoformat(),
                        }
                    )
                    position = 0

            # Close any remaining position
            if position > 0:
                revenue = position * prices[-1]
                profit_loss = revenue - trades[-1]["cost"]
                balance += revenue

                trades.append(
                    {
                        "type": "close",
                        "price": float(prices[-1]),
                        "amount": float(position),
                        "revenue": float(revenue),
                        "profit_loss": float(profit_loss),
                        "balance": float(balance),
                        "timestamp": df_test.index[-1].isoformat(),
                    }
                )

            # Calculate metrics
            total_return = (balance - initial_balance) / initial_balance * 100
            n_trades = len([t for t in trades if t["type"] in ("buy", "sell")])

            if n_trades > 0:
                profits = sum([t.get("profit_loss", 0) for t in trades if t.get("profit_loss", 0) > 0])
                losses = sum([t.get("profit_loss", 0) for t in trades if t.get("profit_loss", 0) < 0])
                win_trades = sum(1 for t in trades if t.get('profit_loss', 0) > 0)

                metrics = {
                    'win_rate': win_trades / n_trades * 100,
                    'profit_factor': abs(profits / losses) if losses != 0 else float('inf'),
                    'avg_profit_per_trade': (balance - initial_balance) / n_trades
                }
            else:
                metrics = {
                    'win_rate': 0,
                    'profit_factor': 0,
                    'avg_profit_per_trade': 0
                }

            return {
                'final_balance': float(balance),
                'return': float(total_return),
                'n_trades': n_trades,
                'trades': trades,
                'metrics': metrics
            }

        if __name__ == "__main__":
            """
            Example usage:
            ```python
            from asyncio import run
            trader = MLTrader(sequence_length=14, hidden_size=64)
            results = run(trader.train(pair="ADAUSDT", epochs=30, batch_size=64))
            print(results)
            ```
            """
            pass