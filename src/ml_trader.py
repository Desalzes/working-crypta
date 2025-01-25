import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import ta
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Set thread numbers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logger = logging.getLogger(__name__)


class Chomp1d(nn.Module):
    """Remove extra padding from the right side for causal convolution"""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN block with dilated causal convolution and residual connection"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        super().__init__()

        # First dilated convolution
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated convolution
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection if needed
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
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
    """Multi-scale TCN with skip connections"""

    def __init__(
        self, num_inputs, num_channels, kernel_size=2, dropout=0.2
    ):
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
    """Multi-timeframe TCN for market data processing"""

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
        channel_list = (
            [hidden_size // 2, hidden_size, hidden_size]
            if num_levels >= 2
            else [hidden_size]
        )

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
        x = (
            x.view(-1, seq_len, features).permute(0, 2, 1)
        )  # (batch*n_timeframes, features, seq_len)
        out = self.tcn(x)  # (batch*n_timeframes, hidden_size, seq_len_out)
        out = out[:, :, -1]  # Last timestep (batch*n_timeframes, hidden_size)
        out = out.view(b, n_timeframes, -1)  # (batch, n_timeframes, hidden_size)
        out = torch.mean(out, dim=1)  # Average across timeframes
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class MLTrader:
    def __init__(
        self,
        sequence_length: int = 12,
        hidden_size: int = 64,
        num_levels: int = 3,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        """Initialize MLTrader with TCN model for multi-timeframe trading"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.scalers: Dict[str, StandardScaler] = {}
        self.n_features = (
            10  # OHLCV + RSI, MACD, MACD Signal, VWAP, ATR
        )

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
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.data_dir = os.path.join(
            Path(__file__).parent.parent, "data", "historical"
        )
        self.model_dir = os.path.join(Path(__file__).parent.parent, "data", "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators required for the model"""
        df = df.copy()

        # Remove existing indicators if present
        for col in ["rsi", "macd", "macd_signal", "vwap", "atr"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Prevent divide by zero in volume calculations
        df["volume"] = df["volume"].replace(0, 1e-8)

        # Calculate indicators using ta library
        rsi = ta.momentum.RSIIndicator(df["close"], window=14)
        df["rsi"] = rsi.rsi()

        macd = ta.trend.MACD(
            df["close"], window_fast=12, window_slow=26, window_sign=9
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        vwap = ta.volume.VolumeWeightedAveragePrice(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            window=14,
        )
        df["vwap"] = vwap.volume_weighted_average_price()

        atr = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        )
        df["atr"] = atr.average_true_range()

        # Fill NaN values
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    def get_data_files(self, pair: str) -> Dict[str, str]:
        """Get CSV files for all timeframes of a trading pair"""
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
            filepath = os.path.join(pair_dir, fname)
            if os.path.exists(filepath):
                files[tf] = filepath
            else:
                logger.warning(f"No file found for timeframe {tf} -> {filepath}")
        return files

    def load_model(self, pair: str, model_filename: str) -> bool:
        """
        Load a pre-trained model for a specific trading pair.

        Args:
            pair (str): Trading pair identifier (e.g., "BTCUSD").
            model_filename (str): Filename of the pre-trained model.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            model_path = os.path.join(self.model_dir, model_filename)
            if not os.path.exists(model_path):
                logger.error(f"Model file does not exist: {model_path}")
                return False

            # Load the model state
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model for {pair} from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model for {pair}: {e}", exc_info=True)
            return False

    def save_model(self, pair: str, suffix: str = ""):
        """
        Save the current state of the model.

        Args:
            pair (str): Trading pair identifier (e.g., "BTCUSD").
            suffix (str, optional): Suffix to append to the model filename. Defaults to "".
        """
        try:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{pair}_{timestamp}{'_' + suffix if suffix else ''}.pt"
            model_path = os.path.join(self.model_dir, model_filename)
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Saved model for {pair} at {model_path}")
        except Exception as e:
            logger.error(f"Error saving model for {pair}: {e}", exc_info=True)
    def load_data(self, pair: str) -> Dict[str, pd.DataFrame]:
        """Load and preprocess data for all timeframes"""
        dataframes = {}
        files = self.get_data_files(pair)

        if not files:
            logger.error(f"No data files found for {pair}.")
            return dataframes

        for tf, filepath in files.items():
            try:
                df = pd.read_csv(filepath)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df = self.add_technical_features(df)
                dataframes[tf] = df
                logger.info(
                    f"Loaded {filepath} with {len(df)} rows after indicators"
                )
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")

        return dataframes

    def split_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
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
            self,
            dataframes: Dict[str, pd.DataFrame],
            phase: str = "train",
            max_samples: int = 10000,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare multi-timeframe data for training/prediction"""
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
                # Use more data for validation
                phase_dfs[tf] = pd.concat([val_df, test_df.iloc[:len(val_df)]])
            else:  # test
                phase_dfs[tf] = test_df

        # Find minimum usable length across timeframes
        min_len_across = min(len(df) for df in phase_dfs.values())
        usable_length = min(min_len_across, max_samples)

        # Check sequence length requirements
        min_required = self.sequence_length + 3  # sequence + lookahead
        if usable_length < min_required:
            logger.error(
                f"[{phase}] Insufficient data. Have {usable_length}, need {min_required}"
            )
            return None, None

        # Validation size check
        if phase == "val" and usable_length < self.sequence_length * 2:
            logger.error(f"[{phase}] Validation set too small: {usable_length}")
            return None, None

        # Process 1m data for labels
        df_1m = phase_dfs.get("1m", pd.DataFrame())
        if df_1m.empty:
            logger.error(f"[{phase}] 1m data missing")
            return None, None

        # Calculate labels from 1m data
        df_1m = df_1m.iloc[-usable_length:].copy()
        future_returns = df_1m["close"].pct_change(3).shift(-3)
        threshold = future_returns.std() * 0.3

        # Process features for each timeframe
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
        X_timeframes = []

        for tf, df_slice in phase_dfs.items():
            if df_slice.empty:
                logger.error(f"[{phase}] {tf} timeframe empty")
                return None, None

            # Prepare features
            df_slice = df_slice.iloc[-usable_length:].copy()
            features = df_slice[feature_columns].values

            # Scale features
            if phase == "train":
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(features)
                self.scalers[tf] = scaler
            else:
                if tf not in self.scalers:
                    logger.error(f"No scaler for {tf} in {phase}")
                    return None, None
                scaled_data = self.scalers[tf].transform(features)

            # Create sequences
            sequences = []
            for i in range(len(scaled_data) - self.sequence_length + 1):
                seq = scaled_data[i : i + self.sequence_length]
                sequences.append(seq)

            if not sequences:
                logger.error(f"[{phase}] No sequences for {tf}")
                return None, None

            seq_tensor = torch.tensor(
                np.array(sequences), dtype=torch.float32
            )
            X_timeframes.append(seq_tensor)

        # Align sequences across timeframes
        min_windows = min(x.size(0) for x in X_timeframes)
        X_timeframes = [x[-min_windows :] for x in X_timeframes]
        X = torch.stack(X_timeframes, dim=1)

        # Create labels
        future_returns = future_returns.iloc[self.sequence_length - 1 :][-min_windows:]
        if len(future_returns) < min_windows:
            logger.error(f"[{phase}] Label/sample size mismatch")
            return None, None

        y = torch.zeros(min_windows, dtype=torch.long)
        y[future_returns.values > threshold] = 1  # BUY
        y[future_returns.values < -threshold] = 2  # SELL

        return X, y

    async def train(
        self, pair: Optional[str] = None, epochs: int = 30, batch_size: int = 64
    ):
        """Train the model on historical data"""
        try:
            if pair:
                all_pairs = [pair]
            else:
                all_pairs = [
                    d
                    for d in os.listdir(self.data_dir)
                    if os.path.isdir(os.path.join(self.data_dir, d))
                ]
            all_pairs.sort()

            results = {}
            for current_pair in all_pairs:
                print(f"\nTraining model for {current_pair}...")
                print("Loading data...")
                pair_data = self.load_data(current_pair)
                if not pair_data:
                    print(f"No data for {current_pair}, skipping.")
                    continue

                print("Preparing training data...")
                X_train, y_train = self.prepare_data(pair_data, phase="train")
                if X_train is None or y_train is None:
                    print(
                        f"Could not prepare training data for {current_pair}, skipping."
                    )
                    continue

                # Get validation data
                X_val, y_val = self.prepare_data(pair_data, phase="val")
                has_validation = X_val is not None and y_val is not None

                # Create data loaders
                train_ds = torch.utils.data.TensorDataset(X_train, y_train)
                train_loader = torch.utils.data.DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=0,
                )

                if has_validation:
                    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
                    val_loader = torch.utils.data.DataLoader(
                        val_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=0,
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

                # Training loop
                epoch_logs = []
                best_val_loss = float("inf")
                patience = 15
                patience_counter = 0
                min_epochs = 5

                for ep in range(epochs):
                    self.model.train()
                    total_loss = 0.0
                    correct = 0
                    total = 0
                    batch_losses = []

                    # Training phase
                    for batch_x, batch_y in train_loader:
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
                            print(f"Training error on batch: {e}")
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

                    # Print progress
                    val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                    val_acc_str = f"{val_acc:.2f}" if val_acc is not None else "N/A"
                    print(
                        f"Epoch {ep + 1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Train Acc: {train_acc:.2f}%, "
                        f"Val Loss: {val_loss_str}, "
                        f"Val Acc: {val_acc_str}% "
                        f"(Min/Max/Mean Batch Loss: {min(batch_losses):.4f}/{max(batch_losses):.4f}/{np.mean(batch_losses):.4f})"
                    )

                    # Save best model if validation improves
                    if ep + 1 >= min_epochs:
                        if val_loss is not None and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            self.save_model(pair=current_pair, suffix="best")
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(
                                    f"\nEarly stopping triggered after {ep + 1} epochs"
                                )
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

                # Store results
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

    if __name__ == "__main__":
        """
        Example usage:
        ```python
        from ml_trader import MLTrader
        trader = MLTrader(sequence_length=12, hidden_size=64)
        results = await trader.train(pair="ADAUSDT", epochs=30, batch_size=32)
        print(results)
        ```
        """
        pass
