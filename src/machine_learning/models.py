import torch
import torch.nn as nn
from tcn_blocks import TemporalConvNet


class MultiTimeframeTCN(nn.Module):
    """
    Processes multi-timeframe data with a TCN.
    Expected input shape: [batch, n_timeframes, seq_len, features].
    """
    def __init__(self, input_size: int, hidden_size: int, num_levels: int = 3,
                 kernel_size: int = 2, dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        channel_list = [hidden_size // 2, hidden_size, hidden_size] if num_levels >= 2 else [hidden_size]

        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=channel_list,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, n_timeframes, seq_len, features = x.shape
        x = x.view(-1, seq_len, features).permute(0, 2, 1)  # (batch*n_timeframes, features, seq_len)
        out = self.tcn(x)
        out = out[:, :, -1]  # Last timestep (batch*n_timeframes,
