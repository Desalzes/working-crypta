# src/models.py

import torch
import torch.nn as nn

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
        input_size: int = 21,
        hidden_size: int = 512,
        num_levels: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        super().__init__()

        channel_list = [hidden_size // 2, hidden_size, hidden_size * 2, 
                       hidden_size * 2, hidden_size * 4, hidden_size * 4][:num_levels]

        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=channel_list,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(channel_list[-1], hidden_size * 2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Handle both single and multi-timeframe inputs
        if len(x.shape) == 3:  # [batch, seq_len, features]
            x = x.permute(0, 2, 1)  # [batch, features, seq_len]
            out = self.tcn(x)  # [batch, hidden_size, seq_len]
            out = out[:, :, -1]  # Last timestep: [batch, hidden_size]
        else:  # [batch, n_timeframes, seq_len, features]
            b, n_timeframes, seq_len, features = x.shape
            x = x.view(-1, seq_len, features).permute(0, 2, 1)  # [batch * n_timeframes, features, seq_len]
            out = self.tcn(x)  # [batch * n_timeframes, hidden_size, seq_len]
            out = out[:, :, -1]  # Last timestep: [batch * n_timeframes, hidden_size]
            out = out.view(b, n_timeframes, -1)  # [batch, n_timeframes, hidden_size]
            out = torch.mean(out, dim=1)  # Average across timeframes: [batch, hidden_size]
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.bn(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out