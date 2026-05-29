import torch

from .aiosa import LSTM


class _RVarDSepBlock(torch.nn.Module):
    """Depthwise-separable conv block with variational-dropout LSTM on top."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.out_ch = out_ch
        self.conv1 = torch.nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch)
        self.bn1 = torch.nn.BatchNorm1d(in_ch)
        self.conv2 = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, padding=0)

        self.relu = torch.nn.ReLU()
        self.out_ch = in_ch + out_ch

        self.lstm = LSTM(input_size=out_ch, hidden_size=out_ch, batch_first=True, dropouti=0.1)

    def forward(self, x, training=True):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = torch.add(x, y)
        x = self.conv2(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)

        return x
