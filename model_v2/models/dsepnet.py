import torch
from torch import nn

from ._dsep_block import _RVarDSepBlock


class DSepST15Net_no_branch(nn.Module):
    def __init__(self, num_channels=7, hidden_size=64, hidden_size2=4, layer=2, dropout=0.3, winsize=60):
        super().__init__()

        in_ch = num_channels

        self.dsep00 = _RVarDSepBlock(in_ch, 64)
        self.dsep01 = _RVarDSepBlock(64, 64)
        self.dsep02 = _RVarDSepBlock(64, 64)
        self.dsep03 = _RVarDSepBlock(64, 64)
        self.avg0 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.dsep10 = _RVarDSepBlock(32, 32)
        self.dsep11 = _RVarDSepBlock(32, 32)
        self.dsep12 = _RVarDSepBlock(32, 32)
        self.dsep13 = _RVarDSepBlock(32, 32)
        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.dsep20 = _RVarDSepBlock(16, 16)
        self.dsep21 = _RVarDSepBlock(16, 16)
        self.dsep22 = _RVarDSepBlock(16, 16)
        self.dsep23 = _RVarDSepBlock(16, 16)
        self.avg2 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.dsep30 = _RVarDSepBlock(8, 8)
        self.dsep31 = _RVarDSepBlock(8, 8)
        self.dsep32 = _RVarDSepBlock(8, 8)
        self.dsep33 = _RVarDSepBlock(8, 8)
        self.avg3 = nn.AvgPool1d(kernel_size=2, stride=2)

        flattened_size = 240

        self.dense1 = nn.Linear(flattened_size, 240)
        self.dense2 = nn.Linear(240, 1)

        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(num_features=hidden_size2)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_size2)
        self.bn_dense_1 = nn.BatchNorm1d(num_features=240)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.dsep00(x); x = self.dsep01(x); x = self.dsep02(x); x = self.dsep03(x)
        x = x.permute(0, 2, 1); x = self.avg0(x); x = x.permute(0, 2, 1)

        x = self.dsep10(x); x = self.dsep11(x); x = self.dsep12(x); x = self.dsep13(x)
        x = x.permute(0, 2, 1); x = self.avg1(x); x = x.permute(0, 2, 1)

        x = self.dsep20(x); x = self.dsep21(x); x = self.dsep22(x); x = self.dsep23(x)
        x = x.permute(0, 2, 1); x = self.avg2(x); x = x.permute(0, 2, 1)

        x = self.dsep30(x); x = self.dsep31(x); x = self.dsep32(x); x = self.dsep33(x)
        x = x.permute(0, 2, 1); x = self.avg3(x); x = x.permute(0, 2, 1)

        x = torch.flatten(x, start_dim=1)

        x = self.dense1(x)
        x = self.bn_dense_1(x)
        x = self.relu(x)
        x = self.dense2(x)

        x = torch.squeeze(x)
        return x
