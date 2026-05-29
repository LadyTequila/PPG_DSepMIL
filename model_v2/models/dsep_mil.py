import torch
from torch import nn

from ._dsep_block import _RVarDSepBlock


class _SmoothedGatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL + 对 attention score 做局部平滑。
    motivation: 呼吸事件持续 10~60 秒，注意力权重应该是"连续的波段"
                而不是离散的尖峰。softmax 前对 score 加 AvgPool1d 强约束连续性。
    """
    def __init__(self, instance_dim, attn_hidden=64, dropout=0.1, smooth_window=5):
        super().__init__()
        self.V = nn.Linear(instance_dim, attn_hidden)
        self.U = nn.Linear(instance_dim, attn_hidden)
        self.W = nn.Linear(attn_hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.smooth_pool = nn.AvgPool1d(
            kernel_size=smooth_window, stride=1, padding=smooth_window // 2
        )

    def forward(self, H):
        # H: (B, N, D)
        v = torch.tanh(self.V(H))
        u = torch.sigmoid(self.U(H))
        gated = self.dropout(v * u)
        score = self.W(gated).squeeze(-1)                 # (B, N)

        score_smoothed = self.smooth_pool(score.unsqueeze(1)).squeeze(1)
        score_smoothed = score_smoothed[:, :score.size(-1)]   # 对齐可能的 +1 padding

        w = torch.softmax(score_smoothed, dim=-1)              # (B, N)
        z = torch.sum(w.unsqueeze(-1) * H, dim=1)              # (B, D)
        return z, w


class DSepMIL(nn.Module):
    """
    改进版 DSep + MIL：
      1. Backbone 只保留前 2 阶段（输出 16 通道 × 60 时间步），避免 4 维瓶颈
      2. Instance 升维到 d_model=64，给 attention 足够的表达空间
      3. Smoothed Gated Attention Pooling（借鉴 MultiScaleTransformerMIL 的思路）
         强制 attention 产生连续波段，契合呼吸事件的物理本质
    配合 ComboFocalF1Loss 使用以抑制 mode collapse。
    """

    def __init__(self, num_channels=7, winsize=60,
                 d_model=64, attn_hidden=64, smooth_window=5, dropout=0.3,
                 hidden_size=64, hidden_size2=4, layer=2):
        super().__init__()

        in_ch = num_channels

        # ----- Backbone: 2 stages（16 通道瓶颈）-----
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

        # ----- Instance projection -----
        instance_in_dim = 16
        self.instance_proj = nn.Linear(instance_in_dim, d_model)
        self.instance_bn = nn.BatchNorm1d(d_model)

        # ----- MIL pooling -----
        self.mil = _SmoothedGatedAttentionMIL(
            instance_dim=d_model,
            attn_hidden=attn_hidden,
            dropout=dropout,
            smooth_window=smooth_window,
        )

        # ----- Classifier -----
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.last_attention = None

    def _backbone(self, x):
        x = x.permute(0, 2, 1)

        x = self.dsep00(x); x = self.dsep01(x); x = self.dsep02(x); x = self.dsep03(x)
        x = x.permute(0, 2, 1); x = self.avg0(x); x = x.permute(0, 2, 1)

        x = self.dsep10(x); x = self.dsep11(x); x = self.dsep12(x); x = self.dsep13(x)
        x = x.permute(0, 2, 1); x = self.avg1(x); x = x.permute(0, 2, 1)

        # (B, 16, 60) -> (B, 60, 16): 60 instances × 16-dim feature
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        H = self._backbone(x)                                # (B, 60, 16)

        B, N, D = H.shape
        H = self.instance_proj(H)                            # (B, 60, d_model)
        H = self.instance_bn(H.reshape(B * N, -1)).reshape(B, N, -1)
        H = self.relu(H)
        H = self.dropout(H)

        z, attn = self.mil(H)                                # z: (B, d_model)
        self.last_attention = attn.detach()

        out = self.classifier(z)                             # (B, 1)
        return out.squeeze(-1)
