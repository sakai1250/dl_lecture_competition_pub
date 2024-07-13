import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


# conda activate dlbasics

class Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        len_ids: int,
        hid_dim: int = 64,
    ) -> None:
        super().__init__()
        
        self.len_ids = len_ids
        
        self.basicconv = BasicConvClassifier(num_classes, seq_len, in_channels, hid_dim)
        self.basicconv.load_state_dict(torch.load("model_best.pt"))
        
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b c h -> b (c h)"),
            nn.Linear(hid_dim+hid_dim, num_classes),
        )
        
        self.id_encoder = nn.Linear(len_ids, hid_dim)
        
        self.dropout = nn.Dropout(0.2)
        
        with torch.no_grad():
            self.head[-1].weight.data[:, :hid_dim] = self.basicconv.head[-1].weight.data
            self.head[-1].bias.data = self.basicconv.head[-1].bias.data
        
    def one_hot(self, ids: torch.Tensor, len_ids: int) -> torch.Tensor:
        """_summary_
        Args:
            ids ( b, 1 ): _description_
            len_ids: _description_
        Returns:
            one_hot ( b, num_classes ): _description_
        """
        return F.one_hot(ids, num_classes=self.len_ids).float()

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, h, w ): _description_
            subject_idxs ( b, len_ids ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                _h = self.id_encoder(self.one_hot(subject_idxs, self.len_ids))
                X = torch.cat([X, _h], dim=-1)
                X = layer(X)
                X = self.dropout(X)
            else:
                X = layer(X)
        return X

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 64
    ) -> None:
        super().__init__()
        
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )
        self.skip = ConvBlock(in_channels, hid_dim)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b c h -> b (c h)"),
            nn.Linear(hid_dim, num_classes),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, h, w ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """

        X = self.blocks(X) + self.skip(X)
        X = self.dropout(X)

        return self.head(X)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.20,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        #self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

        self._initialize_weights()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if len(X.shape) == 4:
            X = Rearrange("b c h w -> b c (h w)")(X)
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.relu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.relu(self.batchnorm1(X))

        # X = self.conv2(X)
        #X = F.glu(X, dim=-2)

        return self.dropout(X)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                



# from .SeparableConv import SeparableConv1d

# import torch.optim as optim

# class EEGNet(nn.Module):
#     def __init__(self, nb_classes: int, Chans: int = 128, Samples: int = 128,
#                  dropoutRate: float = 0.2, kernLength: int = 63,
#                  F1:int = 8, D:int = 2):
#         super().__init__()

#         F2 = F1 * D

#         # Make kernel size and odd number
#         try:
#             assert kernLength % 2 != 0
#         except AssertionError:
#             raise ValueError("ERROR: kernLength must be odd number")

#         # In: (B, Chans, Samples, 1)
#         # Out: (B, F1, Samples, 1)
#         self.conv1 = nn.Conv1d(Chans, F1, kernLength, padding=(kernLength // 2))
#         self.bn1 = nn.BatchNorm1d(F1) # (B, F1, Samples, 1)
#         # In: (B, F1, Samples, 1)
#         # Out: (B, F2, Samples - Chans + 1, 1)
#         self.conv2 = nn.Conv1d(F1, F2, Chans, groups=F1)
#         self.bn2 = nn.BatchNorm1d(F2) # (B, F2, Samples - Chans + 1, 1)
#         # In: (B, F2, Samples - Chans + 1, 1)
#         # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
#         self.avg_pool = nn.AvgPool1d(4)
#         self.dropout = nn.Dropout(dropoutRate)

#         # In: (B, F2, (Samples - Chans + 1) / 4, 1)
#         # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
#         self.conv3 = SeparableConv1d(F2, F2, kernel_size=15, padding=7)
#         self.bn3 = nn.BatchNorm1d(F2)
#         # In: (B, F2, (Samples - Chans + 1) / 4, 1)
#         # Out: (B, F2, (Samples - Chans + 1) / 32, 1)
#         self.avg_pool2 = nn.AvgPool1d(8)
#         # In: (B, F2 *  (Samples - Chans + 1) / 32)
#         self.fc = nn.Linear(32, nb_classes)

#     def forward(self, x: torch.Tensor):
#         # Block 1
#         y1 = self.conv1(x)
#         #print("conv1: ", y1.shape)
#         y1 = self.bn1(y1)
#         #print("bn1: ", y1.shape)
#         y1 = self.conv2(y1)
#         #print("conv2", y1.shape)
#         y1 = F.relu(self.bn2(y1))
#         #print("bn2", y1.shape)
#         y1 = self.avg_pool(y1)
#         #print("avg_pool", y1.shape)
#         y1 = self.dropout(y1)
#         #print("dropout", y1.shape)

#         # Block 2
#         y2 = self.conv3(y1)
#         #print("conv3", y2.shape)
#         y2 = F.relu(self.bn3(y2))
#         #print("bn3", y2.shape)
#         y2 = self.dropout(y2)
#         #print("dropout", y2.shape)
#         y2 = torch.flatten(y2, 1)
#         #print("flatten", y2.shape)
#         y2 = self.fc(y2)
#         #print("fc", y2.shape)

#         return y2


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BrainDecoder(nn.Module):
    def __init__(self, num_subjects, num_channels, num_output_features):
        super().__init__()
        self.D1 = 281  # 空間的注意層の出力チャネル数
        
        self.spatial_attention = SpatialAttention(num_channels, self.D1)
        self.subject_layer = SubjectLayer(num_subjects, self.D1)
        
        self.conv_blocks = nn.Sequential(
            ConvBlock(self.D1, 320),
            ConvBlock(320, 320),
            ConvBlock(320, 320),
            ConvBlock(320, 320),
            ConvBlock(320, 320)
        )
        
        self.output_layer = nn.Sequential(
            nn.Conv1d(320, 640, 1),
            nn.GELU(),
            nn.Conv1d(640, num_output_features, 1),
            nn.AdaptiveAvgPool1d(1)  # 時間軸方向の平均をとる
        )

    def forward(self, x, subject_idx=None):
        x = self.spatial_attention(x)
        if subject_idx is not None:
            x = self.subject_layer(x, subject_idx)
        x = self.conv_blocks(x)
        x = self.output_layer(x)
        return x.squeeze(-1)  # 最後の次元（時間軸）を削除

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, K=32):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fourier_features = nn.Parameter(torch.randn(out_channels, K, K, 4))
        
        # センサー位置を学習可能なパラメータとして設定（入力チャンネル数に基づく）
        self.sensor_positions = nn.Parameter(torch.rand(in_channels, 2))
        
        # 入力チャンネル数を出力チャンネル数に変換する1x1畳み込み層
        self.channel_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, _, seq_len = x.shape
        
        # フーリエ特徴量の計算
        freqs_x = 2 * math.pi * torch.arange(self.K, device=x.device).float() / self.K
        freqs_y = 2 * math.pi * torch.arange(self.K, device=x.device).float() / self.K
        
        # センサー位置を拡張 [in_channels, 2] -> [in_channels, K, K, 2]
        sensor_pos_expanded = self.sensor_positions.unsqueeze(1).unsqueeze(1).expand(-1, self.K, self.K, -1)
        
        # 周波数を拡張 [K] -> [1, K, K, 2]
        freqs_expanded = torch.stack(torch.meshgrid(freqs_x, freqs_y, indexing='ij'), dim=-1).unsqueeze(0)
        
        # 特徴マップの計算
        feature_map = sensor_pos_expanded * freqs_expanded
        feature_map = torch.cat([torch.sin(feature_map), torch.cos(feature_map)], dim=-1)
        
        # バッチサイズに合わせて拡張
        feature_map = feature_map.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        
        # 注意力の計算
        attention = torch.einsum('bckhw,okhw->boc', feature_map, self.fourier_features)
        attention = F.softmax(attention, dim=1)
        # 入力チャンネル数を出力チャンネル数に変換
        x = self.channel_proj(x)
        
        # 注意力の適用
        out = torch.einsum('bci,bio->bco', x, attention)
        return out
class SubjectLayer(nn.Module):
    def __init__(self, num_subjects, num_channels):
        super().__init__()
        self.subject_convs = nn.ModuleList([
            nn.Conv1d(num_channels, num_channels, 1, bias=False) 
            for _ in range(num_subjects)
        ])

    def forward(self, x, subject_idx):
        batch_size, num_channels, seq_len = x.shape
        output = torch.zeros_like(x)
        for i in range(batch_size):
            subject = subject_idx[i].item()
            output[i] = self.subject_convs[subject](x[i].unsqueeze(0)).squeeze(0)
        return output

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(out_channels, out_channels * 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.conv3(x)
        x = F.glu(x, dim=1)
        if x.shape[1] == residual.shape[1]:
            x = x + residual
        return x

# # モデルの使用例
# num_subjects = 4
# num_channels = 271
# num_output_features = 1854
# model = BrainDecoder(num_subjects, num_channels, num_output_features)

# # ダミーデータの作成
# batch_size = 32
# time_steps = 281
# x = torch.randn(batch_size, num_channels, time_steps)
# subject_idx = torch.randint(0, num_subjects, (batch_size,))

# # モデルの実行
# output = model(x, subject_idx)
# print(output.shape) 