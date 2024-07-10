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
        
        self.dropout = nn.Dropout(0.1)
        
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
                X = self.dropout(X)
                _h = self.id_encoder(self.one_hot(subject_idxs, self.len_ids))
                X = torch.cat([X, _h], dim=-1)
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