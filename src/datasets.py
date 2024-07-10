import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision

from typing import Tuple
from termcolor import cprint


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        # if len(self.X.shape) == 3:
        #     print(self.X.shape)
        #     self.X = self.X.unsqueeze(1) # (b, c, h, w)
        # self.X = F.interpolate(self.X, size=(128, 128), mode="bilinear", align_corners=False)
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        # save image 10
        
        # for i in range(10):
        #     img = self.X[i].numpy()
        #     img = (img - img.min()) / (img.max() - img.min())
        #     plt.imshow(img, cmap="gray")
        #     # y label
        #     plt.title(f"y: {self.y[i]}")
        #     plt.savefig(f"img_{i}.png")
        #     plt.close()
        

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
    def get_len_ids(self):
        return self.subject_idxs.max() + 1
    
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]