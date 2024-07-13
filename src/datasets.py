import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchaudio

import random


from typing import Tuple
from termcolor import cprint


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", preprocess=True, augment=True) -> None:
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

        self.preprocess = preprocess
        self.augment = augment
        

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        # フーリエ変換
        fft = torch.fft.rfft(data, dim=-1)
        
        # 周波数軸
        freq = torch.fft.rfftfreq(data.shape[-1], d=1/fs)
        
        # バンドパスフィルタの周波数応答
        h = torch.zeros_like(freq)
        h[(freq >= low) & (freq <= high)] = 1
        
        # フィルタリング
        fft_filtered = fft * h
        
        # 逆フーリエ変換
        filtered = torch.fft.irfft(fft_filtered, n=data.shape[-1], dim=-1)
        
        return filtered

    def preprocess_meg(self, meg_data):
        # リサンプリング（必要な場合）
        if self.seq_len != 281:  # 元のseq_lenが281でない場合
            meg_data = torch.nn.functional.interpolate(meg_data.unsqueeze(0), size=281, mode='linear', align_corners=False).squeeze(0)

        # バンドパスフィルタ（0.5Hz-50Hz）
        meg_data = self.bandpass_filter(meg_data, 0.5, 50, 200)

        # スケーリング
        meg_data = meg_data * 0.5  # スケーリング係数は必要に応じて調整

        # ベースライン補正
        window_size = 41  # 奇数にして、中心を明確に
        pad = window_size // 2
        padded = torch.nn.functional.pad(meg_data, (pad, pad), mode='reflect')
        moving_avg = torch.nn.functional.avg_pool1d(padded.unsqueeze(0), kernel_size=window_size, stride=1).squeeze(0)
        
        # サイズを確認して調整
        if moving_avg.shape[-1] != meg_data.shape[-1]:
            moving_avg = moving_avg[:, :meg_data.shape[-1]]
        
        meg_data = meg_data - moving_avg

        return meg_data
    
    def augment_meg(self, meg_data):
        # ランダムなノイズ追加
        if random.random() < 0.1:
            noise = torch.randn_like(meg_data) * random.uniform(0.01, 0.05)
            meg_data = meg_data + noise

        # ランダムな時間シフト
        if random.random() < 0.1:
            shift = random.randint(-10, 10)
            meg_data = torch.roll(meg_data, shifts=shift, dims=-1)

        # ランダムなスケーリング
        if random.random() < 0.1:
            scale = random.uniform(0.9, 1.1)
            meg_data = meg_data * scale

        # ランダムな反転
        if random.random() < 0.1:
            meg_data = torch.flip(meg_data, [-1])

        # チャンネルのランダムな入れ替え
        if random.random() < 0.1:
            num_channels = meg_data.shape[0]
            perm = torch.randperm(num_channels)
            meg_data = meg_data[perm]

        return meg_data

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        # if self.preprocess:
        #     self.X[i] = self.preprocess_meg(self.X[i])
        # if self.augment:
        #     self.X[i] = self.augment_meg(self.X[i])
            
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