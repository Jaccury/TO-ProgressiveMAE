# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def patchify(img: torch.Tensor, patch_size: int):
    """
    img: (B, C, H, W)
    return: patches (B, L, C * P * P), where L = (H/P)*(W/P)
    """
    B, C, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        "H and W must be divisible by patch_size"
    ph = H // patch_size
    pw = W // patch_size
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # (B, C, ph, pw, P, P)
    patches = patches.contiguous().view(B, C, ph * pw, patch_size * patch_size)
    patches = patches.permute(0, 2, 1, 3).contiguous()  # (B, L, C, P*P)
    patches = patches.view(B, ph * pw, C * patch_size * patch_size)
    return patches


def unpatchify(patches: torch.Tensor, patch_size: int, H: int, W: int):
    """
    patches: (B, L, C * P * P)
    return: img (B, C, H, W)
    """
    B, L, D = patches.shape
    C = D // (patch_size * patch_size)
    ph = H // patch_size
    pw = W // patch_size
    assert L == ph * pw
    patches = patches.view(B, ph, pw, C, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    img = patches.view(B, C, H, W)
    return img


class PositionalEncoding1D(nn.Module):
    """简单的 1D 正弦位置编码，用于 patch 序列"""

    def __init__(self, dim, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]
