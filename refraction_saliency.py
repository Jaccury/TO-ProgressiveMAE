# refraction_saliency.py
import cv2
import numpy as np
import torch


def compute_refraction_saliency(rgb: torch.Tensor, depth: torch.Tensor):
    """
    计算折射显著性图 S_ref
    rgb: (B, 3, H, W), [0,1]
    depth: (B, 1, H, W), 任意单位
    return:
        s_ref: (B, 1, H, W), [0,1]
    """
    rgb_np = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    depth_np = depth.detach().cpu().numpy()

    B, _, H, W = rgb_np.shape
    out = np.zeros((B, 1, H, W), dtype=np.float32)

    for b in range(B):
        img = rgb_np[b].transpose(1, 2, 0)  # H,W,3
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 边缘（玻璃边界、高光区域）
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

        # Sobel 梯度
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_mag = grad_mag / (grad_mag.max() + 1e-6)

        # 高亮区域（高亮 + 低纹理）
        gray_norm = gray.astype(np.float32) / 255.0
        mean = cv2.blur(gray_norm, (7, 7))
        sqmean = cv2.blur(gray_norm ** 2, (7, 7))
        var = sqmean - mean ** 2
        var = np.clip(var, 0, None)
        texture = var / (var.max() + 1e-6)

        # 纹理低 + 亮度高 ⇒ 有可能是玻璃大片
        highlight = gray_norm * (1 - texture)
        highlight = highlight / (highlight.max() + 1e-6)

        # 深度缺失/异常（透明区域常见）
        d = depth_np[b, 0]
        invalid = (d <= 0) | np.isnan(d) | np.isinf(d)
        invalid = invalid.astype(np.float32)

        # 融合
        s = 0.4 * edges + 0.3 * grad_mag + 0.2 * highlight + 0.1 * invalid
        s = s / (s.max() + 1e-6)
        out[b, 0] = s

    return torch.from_numpy(out).to(rgb.device)


def patch_saliency_from_map(s_ref: torch.Tensor, patch_size: int):
    """
    将像素级 S_ref 汇聚到 patch 上
    s_ref: (B, 1, H, W)
    return: (B, L) patch-level saliency in [0,1]
    """
    B, _, H, W = s_ref.shape
    assert H % patch_size == 0 and W % patch_size == 0
    ph = H // patch_size
    pw = W // patch_size
    s = s_ref.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # (B,1,ph,pw,P,P)
    s = s.contiguous().view(B, 1, ph * pw, patch_size * patch_size)
    s = s.mean(-1)  # (B,1,L)
    s = s.view(B, ph * pw)
    return s  # (B, L)
