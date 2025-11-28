# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_info_nce_loss(z_rgb, z_depth, weights, temperature=0.07):
    """
    加权 InfoNCE：
    z_rgb, z_depth: (B*L, C) L2-normalized
    weights: (B*L,) in [1, +∞)，由折射显著性放大透明区域权重
    """
    # 计算相似度矩阵 (N,N)
    sim = z_rgb @ z_depth.t()  # cosine，因为已经 L2 norm
    sim = sim / temperature
    N = sim.size(0)

    # 正样本是对角线
    labels = torch.arange(N, device=sim.device)

    log_prob_rgb2depth = F.log_softmax(sim, dim=1)
    log_prob_depth2rgb = F.log_softmax(sim.t(), dim=1)

    weight = weights / (weights.mean() + 1e-6)
    loss_rgb2depth = -(weight * log_prob_rgb2depth[torch.arange(N), labels]).mean()
    loss_depth2rgb = -(weight * log_prob_depth2rgb[torch.arange(N), labels]).mean()

    return 0.5 * (loss_rgb2depth + loss_depth2rgb)


class MultiTaskMAELoss(nn.Module):
    """
    Stage 2：多任务 MAE 损失
    - 深度重建 L1
    - RGB 语义特征重建 MSE
    - 去噪 MSE
    """

    def __init__(self,
                 lambda_depth=1.0,
                 lambda_sem=1.0,
                 lambda_denoise=0.1):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.lambda_sem = lambda_sem
        self.lambda_denoise = lambda_denoise
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self,
                pred_depth_patches, gt_depth_patches, mask,
                pred_sem_features=None, gt_sem_features=None,
                pred_noise=None, gt_noise=None):
        """
        pred_depth_patches, gt_depth_patches: (B, L, P_dim)
        mask: (B, L) 1 表示 masked patch（只在这些位置计算深度重建）
        语义特征、噪声在 visible patch 上计算
        """
        loss = 0.0

        # 深度重建：只在 masked 位置
        if self.lambda_depth > 0:
            mask_ = mask.unsqueeze(-1)  # (B, L, 1)
            pred_m = pred_depth_patches * mask_
            gt_m = gt_depth_patches * mask_
            denom = mask_.sum() * gt_depth_patches.size(-1) + 1e-6
            loss_depth = torch.abs(pred_m - gt_m).sum() / denom
            loss = loss + self.lambda_depth * loss_depth
        else:
            loss_depth = torch.tensor(0.0, device=gt_depth_patches.device)

        # 语义特征重建（只在 visible 部分）
        if pred_sem_features is not None and gt_sem_features is not None and self.lambda_sem > 0:
            visible_mask = (1.0 - mask).unsqueeze(-1)
            pred_v = pred_sem_features * visible_mask
            gt_v = gt_sem_features * visible_mask
            denom = visible_mask.sum() * gt_sem_features.size(-1) + 1e-6
            loss_sem = ((pred_v - gt_v) ** 2).sum() / denom
            loss = loss + self.lambda_sem * loss_sem
        else:
            loss_sem = torch.tensor(0.0, device=gt_depth_patches.device)

        # 去噪
        if pred_noise is not None and gt_noise is not None and self.lambda_denoise > 0:
            loss_denoise = self.mse(pred_noise, gt_noise)
            loss = loss + self.lambda_denoise * loss_denoise
        else:
            loss_denoise = torch.tensor(0.0, device=gt_depth_patches.device)

        return loss, {
            "loss_depth": loss_depth.item(),
            "loss_sem": loss_sem.item(),
            "loss_denoise": loss_denoise.item()
        }


def distillation_loss(f_teacher, f_student, beta=1.0):
    """
    跨阶段特征蒸馏 (Smooth L1)
    f_teacher, f_student: (B, C)
    """
    return beta * F.smooth_l1_loss(f_student, f_teacher.detach())
