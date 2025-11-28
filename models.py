# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

from utils import patchify, unpatchify, PositionalEncoding1D
from losses import weighted_info_nce_loss, MultiTaskMAELoss, distillation_loss
from refraction_saliency import patch_saliency_from_map


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B,C,H,W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, L, C)
        return x, (H, W)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=6, num_heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)  # (B,L,C)


class VisionEncoder(nn.Module):
    """简单的 ViT-like Encoder，用于 RGB 或 Depth"""

    def __init__(self, in_chans=3, embed_dim=384, patch_size=16,
                 depth=6, num_heads=6, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.pos_embed = PositionalEncoding1D(embed_dim)
        self.encoder = TransformerEncoder(embed_dim, depth, num_heads, mlp_ratio, drop)

    def forward(self, x):
        tokens, (Hp, Wp) = self.patch_embed(x)
        tokens = self.pos_embed(tokens)
        feats = self.encoder(tokens)
        global_feat = feats.mean(dim=1)  # global pooled
        return feats, global_feat, (Hp, Wp)  # (B,L,C),(B,C),(Hp,Wp)


# ----------------------
# Stage 1: Contrastive
# ----------------------

class TOProgressiveMAEStage1(nn.Module):
    def __init__(self,
                 img_channels=3,
                 depth_channels=1,
                 embed_dim=384,
                 patch_size=16,
                 depth_layers=6,
                 rgb_layers=6,
                 num_heads=6,
                 temperature=0.07,
                 refraction_alpha=2.0):
        super().__init__()
        self.rgb_encoder = VisionEncoder(
            in_chans=img_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depth=rgb_layers,
            num_heads=num_heads
        )
        self.depth_encoder = VisionEncoder(
            in_chans=depth_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depth=depth_layers,
            num_heads=num_heads
        )
        self.proj_rgb = nn.Linear(embed_dim, embed_dim)
        self.proj_depth = nn.Linear(embed_dim, embed_dim)
        self.temperature = temperature
        self.refraction_alpha = refraction_alpha
        self.patch_size = patch_size

    def forward(self, rgb, depth, s_ref):
        """
        rgb: (B,3,H,W)
        depth: (B,1,H,W)
        s_ref: (B,1,H,W) 折射显著性图
        """
        B, _, H, W = rgb.shape
        # 编码
        feats_rgb, global_rgb, _ = self.rgb_encoder(rgb)
        feats_depth, global_depth, _ = self.depth_encoder(depth)

        # 投影 + L2 norm
        z_rgb = F.normalize(self.proj_rgb(feats_rgb), dim=-1)
        z_depth = F.normalize(self.proj_depth(feats_depth), dim=-1)

        # patch-level 折射显著性
        s_patch = patch_saliency_from_map(s_ref, self.patch_size)  # (B,L)
        w_patch = 1.0 + self.refraction_alpha * s_patch  # 放大透明区域权重

        # 展平为 (B*L, C)
        B, L, C = z_rgb.shape
        z_rgb_flat = z_rgb.reshape(B * L, C)
        z_depth_flat = z_depth.reshape(B * L, C)
        w_flat = w_patch.reshape(B * L)

        loss_contrastive = weighted_info_nce_loss(
            z_rgb_flat, z_depth_flat, w_flat, self.temperature
        )

        return {
            "loss": loss_contrastive,
            "loss_contrastive": loss_contrastive,
            "global_rgb": global_rgb,
            "global_depth": global_depth,
        }


# ----------------------
# Stage 2: MAE + Distillation
# ----------------------

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=384, decoder_dim=256, depth=4, num_heads=4, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(embed_dim, decoder_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=drop,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.depth_head = nn.Linear(decoder_dim, 1)  # 输出每个 patch 的均值深度（示例）
        self.sem_head = nn.Linear(decoder_dim, 256)  # 与 ViT 语义特征维度对齐
        self.noise_head = nn.Linear(decoder_dim, embed_dim)  # 预测噪声（token 维度）

    def forward(self, tokens, pos_embed):
        x = self.proj(tokens + pos_embed)
        x = self.decoder(x)
        depth_pred = self.depth_head(x)   # (B,L,1)
        sem_pred = self.sem_head(x)      # (B,L,256)
        noise_pred = self.noise_head(x)  # (B,L,E)
        return depth_pred, sem_pred, noise_pred


class TOProgressiveMAEStage2(nn.Module):
    def __init__(self,
                 stage1_model: TOProgressiveMAEStage1,
                 patch_size=16,
                 base_mask_rate=0.8,
                 beta_mask=1.0,
                 noise_std=0.1,
                 lambda_depth=1.0,
                 lambda_sem=1.0,
                 lambda_denoise=0.1,
                 lambda_distill=0.1):
        super().__init__()
        # Stage1 编码器作为初始化
        self.rgb_encoder = stage1_model.rgb_encoder
        self.depth_encoder = stage1_model.depth_encoder

        self.patch_size = patch_size
        self.base_mask_rate = base_mask_rate
        self.beta_mask = beta_mask
        self.noise_std = noise_std
        self.lambda_distill = lambda_distill

        # decoder
        self.pos_embed = PositionalEncoding1D(self.depth_encoder.patch_embed.proj.out_channels)
        self.decoder = MAEDecoder(embed_dim=self.depth_encoder.patch_embed.proj.out_channels)

        # 冻结的 ViT 提取 RGB 语义特征
        self.rgb_sem_encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        for p in self.rgb_sem_encoder.parameters():
            p.requires_grad = False

        # 损失
        self.mae_loss = MultiTaskMAELoss(
            lambda_depth=lambda_depth,
            lambda_sem=lambda_sem,
            lambda_denoise=lambda_denoise
        )

    @torch.no_grad()
    def extract_rgb_semantic(self, rgb):
        """
        使用预训练 ViT 提取 patch-level 语义特征
        输出 (B, L_sem, 256) 这里我们取中间层/cls token 等，做一个简化版本：
        为了简洁，直接使用 vit 的 cls token 作为全局特征，然后 repeat 到 patch 上。
        更严谨的话可以取 vit 的 patch embedding。
        """
        self.rgb_sem_encoder.eval()
        # ViT 输入必须是 224x224
        rgb_resized = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        # torchvision 的 ViT 预处理（简单版）
        x = rgb_resized
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        feats = self.rgb_sem_encoder(x)  # (B,1000) 分类 logits，简单起见当作 1000-d 特征
        # 压到 256 维
        proj = nn.functional.normalize(feats, dim=-1)
        proj = proj[:, :256]  # (B,256) 截断
        # repeat 到 patch 数
        return proj  # (B,256)

    def refraction_guided_mask(self, s_ref, H, W):
        """
        s_ref: (B,1,H,W) ∈[0,1]
        返回 mask: (B,L), 1=masked, 0=visible
        Pmask = base + beta * s_ref
        """
        B = s_ref.size(0)
        s_patch = patch_saliency_from_map(s_ref, self.patch_size)  # (B,L)
        # 归一化一下
        s_norm = (s_patch - s_patch.min(dim=1, keepdim=True)[0]) / \
                 (s_patch.max(dim=1, keepdim=True)[0] - s_patch.min(dim=1, keepdim=True)[0] + 1e-6)

        p_mask = self.base_mask_rate + self.beta_mask * s_norm
        p_mask = torch.clamp(p_mask, 0.0, 0.95)  # 防止全遮
        rand = torch.rand_like(p_mask)
        mask = (rand < p_mask).float()  # 1=masked
        return mask  # (B,L)

    def forward(self, rgb, depth, s_ref,
                global_rgb_stage1, global_depth_stage1):
        """
        rgb: (B,3,H,W)
        depth: (B,1,H,W)
        s_ref: (B,1,H,W)
        global_*_stage1: 来自 Stage1 的全局特征，用于蒸馏
        """
        B, _, H, W = depth.shape

        # Depth patch 化，作为 GT
        depth_patches = patchify(depth, self.patch_size)  # (B,L,P_dim)
        B, L, P_dim = depth_patches.shape

        # refraction-guided masking
        mask = self.refraction_guided_mask(s_ref, H, W)  # (B,L)
        visible_mask = 1.0 - mask  # 1=visible

        # 编码 depth（只对 visible token 加噪声后送 encoder）
        depth_tokens, _, _ = self.depth_encoder.patch_embed(depth)
        pos = self.pos_embed(depth_tokens)

        noise = torch.randn_like(depth_tokens) * self.noise_std
        depth_tokens_noisy = depth_tokens + noise

        # visible token 选择
        visible_tokens = depth_tokens_noisy * visible_mask.unsqueeze(-1)
        # 为了让 transformer 知道哪些是 padding，我们仍然把全部 token 输入，
        # 但 masked token 的输入设为 0 并依赖 self-attention 的上下文恢复。
        feats_depth = self.depth_encoder.encoder(
            visible_tokens + pos
        )  # (B,L,C)

        # decoder 重建全部 patch
        depth_pred_patch, sem_pred, noise_pred = self.decoder(feats_depth, pos)

        # 语义 GT（简化为全局 256-d，repeat 到每个 patch）
        sem_global = self.extract_rgb_semantic(rgb)  # (B,256)
        sem_gt = sem_global.unsqueeze(1).repeat(1, L, 1)  # (B,L,256)

        # 去噪 GT
        noise_gt = noise  # (B,L,C)

        # MAE 多任务损失
        loss_mae, loss_dict = self.mae_loss(
            pred_depth_patches=depth_pred_patch,
            gt_depth_patches=depth_patches,
            mask=mask,
            pred_sem_features=sem_pred,
            gt_sem_features=sem_gt,
            pred_noise=noise_pred,
            gt_noise=noise_gt
        )

        # Stage2 的全局特征
        _, global_rgb2, _ = self.rgb_encoder(rgb)
        _, global_depth2, _ = self.depth_encoder(depth)

        # 蒸馏损失
        loss_distill_rgb = distillation_loss(global_rgb_stage1, global_rgb2, beta=self.lambda_distill)
        loss_distill_depth = distillation_loss(global_depth_stage1, global_depth2, beta=self.lambda_distill)
        loss_distill_total = loss_distill_rgb + loss_distill_depth

        loss_total = loss_mae + loss_distill_total

        return {
            "loss": loss_total,
            "loss_mae": loss_mae.item(),
            "loss_depth": loss_dict["loss_depth"],
            "loss_sem": loss_dict["loss_sem"],
            "loss_denoise": loss_dict["loss_denoise"],
            "loss_distill": loss_distill_total.item()
        }

# models.py 中 TOProgressiveMAEStage2 类里，末尾加上：

    @torch.no_grad()
    def inference_depth(self, rgb, depth, s_ref):
        """
        验证 / 可视化用：不做随机 masking，只做一个“低 mask 率”的前向，
        并返回重建的深度图(pred_depth_img)。

        rgb: (B,3,H,W)
        depth: (B,1,H,W) 仅用于 patchify 的 H,W、也可以不传真实 GT
        s_ref: (B,1,H,W)
        return:
            pred_depth_img: (B,1,H,W)
        """
        self.eval()
        B, _, H, W = depth.shape

        # 和训练时一样的 patchify，只是我们这里不关心 GT 的 patch，只需要形状
        depth_patches = patchify(depth, self.patch_size)  # (B,L,P_dim)
        B, L, P_dim = depth_patches.shape

        # 这里我们将 mask 率固定为一个很小值（例如 0.1），尽量让 decoder
        # 有信息可用，同时保留一点 MAE 的形式；如果你想完全不 mask，也可以
        # 直接把 mask 全设为 0。
        base_mask_rate_backup = self.base_mask_rate
        self.base_mask_rate = 0.1  # 可自行调小或改为 0.0
        mask = self.refraction_guided_mask(s_ref, H, W)  # (B,L)
        self.base_mask_rate = base_mask_rate_backup

        visible_mask = 1.0 - mask

        # depth 编码
        depth_tokens, _, _ = self.depth_encoder.patch_embed(depth)
        pos = self.pos_embed(depth_tokens)

        # 验证时一般不加噪声
        depth_tokens_noisy = depth_tokens  # + 0

        visible_tokens = depth_tokens_noisy * visible_mask.unsqueeze(-1)

        feats_depth = self.depth_encoder.encoder(
            visible_tokens + pos
        )  # (B,L,C)

        depth_pred_patch, _, _ = self.decoder(feats_depth, pos)  # (B,L,1)

        # 把 (B,L,1) 展开为 patch 格式，简单起见：每个 patch 用同一深度值
        depth_pred_patch = depth_pred_patch.view(B, L, 1)
        # 扩展到 patch 内所有像素
        depth_pred_patch_expand = depth_pred_patch.repeat(1, 1, P_dim)  # (B,L,P_dim)

        pred_depth_img = unpatchify(depth_pred_patch_expand, self.patch_size, H, W)  # (B,1,H,W)
        return pred_depth_img
