# trainval_stage2.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import RGBDTransparentDataset
from models import TOProgressiveMAEStage1, TOProgressiveMAEStage2
from refraction_saliency import compute_refraction_saliency
from metrics import compute_mae, compute_rmse, compute_rel, compute_delta1


def load_stage1(stage1_path, device="cuda"):
    model = TOProgressiveMAEStage1()
    ckpt = torch.load(stage1_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def visualize_examples(rgb, gt_depth, pred_depth, save_dir, prefix, max_num=4):
    """
    rgb: (B,3,H,W)
    gt_depth, pred_depth: (B,1,H,W)
    """
    os.makedirs(save_dir, exist_ok=True)
    B = rgb.size(0)
    num = min(B, max_num)

    rgb_np = rgb.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B,H,W,3)
    gt_np = gt_depth.detach().cpu().squeeze(1).numpy()        # (B,H,W)
    pred_np = pred_depth.detach().cpu().squeeze(1).numpy()    # (B,H,W)
    err_np = np.abs(pred_np - gt_np)

    for i in range(num):
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(rgb_np[i])
        axs[0].set_title("RGB")
        axs[0].axis("off")

        im1 = axs[1].imshow(gt_np[i], cmap="plasma")
        axs[1].set_title("GT Depth")
        axs[1].axis("off")
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        im2 = axs[2].imshow(pred_np[i], cmap="plasma")
        axs[2].set_title("Pred Depth")
        axs[2].axis("off")
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        im3 = axs[3].imshow(err_np[i], cmap="inferno")
        axs[3].set_title("Abs Error")
        axs[3].axis("off")
        plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fname = os.path.join(save_dir, f"{prefix}_sample_{i}.png")
        plt.savefig(fname, dpi=150)
        plt.close(fig)


def validate(stage1, stage2, dataloader, device="cuda", vis_dir=None, vis_prefix="val"):
    stage1.eval()
    stage2.eval()

    mae_list = []
    rmse_list = []
    rel_list = []
    delta1_list = []

    first_batch_rgb = None
    first_batch_gt = None
    first_batch_pred = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validating")):
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            s_ref = compute_refraction_saliency(rgb, depth)

            # 先拿到 stage1 的全局特征（虽然验证时主要用 stage2 的 inference_depth，可选）
            out_s1 = stage1(rgb, depth, s_ref)
            # 这里不直接用 out_s1，只是说明可以做联合分析

            # 使用我们在 models.py 中添加的 inference_depth
            pred_depth = stage2.inference_depth(rgb, depth, s_ref)

            # 只在 valid_mask 上计算指标
            mae = compute_mae(pred_depth, depth, valid_mask)
            rmse = compute_rmse(pred_depth, depth, valid_mask)
            rel = compute_rel(pred_depth, depth, valid_mask)
            delta1 = compute_delta1(pred_depth, depth, valid_mask)

            mae_list.append(mae.item())
            rmse_list.append(rmse.item())
            rel_list.append(rel.item())
            delta1_list.append(delta1.item())

            if i == 0:
                first_batch_rgb = rgb.clone()
                first_batch_gt = depth.clone()
                first_batch_pred = pred_depth.clone()

    mae_mean = sum(mae_list) / len(mae_list)
    rmse_mean = sum(rmse_list) / len(rmse_list)
    rel_mean = sum(rel_list) / len(rel_list)
    delta1_mean = sum(delta1_list) / len(delta1_list)

    print(f"Validation Results: "
          f"RMSE={rmse_mean:.4f}, MAE={mae_mean:.4f}, REL={rel_mean:.4f}, δ1={delta1_mean:.4f}")

    if vis_dir is not None and first_batch_rgb is not None:
        # 可视化第一批
        import numpy as np
        visualize_examples(first_batch_rgb, first_batch_gt, first_batch_pred,
                           save_dir=vis_dir, prefix=vis_prefix)

    return {
        "rmse": rmse_mean,
        "mae": mae_mean,
        "rel": rel_mean,
        "delta1": delta1_mean
    }


def train_stage2_with_val(
    base_root,
    stage1_path="stage1.pth",
    save_path="stage2.pth",
    dataset_name="ClearGrasp",
    batch_size=4,
    lr=1e-4,
    epochs=50,
    num_workers=4,
    device="cuda",
    val_interval=1,
    out_dir="outputs"
):
    os.makedirs(out_dir, exist_ok=True)

    # 构建 train / val 数据集
    train_set = RGBDTransparentDataset(
        base_root=base_root,
        dataset_name=dataset_name,
        split="train"
    )
    val_set = RGBDTransparentDataset(
        base_root=base_root,
        dataset_name=dataset_name,
        split="val"
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # 模型
    stage1 = load_stage1(stage1_path, device)
    stage2 = TOProgressiveMAEStage2(stage1_model=stage1)
    stage2.to(device)

    # 冻结 stage1
    for p in stage1.parameters():
        p.requires_grad = False

    optimizer = AdamW(stage2.parameters(), lr=lr, weight_decay=1e-4)

    best_rmse = float("inf")

    for epoch in range(1, epochs + 1):
        stage2.train()
        pbar = tqdm(train_loader, desc=f"[Stage2 Train] Epoch {epoch}/{epochs}")
        running_loss = 0.0
        for batch in pbar:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)

            optimizer.zero_grad()
            s_ref = compute_refraction_saliency(rgb, depth)

            with torch.no_grad():
                out_s1 = stage1(rgb, depth, s_ref)
                global_rgb_s1 = out_s1["global_rgb"]
                global_depth_s1 = out_s1["global_depth"]

            out_s2 = stage2(rgb, depth, s_ref, global_rgb_s1, global_depth_s1)
            loss = out_s2["loss"]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({
                "loss": running_loss / (len(train_loader)),
                "mae_loss": out_s2["loss_mae"],
                "distill": out_s2["loss_distill"]
            })

        # 每个 epoch 保存一次普通 checkpoint
        ckpt_path = os.path.join(out_dir, f"stage2_epoch{epoch}.pth")
        torch.save({
            "stage2": stage2.state_dict(),
            "stage1": stage1.state_dict(),
            "epoch": epoch
        }, ckpt_path)

        # 验证
        if epoch % val_interval == 0:
            vis_dir = os.path.join(out_dir, f"vis_epoch{epoch}")
            metrics = validate(stage1, stage2, val_loader, device=device,
                               vis_dir=vis_dir,
                               vis_prefix=f"{dataset_name}_epoch{epoch}")

            # 如果 RMSE 更低，则保存 best
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_path = os.path.join(out_dir, "stage2_best.pth")
                torch.save({
                    "stage2": stage2.state_dict(),
                    "stage1": stage1.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics
                }, best_path)
                print(f"==> New best RMSE {best_rmse:.4f}, saved to {best_path}")


if __name__ == "__main__":
    # Windows 路径建议用 r"..." 原始字符串
    base_root = r"E:\论文\我的\深度补全预训练模型\pythonProject1\databases"
    stage1_path = r"E:\论文\我的\深度补全预训练模型\pythonProject1\to_progressivemae\stage1.pth"

    # 先在 ClearGrasp 上做预训练 + 验证
    train_stage2_with_val(
        base_root=base_root,
        stage1_path=stage1_path,
        save_path="stage2.pth",  # 实际路径由 out_dir 里保存
        dataset_name="ClearGrasp",
        batch_size=4,
        lr=1e-4,
        epochs=50,
        num_workers=4,
        device="cuda",
        val_interval=1,
        out_dir=r"E:\论文\我的\深度补全预训练模型\pythonProject1\to_progressivemae\outputs_cleargrasp"
    )

    # 若想在 Trans10K 上继续训练 + 验证（例如跨数据集适应），可以再调用一次：
    # train_stage2_with_val(
    #     base_root=base_root,
    #     stage1_path=stage1_path,
    #     save_path="stage2_trans10k.pth",
    #     dataset_name="Trans10K",
    #     batch_size=4,
    #     lr=1e-4,
    #     epochs=50,
    #     num_workers=4,
    #     device="cuda",
    #     val_interval=1,
    #     out_dir=r"E:\论文\我的\深度补全预训练模型\pythonProject1\to_progressivemae\outputs_trans10k"
    # )
