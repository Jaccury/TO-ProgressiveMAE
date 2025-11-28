# train_stage1.py
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from datasets import RGBDTransparentDataset
from models import TOProgressiveMAEStage1
from refraction_saliency import compute_refraction_saliency


def train_stage1(
    rgb_dir,
    depth_dir,
    save_path="stage1.pth",
    batch_size=4,
    lr=1e-4,
    epochs=50,
    device="cuda"
):
    dataset = RGBDTransparentDataset(rgb_dir, depth_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = TOProgressiveMAEStage1()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Stage1 Epoch {epoch+1}/{epochs}")
        avg_loss = 0.0
        for batch in pbar:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            optimizer.zero_grad()

            # 折射显著性图
            s_ref = compute_refraction_saliency(rgb, depth)

            out = model(rgb, depth, s_ref)
            loss = out["loss"]
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            pbar.set_postfix({"loss": avg_loss / (len(loader))})

        # 每个 epoch 保存一次
        torch.save({
            "model": model.state_dict()
        }, save_path)
        print(f"[Stage1] Epoch {epoch+1} finished, avg_loss={avg_loss/len(loader):.4f}")

    return model


if __name__ == "__main__":
    # 示例调用，替换成你自己的路径
    train_stage1(
        rgb_dir="/path/to/rgb",
        depth_dir="/path/to/depth",
        save_path="stage1.pth"
    )
