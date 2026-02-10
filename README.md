# TO-ProgressiveMAE
````markdown
# TO-ProgressiveMAE: 透明物体深度补全预训练框架

本仓库实现了论文中提出的 **TO-ProgressiveMAE** 两阶段 RGB-D 预训练框架，用于提升 **透明物体深度补全** 的精度与泛化能力。框架包含：

- **Stage 1：Refraction-aware Contrastive Alignment**  
  利用折射显著性图（refraction saliency）对 RGB / 深度 patch 特征进行加权 InfoNCE 对比学习，对齐跨模态特征。
- **Stage 2：Transparency-aware Masked Autoencoding**  
  使用折射显著性引导的非均匀遮挡策略进行 MAE 预训练，同时联合：
  - 深度重建
  - RGB 语义特征重建
  - 去噪任务
  - 跨阶段特征蒸馏

支持在 **ClearGrasp** 与 **Trans10K** 数据集上进行预训练和验证，内置评价指标：**RMSE / MAE / REL / δ1**，并可自动生成定性可视化结果。

---

## 1. 目录结构

建议在你的工程根目录保持如下结构（以 `pythonProject1` 为例）：

        └── 深度补全预训练模型\
            └── pythonProject1\
                ├── to_progressivemae\
                │   ├── models.py
                │   ├── refraction_saliency.py
                │   ├── losses.py
                │   ├── utils.py
                │   ├── datasets.py
                │   ├── metrics.py
                │   ├── train_stage1.py
                │   └── trainval_stage2.py
                └── databases\
                    ├── ClearGrasp\
                    │   ├── train\
                    │   │   ├── rgb\
                    │   │   └── depth\
                    │   └── val\
                    │       ├── rgb\
                    │       └── depth\
                    └── Trans10K\
                        ├── train\
                        │   ├── rgb\
                        │   └── depth\
                        └── val\
                            ├── rgb\
                            └── depth\
````

> 如有不同数据结构，只需相应修改 `datasets.py` 中的路径拼接部分即可。

---

## 2. 环境配置

建议使用 Conda 创建独立环境（以 Python 3.10 为例）：

```bash
conda create -n to_progressivemae python=3.10
conda activate to_progressivemae
```

安装依赖：

```bash
pip install torch torchvision
pip install opencv-python pillow numpy tqdm matplotlib
```

> PyTorch 安装方式可根据你的 CUDA 版本从官网拷贝命令。

---

## 3. 数据准备

默认为以下目录结构：

```text
databases/
├── ClearGrasp/
│   ├── train/
│   │   ├── rgb/   # 训练集 RGB 图像 (*.png / *.jpg)
│   │   └── depth/ # 训练集深度图 (*.npy 或 16bit *.png)
│   └── val/
│       ├── rgb/   # 验证集 RGB 图像
│       └── depth/ # 验证集深度图
└── Trans10K/
    ├── train/
    │   ├── rgb/
    │   └── depth/
    └── val/
        ├── rgb/
        └── depth/
```

* RGB 图像建议为 `uint8` 3 通道 `.png` 或 `.jpg`。
* 深度可以是：

  * `float32` 的 `.npy` 文件（单位自定，可按米 / 毫米但需保持一致），或
  * 16bit `.png`，在 `datasets.py` 中会自动 `/1000.0` 转换为米（可根据实际修改）。

文件命名约定：

* 每个样本一个 ID，例如 `0001`，对应：

  * `rgb/0001.png`
  * `depth/0001.npy` 或 `depth/0001.png`

---

## 4. Stage 1：对比预训练

`train_stage1.py` 实现了折射显著性感知的跨模态对比学习（加权 InfoNCE）。

### 4.1 主要功能

* 从 `ClearGrasp` 或 `Trans10K` 训练集读取 RGB-D。
* 利用 `refraction_saliency.py` 生成折射显著性图 `S_ref`。
* 使用 `TOProgressiveMAEStage1` 模型进行 patch-level 对比预训练。
* 保存 `stage1.pth` 作为 Stage 2 的初始化与蒸馏教师。

### 4.2 运行示例

在 `to_progressivemae` 目录下（或通过绝对路径）运行：

```bash
cd E:\论文\我的\深度补全预训练模型\pythonProject1\to_progressivemae

python train_stage1.py
```

脚本内部默认类似：

```python
train_stage1(
    rgb_dir=r"E:\论文\我的\深度补全预训练模型\pythonProject1\databases\ClearGrasp\train\rgb",
    depth_dir=r"E:\论文\我的\深度补全预训练模型\pythonProject1\databases\ClearGrasp\train\depth",
    save_path="stage1.pth",
    batch_size=4,
    lr=1e-4,
    epochs=50,
    device="cuda"
)
```

你可以根据显存情况自行调整 `batch_size` 和 `epochs`。

运行结束后，会在当前目录生成：

```text
stage1.pth
```

---

## 5. Stage 2：MAE 预训练 + 验证 + 可视化

`trainval_stage2.py` 实现了：

* 在 Stage1 初始化基础上进行透明感知 MAE 预训练；
* 在 `ClearGrasp` / `Trans10K` 验证集上计算指标；
* 自动导出可视化对比图（RGB / GT 深度 / 预测深度 / 误差图）。

### 5.1 评价指标

在 `metrics.py` 中实现了常用深度误差指标，均在有效深度像素（`depth>0` 或有效 mask）上计算：

* **MAE**：
  [
  \text{MAE} = \frac{1}{N} \sum |d_{\text{pred}} - d_{\text{gt}}|
  ]
* **RMSE**：
  [
  \text{RMSE} = \sqrt{\frac{1}{N} \sum (d_{\text{pred}} - d_{\text{gt}})^2}
  ]
* **REL**（Absolute Relative Error）：
  [
  \text{REL} = \frac{1}{N}\sum \frac{|d_{\text{pred}} - d_{\text{gt}}|}{d_{\text{gt}}}
  ]
* **δ1**：
  比例满足
  [
  \max\left(\frac{d_{\text{pred}}}{d_{\text{gt}}}, \frac{d_{\text{gt}}}{d_{\text{pred}}}\right) < 1.25
  ]
  的像素数占比。

### 5.2 训练 + 验证示例（以 ClearGrasp 为例）

在 `trainval_stage2.py` 的 `__main__` 中已经给出了示例：

```python
if __name__ == "__main__":
    base_root = r"E:\论文\我的\深度补全预训练模型\pythonProject1\databases"
    stage1_path = r"E:\论文\我的\深度补全预训练模型\pythonProject1\to_progressivemae\stage1.pth"

    train_stage2_with_val(
        base_root=base_root,
        stage1_path=stage1_path,
        save_path="stage2.pth",
        dataset_name="ClearGrasp",
        batch_size=4,
        lr=1e-4,
        epochs=50,
        num_workers=4,
        device="cuda",
        val_interval=1,
        out_dir=r"E:\论文\我的\深度补全预训练模型\pythonProject1\to_progressivemae\outputs_cleargrasp"
    )
```

直接运行：

```bash
cd E:\论文\我的\深度补全预训练模型\pythonProject1\to_progressivemae

python trainval_stage2.py
```

训练过程中会在控制台打印类似：

```text
[Stage2 Train] Epoch 1/50: loss=..., mae_loss=..., distill=...
Validation Results: RMSE=..., MAE=..., REL=..., δ1=...
==> New best RMSE 0.0xxx, saved to ...\stage2_best.pth
```

### 5.3 在 Trans10K 上训练 / 验证

同样可以在 `__main__` 中再加一段调用（或者单独开一个脚本）：

```python
train_stage2_with_val(
    base_root=base_root,
    stage1_path=stage1_path,
    save_path="stage2_trans10k.pth",
    dataset_name="Trans10K",
    batch_size=4,
    lr=1e-4,
    epochs=50,
    num_workers=4,
    device="cuda",
    val_interval=1,
    out_dir=r"E:\论文\我的\深度补全预训练模型\pythonProject1\to_progressivemae\outputs_trans10k"
)
```

---

## 6. 可视化结果

`trainval_stage2.py` 在每个验证 epoch 后，会将第一批样本的可视化结果保存在：

```text
out_dir/
 ├── stage2_epoch1.pth
 ├── stage2_epoch2.pth
 ├── stage2_best.pth
 ├── vis_epoch1/
 │   ├── ClearGrasp_epoch1_sample_0.png
 │   ├── ClearGrasp_epoch1_sample_1.png
 │   └── ...
 ├── vis_epoch2/
 │   └── ...
 └── ...
```

单张图包含 4 列：

1. 原始 RGB 图像
2. Ground Truth 深度图（GT Depth）
3. 模型预测深度图（Pred Depth）
4. 绝对误差热力图（Abs Error）

这些图片可以直接用于论文中的“定性结果对比”部分。


```

---

