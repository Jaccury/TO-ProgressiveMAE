# metrics.py
import torch


def _valid_mask_from_gt(gt, eps=1e-8):
    """
    若外部没传 mask，则根据 gt>0 自动生成。
    """
    return (gt > eps).float()


def compute_mae(pred, gt, mask=None):
    """
    pred, gt: (B,1,H,W)
    mask: (B,1,H,W) 1=有效
    return: 标量
    """
    if mask is None:
        mask = _valid_mask_from_gt(gt)
    diff = torch.abs(pred - gt) * mask
    denom = mask.sum() + 1e-6
    return diff.sum() / denom


def compute_rmse(pred, gt, mask=None):
    if mask is None:
        mask = _valid_mask_from_gt(gt)
    diff2 = (pred - gt) ** 2 * mask
    denom = mask.sum() + 1e-6
    return torch.sqrt(diff2.sum() / denom)


def compute_rel(pred, gt, mask=None, eps=1e-6):
    if mask is None:
        mask = _valid_mask_from_gt(gt)
    diff = torch.abs(pred - gt) / (gt.abs() + eps)
    diff = diff * mask
    denom = mask.sum() + 1e-6
    return diff.sum() / denom


def compute_delta1(pred, gt, mask=None, threshold=1.25, eps=1e-6):
    """
    δ1: max(pred/gt, gt/pred) < 1.25 的比例
    """
    if mask is None:
        mask = _valid_mask_from_gt(gt)
    mask_bool = mask.bool()

    pred_valid = pred[mask_bool]
    gt_valid = gt[mask_bool]

    ratio1 = pred_valid / (gt_valid + eps)
    ratio2 = gt_valid / (pred_valid + eps)
    ratio = torch.max(ratio1, ratio2)
    good = (ratio < threshold).float()
    return good.mean()
