"""基于实例分割掩膜估算病害在像平面上的特征尺寸，并结合 mm/px 换算为物理量。"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def mask_raster_area_px(masks: Any, index: int) -> int | None:
    """从分割头输出的二值掩膜张量统计第 index 个实例的前景像素数（与检测框行序一致时最可靠）。

    Ultralytics `Results.masks.data` 形状为 (N, H, W)。
    """
    if masks is None:
        return None
    try:
        d = getattr(masks, "data", None)
        if d is None:
            return None
        if hasattr(d, "detach"):
            d = d.detach()
        if hasattr(d, "cpu"):
            d = d.cpu().numpy()
        elif not isinstance(d, np.ndarray):
            d = np.asarray(d)
        if d.ndim != 3:
            return None
        if index < 0 or index >= d.shape[0]:
            return None
        layer = d[index]
        return int(np.sum(layer > 0.5))
    except Exception:
        return None


def bbox_area_px(x1: float, y1: float, x2: float, y2: float) -> int:
    """检测框在像平面上的面积（像素²），无分割掩膜时作面积回退。"""
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    return int(round(w * h))


def _polygon_to_mask(xy: np.ndarray, height: int, width: int) -> np.ndarray:
    """xy: (N, 2) 像素坐标，列为 x, y。"""
    mask = np.zeros((height, width), dtype=np.uint8)
    if xy is None or len(xy) < 3:
        return mask
    cnt = np.round(xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [cnt], 1)
    return mask


def _pca_extents_px(points: np.ndarray) -> tuple[float, float]:
    """对点集做 PCA，返回沿第一/第二主轴的投影跨度（像素）。"""
    if points.shape[0] < 3:
        return 0.0, 0.0
    c = points.astype(np.float64)
    c -= c.mean(axis=0, keepdims=True)
    cov = np.cov(c.T)
    if cov.size <= 1:
        return 0.0, 0.0
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    v0 = evecs[:, order[0]]
    v1 = evecs[:, order[1]] if cov.shape[0] > 1 else np.array([0.0, 0.0])
    p0 = c @ v0
    p1 = c @ v1
    major = float(p0.max() - p0.min())
    minor = float(p1.max() - p1.min()) if cov.shape[0] > 1 else 0.0
    return major, minor


def estimate_defect_dimensions(
    xy: np.ndarray,
    orig_shape: tuple[int, int],
    mm_per_pixel: float,
    *,
    max_samples: int = 8000,
) -> dict:
    """根据多边形轮廓（原图坐标）估算病害长度与最大宽度（毫米）。

    - **长度 (length_mm)**: 掩膜内采样点的 PCA 主轴跨度，适合细长裂缝与块状剥落的主延展方向。
    - **最大宽度 (max_width_mm)**: 距离变换得到的最大内切圆直径（2*max(dt)），反映掩膜最厚处，
      对裂缝可近似为最大缝宽；对坑槽可反映横向尺度。

    Args:
        xy: (N, 2)，与原图一致的像素多边形。
        orig_shape: (H, W) 原图高宽。
        mm_per_pixel: 像元物理分辨率（毫米/像素），由各向同性 GSD 或标定得到。
    """
    h, w = orig_shape
    if mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel 必须为正数")

    mask = _polygon_to_mask(xy, h, w)
    area_px = int(mask.sum())
    # 栅格掩膜为 0 时，用轮廓几何面积（像素²）回退，避免多边形有效但未扫到整像素的情况
    if area_px < 1 and len(xy) >= 3:
        cnt = np.round(xy.astype(np.float64)).astype(np.float32).reshape(-1, 1, 2)
        ca = abs(float(cv2.contourArea(cnt)))
        if ca >= 1.0:
            area_px = int(round(ca))
    if area_px < 3:
        return {
            "area_px": area_px,
            "length_mm": 0.0,
            "max_width_mm": 0.0,
            "length_px": 0.0,
            "max_width_px": 0.0,
        }

    ys, xs = np.where(mask > 0)
    pts = np.column_stack((xs, ys))
    if pts.shape[0] > max_samples:
        idx = np.random.choice(pts.shape[0], max_samples, replace=False)
        pts = pts[idx]

    major_px, minor_px = _pca_extents_px(pts)
    length_px = max(major_px, minor_px)

    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_r = float(dt.max())
    max_width_px = 2.0 * max_r

    return {
        "area_px": area_px,
        "length_px": round(length_px, 2),
        "max_width_px": round(max_width_px, 2),
        "length_mm": round(length_px * mm_per_pixel, 2),
        "max_width_mm": round(max_width_px * mm_per_pixel, 2),
    }
