#!/usr/bin/env python3
"""将训练好的 YOLO-MoE 分割权重导出为 ONNX（内置 NMS，便于部署）。

用法示例:
  python export_yolo_moe_onnx.py --weights /path/to/best.pt --imgsz 640
  python export_yolo_moe_onnx.py --weights best.pt --half  # GPU 半精度导出

依赖: pip install "ultralytics[export]"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Export YOLO-MoE segmentation .pt to ONNX")
    p.add_argument("--weights",default="/root/mamba-yolo26/out_dir/yolo26/yolo-mamba30/weights/best.pt", type=str, help="训练得到的 best.pt 路径")
    p.add_argument("--imgsz", type=int, default=640, help="导出 ONNX 的输入边长（与训练/推理一致为佳）")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset")
    p.add_argument("--half", action="store_true", help="FP16 导出（需 GPU）")
    p.add_argument("--simplify", action="store_true", default=True, help="onnxsim 简化（默认开）")
    p.add_argument("--no-simplify", dest="simplify", action="store_false")
    p.add_argument("--dynamic", action="store_true", help="动态 batch/空间尺寸（部分环境兼容性差，默认关）")
    args = p.parse_args()

    w = Path(args.weights)
    if not w.is_file():
        raise SystemExit(f"找不到权重文件: {w.resolve()}")

    model = YOLO(str(w))
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        half=args.half,
        simplify=args.simplify,
        nms=True,
        dynamic=args.dynamic,
    )
    print(f"导出完成: {out}")


if __name__ == "__main__":
    main()
