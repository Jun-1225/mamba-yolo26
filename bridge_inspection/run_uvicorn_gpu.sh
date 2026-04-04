#!/usr/bin/env bash
# 在 bridge_inspection 目录启动 Web 服务，并尝试使用 GPU 跑 ONNX（Ultralytics）。
# 依赖（二选一）：
#   - CUDA 12：requirements-onnx-gpu-cuda12.txt，且 onnxruntime-gpu 须从官方 pypi.org 安装（避免镜像误装 cu11 轮子）
#   - CUDA 11 + cuDNN8：requirements-onnx-gpu-cudnn8.txt（ORT 1.18）

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

export BRIDGE_ONNX_DEVICE="${BRIDGE_ONNX_DEVICE:-cuda:0}"

# 必须与「装 pip 包、跑 uvicorn」是同一个解释器。conda 下常见 bug：python3 指向系统 Python，
# 而 onnxruntime / nvidia-cudnn 装在 conda 的 python 里 → LD_LIBRARY_PATH 指错目录 → libcudnn.so.9 not found。
if [[ -n "${PYTHON:-}" ]] && command -v "$PYTHON" >/dev/null 2>&1; then
  PY="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

# 把当前解释器 site-packages 下所有 nvidia/*/lib（含 cudnn）加入 LD_LIBRARY_PATH
SITE_LIBS="$("$PY" <<'PY'
import glob
import os
import site
from pathlib import Path

roots = []
for r in site.getsitepackages():
    if r and os.path.isdir(r):
        roots.append(r)
u = site.getusersitepackages()
if u and os.path.isdir(u):
    roots.append(u)
seen: list[str] = []
for root in roots:
    for lib in glob.glob(os.path.join(root, "nvidia", "*", "lib")):
        if os.path.isdir(lib) and lib not in seen:
            seen.append(lib)
    # 显式补上 cudnn（个别环境 glob 顺序或挂载问题）
    cudnn = Path(root) / "nvidia" / "cudnn" / "lib"
    if cudnn.is_dir():
        s = str(cudnn)
        if s not in seen:
            seen.append(s)
print(os.pathsep.join(seen), end="")
PY
)"
if [[ -n "$SITE_LIBS" ]]; then
  export LD_LIBRARY_PATH="${SITE_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8080 "$@"
