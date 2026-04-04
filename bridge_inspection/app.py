"""
桥梁病害检测 Web 服务：ONNX 分割推理、物理尺寸估算、可选大模型分析。

【大模型 API（OpenAI 兼容 Chat Completions，同一套 /chat/completions）】

  方式 A — DeepSeek（推荐国内线路）：
  export DEEPSEEK_API_KEY="sk-..."         # 在 platform.deepseek.com 申请
  # 可选：
  export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
  export DEEPSEEK_MODEL="deepseek-chat"    # 或 deepseek-reasoner

  方式 B — OpenAI 或其它兼容网关：
  export OPENAI_API_KEY="sk-..."
  export OPENAI_BASE_URL="https://api.openai.com/v1"
  export OPENAI_MODEL="gpt-4o-mini"

  若同时设置两个 Key，请设 BRIDGE_LLM_PROVIDER=deepseek 或 openai；未设置时默认走 DeepSeek。
  DeepSeek 的基址须为 https://api.deepseek.com/v1（仅写域名时程序会自动补 /v1）。

  DeepSeek 一键启动示例：
  DEEPSEEK_API_KEY=sk-xxx uvicorn app:app --host 0.0.0.0 --port 8080

  不想每次 export：将 .env 放在以下任一路径（后者覆盖前者），填入 DEEPSEEK_API_KEY 后重启服务：
    1) 项目根目录（mamba-yolo26/.env）
    2) 当前工作目录下的 .env
    3) bridge_inspection/.env（优先级最高）
  使用 override 加载，可覆盖环境里错误的空变量。详见 /api/health 中 llm.dotenv_files。

【推理与模型】
  BRIDGE_MODEL_ONNX   ONNX 路径（默认 ./weights/best.onnx）
  BRIDGE_ONNX_DEVICE  推理设备：默认 cpu；GPU 用 cuda:0（或 0）。需与 onnxruntime-gpu、CUDA、cuDNN 版本一致。

  【GPU 与 CUDA/cuDNN 版本】
  - PyPI 的 **onnxruntime-gpu 1.18.x** 按 **CUDA 11.8** 链接，需要 **libcublasLt.so.11** 等；仅装
    PyTorch（CUDA 12）时往往只有 **.so.12**，会报 libcublasLt.so.11 缺失。请安装
    requirements-onnx-gpu-cudnn8.txt 中的 **nvidia-cublas-cu11 / nvidia-cuda-runtime-cu11**，并用
    ./run_uvicorn_gpu.sh 启动（会把 site-packages 下 nvidia/*/lib 加入 LD_LIBRARY_PATH）。
  - onnxruntime-gpu **1.24+** 多为 CUDA 12 + **libcudnn.so.9**；与「只有 cuDNN 8」的环境不兼容。
  - cuDNN 8：许多环境由 PyTorch 的 nvidia-cudnn 提供；若 ldd libonnxruntime_providers_cuda.so 仍缺
    cudnn，把对应 lib 目录加入 LD_LIBRARY_PATH。
  - 验证示例：
      ldd "$(python -c "import onnxruntime as o, os; print(os.path.join(os.path.dirname(o.__file__),'capi/libonnxruntime_providers_cuda.so'))")" | egrep 'cublas|cudnn|cudart|not found'

  BRIDGE_YOLO_TASK    默认 segment（本服务为分割场景）；检测模型可设为 detect；设 auto 则不传 task（由 Ultralytics 猜测并可能告警）
  病害类别中文名    在 _DEFECT_NAME_EN_TO_ZH 中维护（hollowareas、rockpocket、drainage 等 -> 中文），表格/绘图/统计一致

启动:
  cd bridge_inspection && pip install -r requirements.txt   # 会 editable 安装上级目录的 ultralytics
  uvicorn app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from measurement import bbox_area_px, estimate_defect_dimensions, mask_raster_area_px

_BRIDGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BRIDGE_DIR.parent


def _load_dotenv_chain() -> None:
    """从多个常见位置加载 .env，且一律 override=True。

    避免 shell/systemd 里空的 DEEPSEEK_API_KEY= 阻止读取文件中的真实密钥。
    后加载的路径覆盖先前的同名变量；bridge_inspection/.env 最后加载，优先级最高。
    """
    paths: list[Path] = [
        _REPO_ROOT / ".env",
        Path.cwd() / ".env",
        _BRIDGE_DIR / ".env",
    ]
    locals_paths: list[Path] = [
        _REPO_ROOT / ".env.local",
        Path.cwd() / ".env.local",
        _BRIDGE_DIR / ".env.local",
    ]
    for p in paths + locals_paths:
        if p.is_file():
            load_dotenv(p, override=True)


def _dotenv_candidates_status() -> list[dict[str, Any]]:
    """供 /api/health 诊断：哪些路径存在 .env（不读取内容）。"""
    out: list[dict[str, Any]] = []
    for p in (
        _REPO_ROOT / ".env",
        Path.cwd() / ".env",
        _BRIDGE_DIR / ".env",
        _REPO_ROOT / ".env.local",
        Path.cwd() / ".env.local",
        _BRIDGE_DIR / ".env.local",
    ):
        rp = p.resolve()
        out.append({"path": str(rp), "exists": p.is_file()})
    return out


_load_dotenv_chain()

logger = logging.getLogger(__name__)

if not any(
    p.is_file()
    for p in (
        _REPO_ROOT / ".env",
        Path.cwd() / ".env",
        _BRIDGE_DIR / ".env",
        _REPO_ROOT / ".env.local",
        Path.cwd() / ".env.local",
        _BRIDGE_DIR / ".env.local",
    )
):
    logger.warning(
        "未找到 .env / .env.local（已检查项目根、当前工作目录、bridge_inspection）；"
        "大模型密钥需写入上述任一路径后重启服务，详见 app.py 顶部说明。"
    )

_ONNX_DEVICE_LOGGED = False


def _resolve_bridge_yolo_task() -> str | None:
    """ONNX 无法可靠自动推断 task，加载时必须显式指定（默认 segment）。"""
    raw = os.environ.get("BRIDGE_YOLO_TASK")
    if raw is None:
        return "segment"
    t = raw.strip()
    if not t:
        return "segment"
    if t.lower() in ("auto", "guess"):
        return None
    return t


def _maybe_warn_onnx_cuda_mismatch() -> None:
    """用户设了 GPU 但 ORT 未注册 CUDA 时给一条可操作的说明（Ultralytics 也会打 WARNING）。"""
    dev = (os.environ.get("BRIDGE_ONNX_DEVICE") or "cpu").strip() or "cpu"
    if dev.lower() in ("cpu", "mps"):
        return
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return
    except Exception:
        return
    try:
        ver = __import__("onnxruntime").__version__
    except Exception:
        ver = "?"
    logger.warning(
        "BRIDGE_ONNX_DEVICE=%r 但 ONNX Runtime %s 未加载 CUDAExecutionProvider，将使用 CPU。"
        " ORT 1.24+ 需 CUDA12+cuDNN9；若只有 cuDNN8 请用 requirements-onnx-gpu-cudnn8.txt + run_uvicorn_gpu.sh，"
        "或设 BRIDGE_ONNX_DEVICE=cpu。",
        dev,
        ver,
    )


def _yolo_predict_extra() -> dict[str, Any]:
    """Ultralytics 默认会对 ONNX 选 GPU；本服务默认强制 cpu，避免 onnxruntime-gpu 缺 cuDNN 时崩溃。"""
    global _ONNX_DEVICE_LOGGED
    dev = os.environ.get("BRIDGE_ONNX_DEVICE", "cpu").strip() or "cpu"
    if not _ONNX_DEVICE_LOGGED:
        logger.info(
            "YOLO/ONNX 推理 device=%r（GPU 需匹配 onnxruntime-gpu 的 CUDA 主版本，见 app.py 顶部说明）",
            dev,
        )
        _ONNX_DEVICE_LOGGED = True
    kw: dict[str, Any] = {"device": dev}
    task = _resolve_bridge_yolo_task()
    if task is not None:
        kw["task"] = task
    return kw


# -----------------------------------------------------------------------------
# 模型（Ultralytics 可直接加载带 NMS 的 ONNX 分割模型）
# -----------------------------------------------------------------------------
_MODEL = None
_MODEL_PATH = Path(os.environ.get("BRIDGE_MODEL_ONNX", _BRIDGE_DIR / "weights" / "best.onnx"))


def get_model():
    global _MODEL
    if _MODEL is None:
        if not _MODEL_PATH.is_file():
            raise RuntimeError(
                f"未找到 ONNX 模型: {_MODEL_PATH.resolve()}\n"
                "请先运行: python export_yolo_moe_onnx.py --weights your/best.pt\n"
                "或将生成 best.onnx 放到 bridge_inspection/weights/ 或设置 BRIDGE_MODEL_ONNX"
            )
        from ultralytics import YOLO

        _task = _resolve_bridge_yolo_task()
        _MODEL = YOLO(str(_MODEL_PATH), task=_task)
        _maybe_warn_onnx_cuda_mismatch()
    return _MODEL


def _bgr_to_png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("图像编码失败")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


# 模型导出的英文类名（大小写不敏感；无空格与有空格写法由 _class_name_to_zh 归一匹配）-> 中文
_DEFECT_NAME_EN_TO_ZH: dict[str, str] = {
    "alligator crack": "龟裂（网状裂缝）",
    "bearing": "支座",
    "cavity": "孔洞",
    "crack": "裂缝",
    "drainage": "排水设施",
    "efflorescence": "泛碱（白华）",
    "expansion joint": "伸缩缝",
    "exposed rebars": "钢筋外露",
    "graffiti": "涂鸦",
    "hollowareas": "空洞区",
    "hollow areas": "空洞区",
    "joint tape": "接缝胶带",
    "protective equipment": "防护设施",
    "restformwork": "模板残留",
    "rest formwork": "模板残留",
    "rockpocket": "石子窝槽（蜂窝）",
    "rock pocket": "石子窝槽（蜂窝）",
    "rust": "锈蚀",
    "spalling": "剥落",
    "washouts/concrete corrosion": "冲刷与混凝土腐蚀",
    "washouts": "冲刷",
    "concrete corrosion": "混凝土腐蚀",
    "weathering": "风化",
    "wetspot": "渗水湿渍",
    "wet spot": "渗水湿渍",
}


def _class_name_to_zh(model_name: str) -> str:
    """将 ONNX/Ultralytics 给出的英文类别名转为中文；未知则保留原文。"""
    s = str(model_name).strip()
    if not s:
        return s
    k = s.lower().replace("_", " ").replace("-", " ")
    if k in _DEFECT_NAME_EN_TO_ZH:
        return _DEFECT_NAME_EN_TO_ZH[k]
    k2 = k.replace(" ", "")
    for en, zh in _DEFECT_NAME_EN_TO_ZH.items():
        if en.replace(" ", "") == k2:
            return zh
    return s


def _normalize_by_class_counts(by_class: dict[str, int]) -> dict[str, int]:
    """按中文类名合并统计（兼容仍带英文 key 的旧缓存或异常路径）。"""
    out: dict[str, int] = {}
    for k, v in by_class.items():
        zk = _class_name_to_zh(str(k))
        out[zk] = out.get(zk, 0) + int(v)
    return out


def _ensure_result_names(r0: Any) -> dict[int, str]:
    """保证 r0.names 为 int->str 字典，且包含当前所有 box 的类别 id。

    Ultralytics 的 Results.plot() 使用 names[c] 下标；ONNX 元数据不完整或 names 为 None
    时会 KeyError / TypeError，进而导致接口 500。
    """
    raw = r0.names
    if raw is None:
        names: dict[int, str] = {}
    elif isinstance(raw, dict):
        names = {}
        for k, v in raw.items():
            try:
                names[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
    else:
        names = {int(i): str(v) for i, v in enumerate(raw)}
    boxes = r0.boxes
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            cid = int(boxes.cls[i].item())
            names.setdefault(cid, f"class_{cid}")
    for cid in names:
        names[cid] = _class_name_to_zh(names[cid])
    r0.names = names
    return names


def _timing_from_r0(r0: Any, server_total_ms: float) -> dict[str, Any]:
    """合并 Ultralytics 报告的各阶段耗时与本服务端到端耗时（毫秒）。"""
    sp = getattr(r0, "speed", None) or {}

    def ms(x: Any) -> float | None:
        if x is None:
            return None
        try:
            return round(float(x), 2)
        except (TypeError, ValueError):
            return None

    return {
        "preprocess_ms": ms(sp.get("preprocess")),
        "inference_ms": ms(sp.get("inference")),
        "postprocess_ms": ms(sp.get("postprocess")),
        "server_total_ms": round(float(server_total_ms), 2),
    }


def _run_predict(
    bgr: np.ndarray,
    conf: float,
    iou: float,
    mm_per_pixel: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    t0 = time.perf_counter()
    model = get_model()
    h0, w0 = bgr.shape[:2]
    results = model.predict(
        source=bgr,
        conf=conf,
        iou=iou,
        verbose=False,
        **_yolo_predict_extra(),
    )
    r0 = results[0]
    names = _ensure_result_names(r0)
    base = np.ascontiguousarray(bgr).copy()
    try:
        plotted_seg = r0.plot(boxes=False, masks=True, labels=False, conf=False)
    except Exception as e:
        logger.warning("r0.plot(分割) 失败，使用原图: %s", e)
        plotted_seg = base.copy()
    try:
        plotted_boxes = r0.plot(boxes=True, masks=False, labels=True, conf=True)
    except Exception as e:
        logger.warning("r0.plot(检测框) 失败，使用原图: %s", e)
        plotted_boxes = base.copy()

    dets: list[dict[str, Any]] = []
    boxes = r0.boxes
    masks = r0.masks
    if boxes is None or len(boxes) == 0:
        wall_ms = (time.perf_counter() - t0) * 1000
        return plotted_seg, plotted_boxes, dets, {"total": 0, "by_class": {}}, _timing_from_r0(r0, wall_ms)

    if masks is not None:
        try:
            xy_list = masks.xy
        except Exception as e:
            logger.warning("masks.xy 失败，跳过多边形尺寸: %s", e)
            xy_list = []
    else:
        xy_list = []
    if len(xy_list) < len(boxes):
        xy_list = list(xy_list) + [None] * (len(boxes) - len(xy_list))

    by_class: dict[str, int] = {}
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        label = names.get(cls_id, str(cls_id))
        confv = float(boxes.conf[i].item())
        by_class[label] = by_class.get(label, 0) + 1

        xy = xy_list[i] if i < len(xy_list) and xy_list[i] is not None else None
        xyxy = boxes.xyxy[i]
        x1, y1, x2, y2 = (float(t) for t in xyxy.tolist())
        bbox_apx = bbox_area_px(x1, y1, x2, y2)

        if xy is not None and len(xy) >= 3:
            # 独立拷贝，避免底层复用同一缓冲区导致多实例量纲串扰
            dim = estimate_defect_dimensions(
                np.asarray(xy, dtype=np.float64).copy(),
                (h0, w0),
                mm_per_pixel,
            )
        else:
            dim = {
                "area_px": 0,
                "length_mm": 0.0,
                "max_width_mm": 0.0,
                "length_px": 0.0,
                "max_width_px": 0.0,
            }

        # 面积（px²）：分割模型优先用 masks.data 栅格像素计数，与 polygon / 框依次回退
        raster_apx = mask_raster_area_px(masks, i)
        poly_apx = int(dim["area_px"])
        if raster_apx is not None and raster_apx > 0:
            area_px = raster_apx
            area_source = "mask_raster"
        elif poly_apx >= 1:
            area_px = poly_apx
            area_source = "mask_polygon"
        elif bbox_apx > 0:
            area_px = bbox_apx
            area_source = "bbox"
        else:
            area_px = 0
            area_source = "none"

        rho = float(mm_per_pixel)
        area_mm2_partial = (rho**2) * area_px if area_px > 0 else 0.0
        # ρ²·A 由像素面积与标定 ρ 直接换算；α 为待定物理修正系数
        area_mm2_expr = f"{area_mm2_partial:.4g}×α mm²" if area_mm2_partial > 0 else "ρ²·A×α"

        poly = []
        if xy is not None:
            poly = np.asarray(xy, dtype=np.float64).tolist()
        dim_out = {**dim, "area_px": area_px}
        row = {
            "id": i,
            "class_id": cls_id,
            "class_name": label,
            "confidence": round(confv, 4),
            "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "polygon_xy": poly,
            "area_source": area_source,
            "area_mm2_partial": round(area_mm2_partial, 6) if area_mm2_partial > 0 else None,
            "area_mm2_expr": area_mm2_expr,
            **dim_out,
        }
        dets.append(jsonable_encoder(row))

    summary = {"total": len(dets), "by_class": _normalize_by_class_counts(by_class)}
    wall_ms = (time.perf_counter() - t0) * 1000
    return plotted_seg, plotted_boxes, dets, summary, _timing_from_r0(r0, wall_ms)


# -----------------------------------------------------------------------------
# LLM Agent（OpenAI 兼容）
# -----------------------------------------------------------------------------
class AgentRequest(BaseModel):
    detections: list[dict[str, Any]]
    summary: dict[str, Any] | None = None
    mm_per_pixel: float = 1.0
    extra_context: str | None = None
    user_question: str | None = None
    prior_assistant_reply: str | None = None


# 桥梁病害分析专家 Agent：固定短模板，控制篇幅（适用于 DeepSeek / OpenAI 兼容接口）
BRIDGE_AGENT_SYSTEM_PROMPT = """你是「桥梁病害分析专家」。用户 JSON 为无人机分割+检测结果：detections 中 area_px 优先为**分割掩膜前景像素数**，area_source 标明来源（mask_raster/mask_polygon/bbox）；Sm≈ρ²·A×α。ρ、α 与算法均有误差，**不得当作精密量测**。

【输出格式 — 必须严格遵守，勿增删标题】
全文 **不超过 380 个汉字**（不含标点可略宽），不用寒暄、不写「综上所述」、不重复题干。仅用下面 4 行标题起头，每标题下 **最多 2 条短句或短分号句**，单条建议 **不超过 35 字**。

1. 【结论】……（1 句：检出规模 + 风险倾向）
2. 【要点】……；……（按类别/高置信度优先，低置信度注明不确定）
3. 【建议】……；……（可执行：复核/加密巡检/专项检测/荷载或交通控制等，二选一即可）
4. 【局限】……（算法与标定局限 + 须现场复核，1 句即可）

【硬性规则】
- 数字、类别须与 JSON 一致，**禁止编造**未给出的病害或数值。
- 无检出或极少：【结论】写可能原因；【建议】写下一步巡检，勿写「结构完好」。
- extra 有桥型等上下文时，【要点】或【建议】中 **至少体现 1 处**。
- 若 JSON 含 user_question：在四条目内**针对性回应**该追问，仍须与 detections 一致、勿编造。
- 若含 prior_assistant_reply：视为上一轮回复摘要，避免重复空话，承接追问。
- 勿写规范条款号；勿 Markdown 大段、勿列表套娃。"""


def _env_secret(name: str) -> str:
    """读取密钥类环境变量，去掉首尾空白、引号与 Windows 换行。"""
    v = os.environ.get(name)
    if v is None:
        return ""
    v = v.strip().strip("'\"")
    if v.startswith("\ufeff"):
        v = v.lstrip("\ufeff")
    return v.replace("\r", "").strip()


def _normalize_llm_base_url(base: str, *, for_deepseek: bool) -> str:
    """保证 OpenAI 兼容 URL 以 .../v1 结尾（再拼接 /chat/completions）。"""
    b = base.strip().rstrip("/")
    if not b:
        return b
    if for_deepseek or "deepseek.com" in b.lower():
        if not b.endswith("/v1"):
            b = b + "/v1"
    return b.rstrip("/")


def _llm_config() -> tuple[str, str, str, str]:
    """返回 (api_key, base_url, model, provider)。

    provider: \"openai\" | \"deepseek\" | \"none\"

    - 可设置 BRIDGE_LLM_PROVIDER=deepseek 或 openai，避免同时存在两个 Key 时走错线路。
    - 仅配置一个 Key 时自动选用对应服务商。
    - 若同时存在两个 Key 且未指定 BRIDGE_LLM_PROVIDER，默认使用 DeepSeek（常见误配是 .env 里留了占位的 OPENAI_API_KEY）。
    """
    oa_key = _env_secret("OPENAI_API_KEY")
    ds_key = _env_secret("DEEPSEEK_API_KEY")
    pref = _env_secret("BRIDGE_LLM_PROVIDER").lower()

    def openai_cfg() -> tuple[str, str, str, str]:
        base = _normalize_llm_base_url(
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            for_deepseek=False,
        )
        model = _env_secret("OPENAI_MODEL") or "gpt-4o-mini"
        return oa_key, base, model, "openai"

    def deepseek_cfg() -> tuple[str, str, str, str]:
        base = _normalize_llm_base_url(
            os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            for_deepseek=True,
        )
        model = _env_secret("DEEPSEEK_MODEL") or "deepseek-chat"
        return ds_key, base, model, "deepseek"

    if pref in ("deepseek", "ds"):
        if not ds_key:
            logger.warning("BRIDGE_LLM_PROVIDER=deepseek 但未配置有效的 DEEPSEEK_API_KEY")
            return "", "", "", "none"
        return deepseek_cfg()
    if pref in ("openai", "oa"):
        if not oa_key:
            logger.warning("BRIDGE_LLM_PROVIDER=openai 但未配置有效的 OPENAI_API_KEY")
            return "", "", "", "none"
        return openai_cfg()

    if oa_key and ds_key:
        logger.warning(
            "同时存在 OPENAI_API_KEY 与 DEEPSEEK_API_KEY，未设置 BRIDGE_LLM_PROVIDER 时默认使用 DeepSeek；"
            "若要用 OpenAI 请设置 BRIDGE_LLM_PROVIDER=openai"
        )
        return deepseek_cfg()
    if ds_key:
        return deepseek_cfg()
    if oa_key:
        return openai_cfg()
    return "", "", "", "none"


async def _call_openai_compatible(messages: list[dict[str, str]]) -> str:
    import httpx

    key, base, model, provider = _llm_config()
    if not key:
        return _fallback_agent_text(messages)
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 650,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json=payload, headers=headers)
    except httpx.RequestError as e:
        logger.exception("LLM 网络请求失败 provider=%s url=%s", provider, url)
        return f"大模型网络错误（{provider}）: {e}"

    if r.status_code >= 400:
        logger.warning("LLM HTTP %s provider=%s url=%s body=%s", r.status_code, provider, url, r.text[:300])
        return f"大模型请求失败 ({r.status_code}): {r.text[:800]}"

    try:
        data = r.json()
        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        return str(content).strip()
    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.warning("LLM 响应 JSON 非预期: %s raw=%s", e, r.text[:500])
        return f"大模型返回格式异常: {e}；原始片段: {r.text[:400]}"


def _fallback_agent_text(messages: list[dict[str, str]]) -> str:
    """未配置 API Key 时返回结构化本地建议。"""
    import json

    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "{}")
    try:
        payload = json.loads(user)
    except json.JSONDecodeError:
        payload = {}
    dets = payload.get("detections") or []
    lines = [
        "（当前未配置大模型 API 密钥，以下为规则化摘要；配置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 后可启用「桥梁病害分析专家」。）",
        "",
        f"检出实例数: {len(dets)}",
    ]
    def _fmt_mm(v: Any, sym: str) -> str:
        try:
            x = float(v)
            if x > 0:
                return f"{x} mm"
        except (TypeError, ValueError):
            pass
        return sym

    def _fmt_px(v: Any, sym: str) -> str:
        try:
            x = int(float(v))
            if x > 0:
                return f"{x} px"
        except (TypeError, ValueError):
            pass
        return sym

    for d in dets[:20]:
        sm = d.get("area_mm2_expr") or "α·ρ²·A"
        lines.append(
            f"- {d.get('class_name')}: 置信度 {d.get('confidence')}, "
            f"长度 {_fmt_mm(d.get('length_mm'), 'L')}, 最大宽 {_fmt_mm(d.get('max_width_mm'), 'W')}, "
            f"面积 {_fmt_px(d.get('area_px'), 'A')}, Sm {sm}"
        )
    if len(dets) > 20:
        lines.append(f"... 其余 {len(dets) - 20} 条已省略")
    lines.extend(
        [
            "",
            "建议: 对高置信度裂缝/露筋等优先复核；结合现场荷载与耐久性设计规范安排定期巡检或封闭交通评估。",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="桥梁病害检测服务", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = Path(__file__).parent / "static"
if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/")
async def index():
    from fastapi.responses import FileResponse

    p = STATIC / "index.html"
    if not p.is_file():
        return JSONResponse(
            {"error": "缺少 static/index.html", "static_dir": str(STATIC)},
            status_code=500,
        )
    return FileResponse(p)


@app.get("/api/health")
async def health():
    ok = _MODEL_PATH.is_file()
    _key, base, model, provider = _llm_config()
    dotenv_files = _dotenv_candidates_status()
    any_dotenv = any(x["exists"] for x in dotenv_files)
    hints: list[str] = []
    if not _key:
        hints.append("变量名须为 DEEPSEEK_API_KEY 或 OPENAI_API_KEY（区分大小写）。")
        hints.append("密钥写在 .env 内时不要加 export 也可；一行示例：DEEPSEEK_API_KEY=sk-xxxx")
        if not any_dotenv:
            hints.append("当前未发现任何 .env 文件，请放在项目根、bridge_inspection 或启动时的当前工作目录。")
        elif not _env_secret("DEEPSEEK_API_KEY") and not _env_secret("OPENAI_API_KEY"):
            hints.append("已存在 .env 文件但未解析出密钥：检查是否写错变量名、是否用了中文引号、或 Key 与等号之间有空格。")
        hints.append("修改 .env 后必须重启 uvicorn 进程。")
    return {
        "model_path": str(_MODEL_PATH.resolve()),
        "model_exists": ok,
        "llm": {
            "configured": bool(_key),
            "provider": provider,
            "base_url": base,
            "model": model,
            "chat_completions_url": f"{base}/chat/completions" if _key and base else None,
            "dotenv_files": dotenv_files,
            "hints": hints,
        },
    }


@app.post("/api/predict_image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.7),
    mm_per_pixel: float = Form(0.5),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "空文件")
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "无法解码图像（需 jpg/png 等）")
    try:
        plotted_seg, plotted_boxes, dets, summary, timing = _run_predict(bgr, conf, iou, mm_per_pixel)
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        logger.exception("predict_image")
        raise HTTPException(500, f"推理失败: {e}") from e
    try:
        return {
            "image_seg_png_base64": _bgr_to_png_b64(plotted_seg),
            "image_boxes_png_base64": _bgr_to_png_b64(plotted_boxes),
            "detections": dets,
            "summary": summary,
            "mm_per_pixel": float(mm_per_pixel),
            "timing": timing,
        }
    except Exception as e:
        logger.exception("predict_image encode/serialize")
        raise HTTPException(500, f"结果编码失败: {e}") from e


@app.post("/api/predict_video")
async def predict_video(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.7),
    mm_per_pixel: float = Form(0.5),
    max_frames: int = Form(24),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "空文件")
    max_frames = max(1, min(int(max_frames), 120))
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            path = tmp.name
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not cap.isOpened():
            raise HTTPException(400, "无法打开视频")
        indices: list[int] = []
        if n > 0:
            step = max(1, n // max_frames)
            for i in range(0, n, step):
                indices.append(i)
                if len(indices) >= max_frames:
                    break
        else:
            indices = list(range(max_frames))

        frames_out: list[dict[str, Any]] = []
        all_by_class: dict[str, int] = {}
        total_instances = 0
        frame_timings: list[dict[str, Any]] = []

        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            try:
                plotted_seg, plotted_boxes, dets, summary, timing = _run_predict(frame, conf, iou, mm_per_pixel)
            except RuntimeError as e:
                cap.release()
                os.unlink(path)
                raise HTTPException(503, str(e)) from e
            except Exception as e:
                cap.release()
                os.unlink(path)
                logger.exception("predict_video")
                raise HTTPException(500, f"推理失败: {e}") from e
            total_instances += summary["total"]
            for k, v in summary["by_class"].items():
                zk = _class_name_to_zh(str(k))
                all_by_class[zk] = all_by_class.get(zk, 0) + int(v)
            frame_timings.append(timing)
            frames_out.append(
                {
                    "frame_index": fi,
                    "time_sec": round(fi / fps, 3) if fps else None,
                    "image_seg_png_base64": _bgr_to_png_b64(plotted_seg),
                    "image_boxes_png_base64": _bgr_to_png_b64(plotted_boxes),
                    "detections": dets,
                    "summary": summary,
                    "timing": timing,
                }
            )

        cap.release()
        os.unlink(path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e)) from e

    agg: dict[str, Any] = {}
    if frame_timings:
        inf = [t["inference_ms"] for t in frame_timings if t.get("inference_ms") is not None]
        srv = [t["server_total_ms"] for t in frame_timings]
        if inf:
            agg["sum_inference_ms"] = round(sum(inf), 2)
            agg["mean_inference_ms"] = round(sum(inf) / len(inf), 2)
        if srv:
            agg["sum_server_ms"] = round(sum(srv), 2)
            agg["mean_server_ms"] = round(sum(srv) / len(srv), 2)
        agg["frames_timed"] = len(frame_timings)

    return {
        "frames": frames_out,
        "video_summary": {
            "sampled_frames": len(frames_out),
            "total_instances": total_instances,
            "by_class": all_by_class,
        },
        "mm_per_pixel": mm_per_pixel,
        "timing": agg,
    }


@app.post("/api/agent")
async def agent_analyze(body: AgentRequest):
    import json

    t0 = time.perf_counter()
    user_payload: dict[str, Any] = {
        "detections": body.detections,
        "summary": body.summary,
        "mm_per_pixel": body.mm_per_pixel,
        "extra": body.extra_context,
    }
    if body.user_question and str(body.user_question).strip():
        user_payload["user_question"] = str(body.user_question).strip()
    if body.prior_assistant_reply and str(body.prior_assistant_reply).strip():
        user_payload["prior_assistant_reply"] = str(body.prior_assistant_reply).strip()[:1200]
    messages = [
        {"role": "system", "content": BRIDGE_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    text = await _call_openai_compatible(messages)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {"analysis": text, "timing_ms": elapsed_ms}
